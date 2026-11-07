"""The conversation loop: sends messages to the model and runs the tools it calls."""

import shutil
import sys

from ui import style
from api.api_client import Client
from ui.markdown import MarkdownStream
from api.ollama_client import OllamaClient
from api.models import MODELS, Usage, response_usage
from core.redact import Redactor
from core.tools import TOOLS, run_tool
from core.workspace import ToolError, Workspace

MAX_TOKENS = 16000
# Prompt caching: the marker on the system prompt caches the tool definitions
# and system prompt (tools come first in the request), and the top-level
# marker caches the conversation so far. Each call then pays full price only
# for what is new since the previous call. This only works while the history
# is append-only, so earlier messages must never be edited.
CACHE_CONTROL = {"type": "ephemeral"}

# Tools that change files. Refused by name in read-only messages (/talk, /plan)
# so the tool list itself can stay identical in every request.
WRITE_TOOLS = {"write_file", "edit_file", "append_file"}

# One prompt for both providers (the Claude Messages API and Ollama Cloud's GLM
# models), so it stays provider-agnostic and sectioned in Markdown, the format
# both model families follow well. It names every tool with its working rules
# (the full schemas travel in the request as the tool definitions, so this says
# when to reach for each tool rather than repeating their parameters), then the
# sandbox rules, the working method, and the reply style. It sits behind the
# cache_control marker together with the tool definitions, so changing it here
# invalidates the prompt cache once.
SYSTEM_PROMPT = """You are a coding assistant working inside a single project directory. Your tools are the \
only way you can see or change anything there: you cannot run commands, execute code, run \
tests, install packages, or access the network. When a task would normally require running \
something (tests, builds, scripts), make the file changes and tell the user what to run \
themselves.

## The tools

- list_directory(path): the files and subdirectories of one directory ('.' is the project root; \
directories end with '/'). Start unfamiliar work here rather than guessing what exists.
- read_file(path, offset, limit): one page of a text file. Reads are capped at 1 MB of text: \
larger files are cut at a line boundary and end with a note naming the line range shown and \
the offset to continue from. offset is the 1-based first line to read, and line numbers here \
match search results.
- search(pattern, path, ignore_case, max_results): regex search of file contents without \
reading whole files; results are 'path:line: text'. Use it to locate text in large files \
instead of reading them whole, and narrow the pattern or the path when the results hit a \
cap. A search does not count as the read_file needed before editing a file.
- edit_file(path, old_text, new_text): replace the one exact occurrence of old_text with \
new_text. Prefer it for changes to an existing file: only the snippet travels in the call, \
so it costs far fewer output tokens than write_file. Copy old_text exactly from a fresh \
read_file, indentation included — the edit is refused if it appears zero times or more \
than once. new_text may be empty, deleting the snippet.
- write_file(path, content): create a file, or replace an existing one completely. Use it \
for new files and full rewrites; for anything else edit_file is cheaper.
- append_file(path, content): add content to the end of a file, creating it if needed. \
Build a long new file in several append_file chunks (roughly 150 lines per call) rather \
than one huge write_file. Parent directories are created as needed by any write.

## Rules of the workspace

- All paths are relative to the project root.
- The .git directory is protected: no tool can read or write anything inside it, so never \
propose changes there.
- You must read an existing file before editing or overwriting it; it is then saved back in \
its original encoding and line endings, so if read_file says a file is not UTF-8, only use \
characters that encoding can represent.
- Binary files (for example .xlsx or .dll) cannot be read or written, and one write is \
limited to 1 MB of content.
- The user must approve every write; if a write is declined, ask what they would like instead.

## Working method

- Explore before acting: list, search, and read rather than guessing at paths or file contents.
- Make several independent read-only calls (list_directory, read_file, search) in one \
response, but never combine a large write with other tool calls in the same response: \
your response is cut off at 16,000 output tokens, and a tool call that is cut off is \
never run.
- A failed tool call changes nothing, and its error message says why: fix the input (correct \
the path, re-read for an exact old_text, adjust the offset) and try again. Never tell the \
user a change succeeded unless the tool result says so.
- A bracketed note at the start of a user message (read-only, plan, plan approved) sets the \
mode for that message; follow it exactly.
- If a request is ambiguous, ask a short clarifying question; otherwise make a reasonable \
choice and say what you assumed.

## Replies

Replies are rendered as Markdown. Keep them concise: what you did, which files changed, and \
what the user should run themselves (tests, builds, commands)."""

# Prepended to a /talk message. The tool list stays the same (changing it would
# invalidate the prompt cache), so the write tools are also refused at dispatch.
READ_ONLY_NOTE = (
    "[Read-only message: answer using list_directory, read_file, and search only. "
    "Do not write or change any files; the write tools (write_file, edit_file, append_file) "
    "are disabled for this message.]\n\n"
)

# Prepended to a /plan message. Also read-only at dispatch: the model may look
# around the project, but its answer is a plan, not the implementation itself.
PLAN_NOTE = (
    "[Plan request: reply with a concise implementation plan, not the implementation. First "
    "look around the project with list_directory, read_file, and search as needed, then answer with: "
    "the goal in one sentence, a short numbered list of steps, which files would be created "
    "or changed, and anything the user would need to run themselves (tests, builds, commands). "
    "Do not write or change any files; the write tools (write_file, edit_file, append_file) "
    "are disabled for this message.]\n\n"
)

# Prepended to an ordinary message while a plan is being discussed: the user is
# refining the plan, not asking for the work to start. Read-only at dispatch too.
PLAN_FEEDBACK_NOTE = (
    "[Plan update: the user is giving feedback on the implementation plan you proposed. Stay in "
    "planning mode: revise the plan accordingly and present the full revised plan concisely — "
    "goal, numbered steps, files to create or change, anything the user must run themselves. "
    "Do not implement it, and do not write or change any files; the write tools (write_file, "
    "edit_file, append_file) are disabled for this message.]\n\n"
)

# Prepended when the user accepts the plan: this is the signal to do the work,
# and the write tools are available again.
PLAN_APPROVED_NOTE = (
    "[Plan approved: implement the plan above now, making the file changes it describes, "
    "one step at a time.]\n\n"
)


def cut_off_reason(content: list) -> str:
    """Explain a response that hit MAX_TOKENS, based on what it had produced so far."""
    types = {block["type"] for block in content}
    if "tool_use" in types:
        return (
            f"Response hit the {MAX_TOKENS:,}-token output limit in the middle of a tool call, so it was "
            "not run. Ask for smaller steps, e.g. one file at a time."
        )
    if types and types <= {"thinking", "redacted_thinking"}:
        return (
            f"Response hit the {MAX_TOKENS:,}-token output limit while still thinking, so there is no "
            "answer. Ask for smaller steps, or turn thinking off for later messages with /thinking-off "
            "(with Ollama, --think low at startup sets a shorter budget)."
        )
    return f"Response hit the {MAX_TOKENS:,}-token output limit and was cut off."


class StreamPrinter:
    """Prints thinking (light blue) and reply text as they stream in, each under its own
    label, rendering their Markdown. When the thinking block ends and the reply is about
    to start, a rule is drawn across the screen to divide them."""

    def __init__(self, label: str, show_thinking: bool):
        self.label = label
        self.show_thinking = show_thinking
        self.current = None  # "thinking", "text", or None between blocks
        self.last = None  # kind of the block that closed most recently
        self.markdown = None  # renderer for the open block

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def start(self, kind: str, text: str) -> str:
        """Open a labelled block if one of this kind isn't already open, and return
        the text to print (without leading whitespace when the block is new)."""
        if self.current == kind:
            return text
        # The clients differ in when the previous block closes: the Claude stream
        # sends content_block_stop between blocks, while Ollama switches from
        # thinking to content mid-stream, so check both here and in end_block.
        divider = kind == "text" and (self.current == "thinking" or self.last == "thinking")
        self.end_block()
        header = style.dim("thinking>") if kind == "thinking" else style.model_label(self.label + ">")
        if divider:
            # The rule takes the place of the usual blank line after the thinking.
            self.rule()
            self.write(f"{header} ")
        else:
            self.write(f"\n{header} ")
        self.current = kind
        self.markdown = MarkdownStream(self.write, [style.LIGHT_BLUE] if kind == "thinking" else [])
        return text.lstrip()

    def thinking(self, text: str) -> None:
        if self.show_thinking and (text.strip() or self.current == "thinking"):
            text = self.start("thinking", text)
            self.markdown.feed(text)

    def text(self, text: str) -> None:
        if text.strip() or self.current == "text":
            text = self.start("text", text)
            self.markdown.feed(text)

    def end_block(self) -> None:
        if self.current:
            self.markdown.finish()
            if not self.markdown.at_line_start:
                self.write("\n")
            self.last = self.current
            self.current = None
            self.markdown = None

    def rule(self) -> None:
        """A horizontal line across the screen, drawn between thinking and reply.

        Full width, unlike the Markdown rules inside a reply, so the two are easy
        to tell apart."""
        self.write(style.dim("─" * shutil.get_terminal_size().columns) + "\n")

    def notice(self, text: str) -> None:
        self.end_block()
        print(style.warning(f"  [{text}]"))


class Agent:
    def __init__(
        self,
        client: Client | OllamaClient,
        workspace: Workspace,
        model: str,
        show_thinking: bool = True,
        redactor: Redactor | None = None,
    ):
        self.client = client
        self.workspace = workspace
        self.model = model
        # Replaces each developer-configured string with "<redacted>" in every
        # text that enters the conversation below (core/redact.py); None, or an
        # empty Redactor, sends text unchanged.
        self.redactor = redactor
        # Only Claude models have prices; Ollama Cloud is billed by subscription.
        self.priced = model in MODELS
        self.label = "claude" if self.priced else model
        self.show_thinking = show_thinking
        self.messages = []
        self.session_usage = Usage()
        self.read_only = False

    def set_model(self, model: str) -> None:
        """Switch the model used for later messages (/model, like --model at startup).

        request_body() reads self.model on every call, so the change takes effect on
        the next message while the conversation history is kept. priced and label are
        re-derived from the model here, exactly as in __init__.
        """
        self.model = model
        # Only Claude models have prices; Ollama Cloud is billed by subscription.
        self.priced = model in MODELS
        self.label = "claude" if self.priced else model

    def send(self, user_text: str, read_only: bool = False, plan: str | None = None) -> None:
        """Run one user turn, then print its token usage and estimated cost.

        With read_only, the model may list and read files but every write is refused.
        plan selects the planning behaviour, where each note is prepended to the user
        text (keeping the system prompt and tool list cacheable):
          "start"    /plan <task>: answer with a concise implementation plan (read-only)
          "update"   a plan-mode message: revise the plan accordingly (read-only)
          "approved" the user accepted the plan: implement it now (writes allowed)

        The summary is printed even if the turn fails part-way, because the
        API calls that did complete are still billed.
        """
        turn_usage = Usage()
        self.read_only = read_only or plan in ("start", "update")
        # Redaction runs first, so everything the model is told — user input, a
        # /paste buffer (already merged into user_text by loom.py), and the
        # /plan and /talk notes prepended below — has passed the filter.
        # Filtering here rather than in request_body() also keeps configured
        # strings out of the stored history: they never join the conversation.
        if self.redactor:
            user_text = self.redactor.redact(user_text)
        if plan == "start":
            user_text = PLAN_NOTE + user_text
        elif plan == "update":
            user_text = PLAN_FEEDBACK_NOTE + user_text
        elif plan == "approved":
            user_text = PLAN_APPROVED_NOTE + user_text
        elif read_only:
            user_text = READ_ONLY_NOTE + user_text
        try:
            self.run_turn(user_text, turn_usage)
        finally:
            self.read_only = False
            self.session_usage.add(turn_usage)
            if (turn_usage.total_input or turn_usage.output) and self.priced:
                print(
                    "\n"
                    + style.dim(
                        f"  [{turn_usage.total_input:,} input tokens ({turn_usage.cache_read:,} from cache), "
                        f"{turn_usage.output:,} output · ~${turn_usage.cost:.4f} this turn · "
                        f"~${self.session_usage.cost:.4f} this session]"
                    )
                )
            elif turn_usage.total_input or turn_usage.output:
                print("\n" + style.dim(f"  [{turn_usage.total_input:,} input tokens, {turn_usage.output:,} output]"))

    def last_reply_text(self) -> str:
        """The text of the newest assistant reply, without thinking or tool calls.

        Empty when the latest assistant message had no text at all, which happens
        e.g. when a response was cut off while it was still thinking.
        """
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return "".join(
                    block["text"]
                    for block in message.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        return ""

    def request_body(self) -> dict:
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": CACHE_CONTROL}],
            "tools": TOOLS,
            "messages": self.messages,
            "output_config": {"effort": "high"},
            "cache_control": CACHE_CONTROL,
        }
        if self.priced and self.show_thinking:
            # Claude returns empty thinking text unless a summary is asked for.
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.priced and MODELS[self.model].fallbacks:
            body["fallbacks"] = "default"
        return body

    def run_turn(self, user_text: str, turn_usage: Usage) -> None:
        self.messages.append({"role": "user", "content": user_text})
        while True:
            # A fresh printer per API call: the rule only divides a tool-free
            # thinking block from the reply streamed in the same response.
            printer = StreamPrinter(self.label, self.show_thinking)
            response = self.client.create_message(self.request_body(), printer)
            turn_usage.add(response_usage(response, self.model))
            content = response.get("content", [])
            stop_reason = response.get("stop_reason")
            # Append the full content (including thinking blocks) unchanged.
            self.messages.append({"role": "assistant", "content": content})

            if stop_reason == "refusal":
                print("\n" + style.warning("  [The model declined this request.]"))
                return
            if stop_reason == "max_tokens":
                print("\n" + style.warning(f"  [{cut_off_reason(content)}]"))
                # A cut-off tool call has incomplete input and no result, which the
                # next request would be rejected for, so drop it from the history.
                kept = [block for block in content if block["type"] != "tool_use"]
                if kept:
                    self.messages[-1] = {"role": "assistant", "content": kept}
                else:
                    self.messages.pop()
                return
            if stop_reason != "tool_use":
                return

            tool_results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                tool_input = block.get("input") or {}
                if block["name"] == "search":
                    shown_path = tool_input.get("path", ".")
                    shown = tool_input.get("pattern", "") + (f" in {shown_path}" if shown_path != "." else "")
                else:
                    shown = tool_input.get("path", "")
                print(f"  {style.dim('->')} {style.tool_name(block['name'])}({shown})")
                try:
                    if self.read_only and block["name"] in WRITE_TOOLS:
                        raise ToolError("writes are disabled: this message is read-only (sent with /talk or /plan)")
                    result = run_tool(self.workspace, block["name"], tool_input)
                    # Every tool result — file contents, search matches,
                    # listings, confirmations, error notes — is read from user
                    # data, so each one passes the redaction filter too. Both
                    # shapes are handled: plain strings and read_file's note
                    # blocks.
                    if self.redactor:
                        result = self.redactor.redact_content(result)
                    tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": result})
                except (ToolError, OSError) as e:
                    print(style.error(f"     error: {e}"))
                    error_text = f"Error: {e}. This tool call failed and changed nothing."
                    if self.redactor:
                        error_text = self.redactor.redact(error_text)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": error_text,
                            "is_error": True,
                        }
                    )
            self.messages.append({"role": "user", "content": tool_results})