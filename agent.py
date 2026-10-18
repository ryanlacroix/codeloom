"""The conversation loop: sends messages to the model and runs the tools it calls."""

import shutil
import sys

import style
from api_client import Client
from markdown import MarkdownStream
from ollama_client import OllamaClient
from models import MODELS, Usage, response_usage
from tools import TOOLS, run_tool
from workspace import ToolError, Workspace

MAX_TOKENS = 16000
# Prompt caching: the marker on the system prompt caches the tool definitions
# and system prompt (tools come first in the request), and the top-level
# marker caches the conversation so far. Each call then pays full price only
# for what is new since the previous call. This only works while the history
# is append-only, so earlier messages must never be edited.
CACHE_CONTROL = {"type": "ephemeral"}

SYSTEM_PROMPT = """You are a coding assistant working inside a single project directory.

You can only list directories, read files, and write files, using the tools provided. \
You cannot run commands, execute code, run tests, install packages, or access the network. \
All paths are relative to the project root. When a task would normally require running \
something (tests, builds, scripts), make the file changes and tell the user what to run \
themselves.

You must read an existing file before overwriting it. write_file replaces the whole file, \
so always write the complete new contents. Existing files are saved back in their original \
encoding and line endings; if read_file says a file is not UTF-8, only use characters that \
encoding can represent. Binary files (for example .xlsx or .dll) cannot be read or written. \
The user must approve every write; if a write is declined, ask what they would like instead."""

# Prepended to a /talk message. The tool list stays the same (changing it would
# invalidate the prompt cache), so write_file is also refused at dispatch.
READ_ONLY_NOTE = (
    "[Read-only message: answer using list_directory and read_file only. "
    "Do not write or change any files; write_file is disabled for this message.]\n\n"
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
            "answer. Ask for smaller steps, or with Ollama try --think low or --think off."
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
        self, client: Client | OllamaClient, workspace: Workspace, model: str, show_thinking: bool = True
    ):
        self.client = client
        self.workspace = workspace
        self.model = model
        # Only Claude models have prices; Ollama Cloud is billed by subscription.
        self.priced = model in MODELS
        self.label = "claude" if self.priced else model
        self.show_thinking = show_thinking
        self.messages = []
        self.session_usage = Usage()
        self.read_only = False

    def send(self, user_text: str, read_only: bool = False) -> None:
        """Run one user turn, then print its token usage and estimated cost.

        With read_only, the model may list and read files but every write is refused.

        The summary is printed even if the turn fails part-way, because the
        API calls that did complete are still billed.
        """
        turn_usage = Usage()
        self.read_only = read_only
        if read_only:
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
                print(f"  {style.dim('->')} {style.tool_name(block['name'])}({tool_input.get('path', '')})")
                try:
                    if self.read_only and block["name"] == "write_file":
                        raise ToolError("writes are disabled: the user sent this message with /talk (read-only)")
                    result = run_tool(self.workspace, block["name"], tool_input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": result})
                except (ToolError, OSError) as e:
                    print(style.error(f"     error: {e}"))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": f"Error: {e}. This tool call failed and changed nothing.",
                            "is_error": True,
                        }
                    )
            self.messages.append({"role": "user", "content": tool_results})