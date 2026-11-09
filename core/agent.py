"""The conversation loop: sends messages to the model and runs the tools it calls."""

from ui import style
from ui.stream import StreamPrinter
from api.ollama_client import OllamaClient
from api.models import Usage
from core.prompts import (
    PLAN_APPROVED_NOTE,
    PLAN_FEEDBACK_NOTE,
    PLAN_NOTE,
    READ_ONLY_NOTE,
    SYSTEM_PROMPT,
)
from core.redact import Redactor
from core.tools import TOOLS, run_tool
from core.workspace import ToolError, Workspace

MAX_TOKENS = 16000

# Tools that change files. Refused by name in read-only messages (/talk, /plan)
# so the tool list itself can stay identical in every request.
WRITE_TOOLS = {"write_file", "edit_file", "append_file"}


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


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        workspace: Workspace,
        model: str,
        show_thinking: bool = True,
        redactor: Redactor | None = None,
    ):
        self.client = client
        self.workspace = workspace
        self.model = model
        # Replaces each developer-configured string with the redact.json marker
        # ("<redacted>" by default) in every text that enters the conversation
        # below (core/redact.py); None, or an empty Redactor, sends text
        # unchanged.
        self.redactor = redactor
        self.show_thinking = show_thinking
        self.messages = []
        self.session_usage = Usage()
        self.read_only = False

    def set_model(self, model: str) -> None:
        """Switch the model used for later messages (/model, like --model at startup).

        request_body() reads self.model on every call, so the change takes effect on
        the next message while the conversation history is kept, and the streamed
        label follows it.
        """
        self.model = model

    def send(self, user_text: str, read_only: bool = False, plan: str | None = None) -> None:
        """Run one user turn, then print its token usage.

        With read_only, the model may list and read files but every write is refused.
        plan selects the planning behaviour, where each note is prepended to the user
        text:
          "start"    /plan <task>: answer with a concise implementation plan (read-only)
          "update"   a plan-mode message: revise the plan accordingly (read-only)
          "approved" the user accepted the plan: implement it now (writes allowed)

        The summary is printed even if the turn fails part-way, because the
        API calls that did complete were still sent and counted.
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
            if turn_usage.input or turn_usage.output:
                print("\n" + style.dim(f"  [{turn_usage.input:,} input tokens, {turn_usage.output:,} output]"))

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
        """The body sent to the client: the model, the output limit, the system
        prompt, the tool definitions, and the conversation so far. Thinking is
        client-side state (OllamaClient.think), set outside the body."""
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "tools": TOOLS,
            "messages": self.messages,
        }

    def run_turn(self, user_text: str, turn_usage: Usage) -> None:
        self.messages.append({"role": "user", "content": user_text})
        while True:
            # A fresh printer per API call: the rule only divides a tool-free
            # thinking block from the reply streamed in the same response.
            printer = StreamPrinter(self.model, self.show_thinking)
            response = self.client.create_message(self.request_body(), printer)
            turn_usage.add(response.get("usage") or Usage())
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

            self.messages.append({"role": "user", "content": self.execute_tools(content)})

    def execute_tools(self, content: list) -> list[dict]:
        """Run each tool_use block in one assistant response, returning the results.

        Every result — file contents, search matches, listings, confirmations,
        error notes — is read from user data, so each one passes the redaction
        filter, which walks any shape: plain strings, read_file's note blocks,
        or whatever a future tool returns. A failed call changes nothing: the
        console keeps the raw error (the developer wants the real path), while
        the conversation gets a path-blind copy, because a raw OSError embeds
        the resolved absolute target — so scrub_paths rewrites every absolute
        path relative to the workspace root before the redaction filter runs.
        (ToolError text already names paths as the model asked for them, and
        scrubbing is a no-op there.)

        Every failure becomes a failed result rather than an exception, and
        that covers more than the expected ToolError and OSError: the API does
        not guarantee the types the tool schemas ask for (Ollama leaves input
        schemas unchecked), so a null or a number where a string was meant must
        not end the session, and neither must an unforeseen error — the console
        names the class for anything but ToolError and OSError, so nothing is
        swept under the note.
        """
        tool_results = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name") or ""
            tool_use_id = block.get("id") or ""
            tool_input = block.get("input") or {}
            if not isinstance(tool_input, dict):
                tool_input = {}
            if name == "search":
                shown_path = tool_input.get("path", ".")
                shown = tool_input.get("pattern", "")
                shown = f"{shown} in {shown_path}" if shown_path != "." else shown
            else:
                shown = tool_input.get("path", "")
            # The echo line is printed before the tools validate anything, and
            # a model can hand over a non-string where the schema says string,
            # so coerce here rather than crash: style.sanitize needs a str.
            if not isinstance(shown, str):
                shown = repr(shown)
            print(f"  {style.dim('->')} {style.tool_name(name)}({style.sanitize(shown)})")
            try:
                if self.read_only and name in WRITE_TOOLS:
                    raise ToolError("writes are disabled: this message is read-only (sent with /talk or /plan)")
                result = run_tool(self.workspace, name, tool_input)
                if self.redactor:
                    result = self.redactor.redact_content(result)
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": result})
            except Exception as e:
                detail = str(e) or e.__class__.__name__
                if isinstance(e, (ToolError, OSError)):
                    print(style.error(f"     error: {detail}"))
                else:
                    print(style.error(f"     error: {detail} ({e.__class__.__name__})"))
                error_text = self.workspace.scrub_paths(
                    f"Error: {detail}. This tool call failed and changed nothing."
                )
                if self.redactor:
                    error_text = self.redactor.redact(error_text)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": error_text,
                        "is_error": True,
                    }
                )
        return tool_results
