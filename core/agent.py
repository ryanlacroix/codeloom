"""The conversation loop: sends messages to the model and runs the tools it calls."""

from ui import style
from ui.stream import StreamPrinter
from api.ollama_client import OllamaClient
from api.models import Usage, format_tokens
from core.prompts import (
    COMPACT_FOCUS,
    COMPACT_NOTE,
    COMPACT_SUMMARY_NOTE,
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
        # The input-token count the API charged for the newest completed call's
        # prompt: inside a tool loop every request contains all the earlier
        # ones', so the last call's input is always the widest — the context
        # window measure the footer in send() prints. It is a lower bound (the
        # message typed next is not counted yet), and run_turn refreshes it on
        # every call. compact() and Session.reset zero it when they replace the
        # history with something that has not been measured yet.
        self.context_input = 0
        self.read_only = False

    def set_model(self, model: str) -> None:
        """Switch the model used for later messages (/model, like --model at startup).

        request_body() reads self.model on every call, so the change takes effect on
        the next message while the conversation history is kept, and the streamed
        label follows it.
        """
        self.model = model

    def send(self, user_text: str, read_only: bool = False, plan: str | None = None) -> None:
        """Run one user turn, then print the context size and the output usage.

        With read_only, the model may list and read files but every write is refused.
        plan selects the planning behaviour, where each note is prepended to the user
        text:
          "start"    /plan <task>: answer with a concise implementation plan (read-only)
          "update"   a plan-mode message: revise the plan accordingly (read-only)
          "approved" the user accepted the plan: implement it now (writes allowed)

        The summary is printed even if the turn fails part-way, because the
        API calls that did complete were still sent and counted. The context
        figure is the newest completed call's prompt tokens (the widest request
        so far, since each call of a tool loop contains the earlier ones) — a
        lower bound until the message typed next is counted.
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
                print(
                    "\n"
                    + style.dim(f"  [context {format_tokens(self.context_input)}, {format_tokens(turn_usage.output)} output]")
                )

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

    def compact(self, focus: str | None = None) -> Usage | None:
        """Replace the whole conversation with a summary the model writes (/compact).

        One tool-free API call sends the history with a fixed note after it asking
        for that summary (plus any focus instructions the user typed), and it
        streams like an ordinary reply. Only a complete reply with summary text
        compacts: that text, redacted like everything else joining the
        conversation, then becomes the new single history message. Otherwise the
        history stays exactly as it was and this prints why instead (an empty or
        cut-off summary, a refusal, a stray tool call) — the response shape of a
        normal turn, so nothing here can strand a half-written compaction.

        Returns the call's Usage for the caller's report, or None when nothing
        changed. API and connection errors raise for the caller to report, and
        leave the history untouched too.
        """
        if not self.messages:
            raise ValueError("nothing to compact")
        note = COMPACT_NOTE
        if focus:
            note += COMPACT_FOCUS.format(focus=focus) + "\n"
        printer = StreamPrinter(self.model, self.show_thinking)
        response = self.client.create_message(self.compact_body(note), printer)
        turn_usage = Usage()
        turn_usage.add(response.get("usage") or Usage())
        # The call completed, so it counts even when its summary is unusable.
        self.session_usage.add(turn_usage)
        content = response.get("content", [])
        stop_reason = response.get("stop_reason")
        text = "".join(
            block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        reason = None
        if stop_reason == "refusal":
            reason = "the model declined to summarize"
        elif stop_reason == "max_tokens":
            reason = "the summary was cut off by the output limit"
        elif stop_reason == "tool_use":
            reason = "the reply asked to run a tool instead of summarizing"
        elif not text:
            reason = "the reply held no summary text"
        if reason:
            print("\n" + style.warning(f"  [{reason}; the conversation is unchanged]"))
            print(style.dim(f"  [{format_tokens(turn_usage.input)} input tokens, {format_tokens(turn_usage.output)} output]"))
            return None
        if self.redactor:
            text = self.redactor.redact(text)
        self.messages = [{"role": "user", "content": COMPACT_SUMMARY_NOTE + text}]
        # The old measurement described the history that just left; the next
        # turn's first call re-measures the summary-sized context. A failed
        # compact keeps both the history and the measurement.
        self.context_input = 0
        return turn_usage

    def compact_body(self, note: str) -> dict:
        """The body of the one-off compact call: like request_body(), minus the
        tools — the reply must be summary prose, and a call that cannot name a
        tool cannot run one. The model, the output limit and the system prompt
        are unchanged; the messages are a copy of the history plus the summary
        request as the last user message, so a failed compact never leaves the
        note stranded in the live history."""
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": self.messages + [{"role": "user", "content": note}],
        }

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
            call_usage = response.get("usage") or Usage()
            turn_usage.add(call_usage)
            self.context_input = call_usage.input
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
