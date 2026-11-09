"""The state one interactive session carries between prompt lines.

Session owns what the loop remembers between inputs: the planning state (a
plan is being discussed until it is approved or abandoned), the pending
/paste buffer (whose saved contents are quoted in front of the next submitted
message), and the prompt label that reflects both. It also saves the newest
plan to the harness's .loom-temp/plan.md, so it can be read outside the
conversation while the project the model works on stays untouched.
"""

from pathlib import Path

from ui import style

# The harness's own scratch directory, next to loom.py (one level up from this
# file's core/). It sits outside any workspace the model may be confined to, so
# the model never sees it and no project's .gitignore needs an entry, and it
# survives restarts. loom.py passes TEMP_DIR / PASTE_FILE to the Workspace as a
# blocked path as defense in depth for a workspace that contains the harness
# folder itself.
TEMP_DIR = Path(__file__).resolve().parent.parent / ".loom-temp"
PLAN_FILE = "plan.md"
PASTE_FILE = "paste.txt"


class Submit:
    """A message the loop will send to the model, and how.

    plan is None for an ordinary message, or "start" / "update" / "approved"
    for the planning notes (core/prompts.py); read_only marks /talk's side
    question. The loop quotes the paste buffer into text just before sending.
    """

    def __init__(self, text: str, plan: str | None = None, read_only: bool = False):
        self.text = text
        self.plan = plan
        self.read_only = read_only


class Session:
    """Planning and paste state for one run of the prompt loop.

    The transitions advance() makes happen only after a turn fully succeeded,
    so a failed or interrupted turn leaves the previous state in place.
    """

    def __init__(self, agent, think: str):
        self.agent = agent
        # The startup --think level /thinking-on restores, which may be a
        # level ("low"/"medium"/"high") rather than plain "on".
        self.think = think
        self.plan_mode = False
        self.paste_pending = False

    # -- the prompt --

    def prompt(self) -> str:
        """The coloured prompt line, labelled with the state it reflects.

        " (thinking off)" is appended when thinking is off server-side
        (client.think == "off"); --hide-thinking merely hides the stream while
        thinking continues, so only that counts as off here.
        """
        suffix = " (thinking off)" if self.agent.client.think == "off" else ""
        if self.plan_mode:
            return style.prompt(f"plan{suffix}> ", style.BOLD, style.MAGENTA)
        label = "you (with pasted text)" if self.paste_pending else "you"
        return style.prompt(f"{label}{suffix}> ", style.BOLD, style.GREEN)

    # -- planning --

    def as_plan_update(self, submit: Submit) -> Submit:
        """Re-route a submitted message as feedback that revises the open plan."""
        print(style.dim("  [plan mode: this message revises the plan; no files will be changed]"))
        submit.plan = "update"
        return submit

    def advance(self, plan: str | None) -> None:
        """Move the planning state after a turn fully succeeded.

        "approved" ends plan mode; "start" and "update" open (or keep) it,
        save the newest reply as the plan, and point at /ok.
        """
        if plan == "approved":
            self.plan_mode = False
        elif plan in ("start", "update"):
            self.plan_mode = True
            self.save_plan()
            print(style.dim("  [/ok to approve and implement the plan; anything else revises it]"))

    def save_plan(self) -> None:
        """Save the newest plan reply to the harness's .loom-temp/plan.md.

        The directory is created if missing. Failures are reported but never
        fatal. loom.py's /plan and plan-mode messages print their own notes;
        this only reports what happened to the file.
        """
        text = self.agent.last_reply_text()
        if not text.strip():
            print(style.dim("  [the reply had no plan text; nothing saved]"))
            return
        text = text if text.endswith("\n") else text + "\n"
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            (TEMP_DIR / PLAN_FILE).write_text(text, encoding="utf-8", newline="\n")
        except OSError as e:
            print(style.warning(f"  [could not save the plan to {TEMP_DIR / PLAN_FILE}: {e}]"))
            return
        print(style.dim(f"  [plan saved to {TEMP_DIR / PLAN_FILE}]"))

    # -- reset --

    def reset(self) -> None:
        """Clear the conversation and every state flag (/reset)."""
        self.agent.messages = []
        self.plan_mode = False
        self.paste_pending = False
        print(style.dim("  [conversation cleared]"))

    # -- the paste buffer --

    def open_paste(self) -> Path | None:
        """Create a fresh paste buffer when none is pending, and mark it pending.

        Returns the path to hand to the user's editor, or None when the buffer
        could not be created (in which case nothing is pending). If a paste is
        already pending the file is only returned, so an unsent draft is kept.
        """
        try:
            (TEMP_DIR / PASTE_FILE).parent.mkdir(parents=True, exist_ok=True)
            if not self.paste_pending:
                (TEMP_DIR / PASTE_FILE).write_bytes(b"")  # a fresh buffer for each /paste
        except OSError as e:
            print(style.warning(f"  [could not create {TEMP_DIR / PASTE_FILE}: {e}]"))
            return None
        self.paste_pending = True
        return TEMP_DIR / PASTE_FILE

    def with_paste(self, submit: Submit) -> Submit:
        """Quote the saved paste buffer in front of the submitted text, consuming it.

        The editor may save in a non-UTF-8 encoding; replace any bad bytes
        rather than losing the paste. .strip() drops editor-added
        leading/trailing whitespace and CRLF is already normalised.
        """
        self.paste_pending = False
        try:
            paste_text = (TEMP_DIR / PASTE_FILE).read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            print(style.warning(f"  [could not read {TEMP_DIR / PASTE_FILE}: {e}]"))
            return submit
        if paste_text:
            submit.text = f"The user pasted this text:\n\n{paste_text}\n\n{submit.text}"
            print(style.dim(f"  [pasted {PASTE_FILE} in front of your message]"))
        else:
            print(style.dim(f"  [{TEMP_DIR / PASTE_FILE} was empty; nothing pasted]"))
        return submit
