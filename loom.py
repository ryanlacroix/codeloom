#!/usr/bin/env python3
"""A minimal, locked-down agentic coding harness for Claude or Ollama Cloud models.

The model gets exactly six tools, all confined to a single workspace directory:
  - list_directory: list entries in a directory
  - read_file:      read a text file (UTF-8, or the fallback encoding, cp1252 by default),
                    paged by line (offset/limit) so files of any size are readable in slices
  - search:         regex-search file contents ('path:line: text' per match) without
                    reading whole files; results are capped and binary files are skipped
  - write_file:     create or overwrite a text file (asks you first); existing files
                    keep their original encoding and line endings
  - edit_file:      replace one exact snippet in a file instead of rewriting it whole,
                    which costs a fraction of the output tokens
  - append_file:    add text to the end of a file, or create it; a long file can be
                    written in several chunks, one per model response

There is deliberately no tool for running commands: no module here calls
os.system / exec / eval, and the sole subprocess in the whole codebase is the
/paste editor launcher, started by the harness alone and never given model-supplied
input. The model can only read and write files inside the workspace; any path that
resolves outside it (via "..", absolute paths, or symlinks) is rejected. The .git
directory is off limits too — its config and hooks are a command-execution vector —
and search never follows symlinks or reads blocked or protected files, so the walk
cannot reach anything outside the workspace either.

Auth: put your Claude access token in key.txt, or your Ollama API key in ollama_key.txt,
next to these scripts (sent as a Bearer token). The model is never allowed to read or write
either key file, even if the workspace contains it.

Uses only the Python standard library; the API is called directly over HTTPS.
"""

import argparse
import codecs
import os
import stat
import subprocess
import sys
from pathlib import Path

from ui import splash, style
from core.agent import Agent
from api.api_client import APIConnectionError, APIError, Client
from api.models import DEFAULT_MODEL, MODELS, OLLAMA_DEFAULT_MODEL, OLLAMA_MODELS
from api.ollama_client import OllamaClient
from core.workspace import Workspace

try:
    # Importing readline gives input() line editing (arrow keys, history).
    # It is not available on Windows, where the console already provides this.
    import readline  # noqa: F401
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
KEY_FILES = {
    "claude": (SCRIPT_DIR.parent / "key.txt", "Claude access token"),
    "ollama": (SCRIPT_DIR.parent / "ollama_key.txt", "Ollama API key"),
}

# Plans are saved here, in the harness's own temp directory next to loom.py, so
# they survive restarts and can be read outside the conversation while the project
# the model works on stays untouched. The directory sits outside the workspace,
# so the model never sees it, and no project's .gitignore needs an entry.
TEMP_DIR = SCRIPT_DIR / ".loom-temp"
PLAN_FILE = "plan.md"

# The /paste command creates this scratch file under TEMP_DIR and opens it in the
# user's default text editor; its contents are quoted in front of the next message
# submitted to the model. The path is also given to the Workspace as blocked, so
# the model can neither read nor write the paste buffer.
PASTE_FILE = "paste.txt"

# Slash commands typed at the "you>" prompt; /help lists these.
COMMANDS = [
    ("/help", "show this list of commands"),
    ("/paste", "paste long text into an external editor"),
    ("/thinking-on", "turn thinking on for all later messages"),
    ("/thinking-off", "turn thinking off for all later messages"),
    ("/model [name]", "list the models, or switch to <name> for later messages"),
    ("/talk <question>", "ask the model something without any files being changed"),
    ("/plan <task>", "propose an implementation plan for the task, changing no files"),
    ("/ok", "approve the current plan and have the model implement it"),
    ("/cancel", "leave plan mode without implementing the plan"),
    ("/reset", "clear the conversation and start over"),
    ("/exit", "leave the harness (also: /quit)"),
]


def print_commands() -> None:
    width = max(len(name) for name, _ in COMMANDS)
    print()
    for name, what in COMMANDS:
        print(f"  {style.bold(name.ljust(width))}  {what}")


def print_models(current: str) -> None:
    """The /model list: the models /model can switch to, with the current one marked."""
    print()
    for name in OLLAMA_MODELS:
        line = style.model_label(name)
        if name == current:
            line += style.dim("  (current)")
        print(f"  {line}")


def thinking_suffix(agent: Agent) -> str:
    """The prompt extra when the model's thinking is off: ' (thinking off)'.

    Claude does not think at all while agent.show_thinking is False (agent.py
    only adds the thinking config to the request when it is True), so that flag
    alone decides. Ollama's thinking is separate server-side state (client.think:
    on / off / a low-medium-high level), and --hide-thinking merely hides the
    stream while thinking continues, so there only client.think 'off' counts.
    The /thinking-on and /thinking-off commands set both flags together, so the two
    views agree whenever a toggle happens.
    """
    off = not agent.show_thinking if agent.priced else agent.client.think == "off"
    return " (thinking off)" if off else ""


def read_key(provider: str) -> str:
    key_file, what = KEY_FILES[provider]
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        sys.exit(f"No key file found. Put your {what} in {key_file}")
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"Could not read {key_file}: {e}")
    if not key:
        sys.exit(f"{key_file} is empty. Put your {what} in it.")
    if key_file.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(style.warning(f"Warning: {key_file} can be read by other users. Run: chmod 600 {key_file}") + "\n")
    return key


def make_client(provider: str, think: str | None) -> Client | OllamaClient:
    key = read_key(provider)
    return OllamaClient(key, think=think) if provider == "ollama" else Client(key)


def save_plan(agent: Agent) -> None:
    """Save the newest plan reply to the harness's .loom-temp/plan.md.

    Trusted harness code writes this directly (not through the model's sandboxed
    tools), so saving needs no approval, and the directory is created if missing.
    The file lives next to loom.py, outside the workspace, so the project the
    model works on is left untouched. Failures are reported but never fatal.
    """
    text = agent.last_reply_text()
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


# Handles for editors launched by open_in_editor, kept here so they are never
# garbage-collected while the editor still runs (a finalised Popen can emit a
# "subprocess ... is still running" warning). Nothing ever waits on these.
_EDITOR_PROCS = set()


def open_in_editor(path: Path) -> bool:
    """Open a file with the user's default text-file editor; True if launched.

    The one place the harness ever starts a subprocess: a fixed launcher per
    platform plus the path itself as a single argument, so there is no command
    line for anything to be injected into. The call frees the terminal at once
    and never waits for the editor: os.startfile is async by design on Windows,
    and on POSIX Popen hands the editor its own session (no controlling terminal)
    and DEVNULL std streams, so even an xdg-open that blocks until the app exits,
    or a chatty editor inheriting stdio, can neither print into nor hold the
    harness's terminal, and the prompt is usable immediately. With no parent to
    wait for it, the editor also survives the harness exiting.
    """
    detached = dict(
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            _EDITOR_PROCS.add(subprocess.Popen(["open", str(path)], **detached))
        else:
            _EDITOR_PROCS.add(subprocess.Popen(["xdg-open", str(path)], **detached))
    except (OSError, AttributeError) as e:
        # AttributeError: os.startfile does not exist off Windows; OSError covers
        # a missing xdg-open/open (FileNotFoundError is an OSError subclass).
        print(style.warning(f"  [could not open an editor for {path}: {e}]"))
        return False
    return True


def main() -> None:
    # Clear the console before anything is printed, including argparse usage
    # errors and the key-file warning (which used to precede the splash).
    style.clear_screen()
    parser = argparse.ArgumentParser(description="Locked-down file-only coding agent for Claude or Ollama Cloud.")
    parser.add_argument("dir", help="project directory the model may access (required)")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="approve file writes without asking; you lose the diff review that checks "
        "writes to files you may later execute",
    )
    parser.add_argument(
        "--talk",
        action="store_true",
        help="read-only session: every message behaves like /talk, so no files can be changed",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(KEY_FILES),
        default="ollama",
        help="API to connect to (default: ollama)",
    )
    parser.add_argument(
        "--model",
        help=f"model to use. Claude: one of {', '.join(sorted(MODELS))} (default: {DEFAULT_MODEL}). "
        f"Ollama: any Ollama Cloud model (default: {OLLAMA_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--think",
        choices=["on", "off", "low", "medium", "high"],
        help="Ollama only: thinking for models that support it. Use a level (low/medium/high) "
        "for gpt-oss models, and 'off' for models that cannot think (default: on)",
    )
    parser.add_argument(
        "--hide-thinking",
        action="store_true",
        help="don't show the model's thinking as it streams (the reply still streams)",
    )
    parser.add_argument(
        "--fallback-encoding",
        default="cp1252",
        help="encoding assumed for files that are not valid UTF-8 (default: cp1252)",
    )
    args = parser.parse_args()

    if args.provider == "claude":
        args.model = args.model or DEFAULT_MODEL
        if args.model not in MODELS:
            parser.error(f"unknown Claude model {args.model!r}; choose from {', '.join(sorted(MODELS))}")
        if args.think:
            parser.error("--think only applies to --provider ollama (Claude's reasoning is already on)")
    else:
        args.model = args.model or OLLAMA_DEFAULT_MODEL
        args.think = args.think or "on"
    if not args.dir.strip():
        parser.error("a project directory is required")
    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    try:
        codecs.lookup(args.fallback_encoding)
    except LookupError:
        sys.exit(f"Unknown encoding: {args.fallback_encoding}")

    workspace = Workspace(
        root,
        auto_approve_writes=args.yes,
        fallback_encoding=args.fallback_encoding,
        # The paste buffer is off-limits to the model: it can neither read the
        # pending paste out of the harness's .loom-temp/paste.txt nor plant text
        # there to be quoted into a later "user" message. The buffer lives outside
        # the workspace, so containment already keeps it out of reach; the blocked
        # entry stays as defense in depth for a workspace that contains the
        # harness folder itself (e.g. this repo used as a project). workspace.py
        # resolves every blocked entry with Path.resolve(), which succeeds for a
        # file that does not exist yet, so this holds even before the first /paste
        # creates the file.
        blocked=(TEMP_DIR / PASTE_FILE, *(path for path, _ in KEY_FILES.values())),
    )
    agent = Agent(
        make_client(args.provider, args.think), workspace, model=args.model, show_thinking=not args.hide_thinking
    )

    splash.print_splash()
    print(f"{style.bold('Workspace:')} {workspace.root}")
    print(f"{style.bold('Model:')}     {style.model_label(args.model)} {style.dim(f'({args.provider})')}")
    if args.think:
        print(f"{style.bold('Thinking:')}  {args.think} {style.dim('(toggle with /thinking-on, /thinking-off)')}")
    print()
    print(style.dim("The model can read and write files here only. No commands can be run, and .git is protected."))
    if args.talk:
        print(style.warning("Read-only mode: every message behaves like /talk; no files can be changed."))
    else:
        print(style.dim("Start a message with /talk to ask, or /plan to get a plan to approve, without any files being changed."))
    print(style.dim("Type /help to list commands, /reset to clear the conversation, /exit to quit."))

    # While a plan is being discussed (after /plan, until it is approved or abandoned),
    # ordinary messages revise the plan instead of asking for work, write_file stays
    # disabled, and the prompt is "plan>" to make the mode visible.
    plan_mode = False

    # Set by /paste: the next message submitted to the model gets the contents of
    # the harness's .loom-temp/paste.txt quoted in front of it, and the flag is
    # then consumed. While it is set the prompt reads "you (with pasted text)>".
    # Commands that send nothing to the model leave it set; /reset clears it and
    # reverts the prompt.
    paste_pending = False

    while True:
        # Rebuilt every iteration so /thinking-on and /thinking-off take effect on the
        # very next prompt.
        suffix = thinking_suffix(agent)
        try:
            # A blank line always separates the prompt from whatever came before it
            # (the splash, a command's note, the tail of a model turn), so no other
            # output needs to end with a blank line of its own.
            print()
            if plan_mode:
                prompt_line = style.prompt(f"plan{suffix}> ", style.BOLD, style.MAGENTA)
            else:
                # While a paste is pending, label the prompt so the user can see
                # the editor buffer will be quoted in front of this message;
                # submitting (or /reset) consumes the flag and reverts the label.
                # thinking_suffix() parentheses compose with this one.
                label = "you (with pasted text)" if paste_pending else "you"
                prompt_line = style.prompt(f"{label}{suffix}> ", style.BOLD, style.GREEN)
            user_text = input(prompt_line).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/reset":
            agent.messages = []
            plan_mode = False
            paste_pending = False
            if agent.priced:
                print(style.dim(f"  [conversation cleared; ~${agent.session_usage.cost:.4f} spent this session]"))
            else:
                print(style.dim("  [conversation cleared]"))
            continue
        if user_text == "/help":
            print_commands()
            continue
        # /thinking-on and /thinking-off toggle model thinking for all later messages, in
        # either prompt (in plan mode the setting is applied and the `plan>` prompt
        # returns with its pending plan intact). Anything typed after the command
        # word is ignored, and nothing is sent to the model. Handled here, before
        # the planning state machine, like /reset and /help.
        if user_text.startswith("/thinking-on") or user_text.startswith("/thinking-off"):
            on = user_text.startswith("/thinking-on")
            agent.show_thinking = on
            if not agent.priced:
                # Ollama: "off" stops thinking server-side as well as hiding it, and
                # turning it back on restores the startup --think setting, which may
                # be a level (low/medium/high) rather than plain "on".
                agent.client.think = args.think if on else "off"
            print(style.dim(f"  [thinking {'on' if on else 'off'}]"))
            continue
        # /model with no argument lists the models it can switch to (the current one
        # marked); /model <name> switches for all later messages, accepting any Ollama
        # Cloud model name, like --model at startup. Handled here, before the planning
        # state machine, so it also works while a plan is open, and nothing is sent
        # to the model.
        if user_text == "/model" or user_text.startswith(("/model ", "/model\t")):
            name = user_text[len("/model") :].strip()
            if not name:
                print_models(agent.model)
                continue
            agent.set_model(name)
            print(style.dim(f"  [model set to {name}]"))
            if name not in OLLAMA_MODELS:
                print(style.dim("  [not in the /model list; Ollama Cloud accepts any model name]"))
            continue

        # /paste creates a fresh empty paste buffer in the harness's temp folder
        # (next to loom.py, outside the workspace) and opens it in the user's
        # default editor. Whenever the next message is actually submitted to the
        # model, its saved contents are quoted in front of the prompt and the
        # pending state is consumed (the hook sits just before agent.send). Like
        # the commands above, it works while a plan is open and sends nothing to
        # the model; anything typed after the command word is ignored. If a paste
        # is already pending the file is only reopened, so an unsent draft is kept.
        if user_text == "/paste" or user_text.startswith(("/paste ", "/paste\t")):
            paste_path = TEMP_DIR / PASTE_FILE
            try:
                paste_path.parent.mkdir(parents=True, exist_ok=True)
                if not paste_pending:
                    paste_path.write_bytes(b"")  # a fresh buffer for each /paste
            except OSError as e:
                print(style.warning(f"  [could not create {paste_path}: {e}]"))
                continue
            open_in_editor(paste_path)
            paste_pending = True
            print(style.dim(f"  [pasting into file: {paste_path}]"))
            continue

        # Planning state machine. plan tells Agent.send what kind of message this is:
        # "start" proposes a plan, "update" revises it, "approved" implements it.
        plan = None
        if user_text == "/ok":
            if not plan_mode:
                print(style.dim("  [no plan to accept — start one with /plan <task>]"))
                continue
            plan = "approved"
            user_text = ""  # the approval note is all the model needs; prepended by agent.send
            print(style.dim("  [plan approved]"))
        elif user_text == "/cancel":
            if not plan_mode:
                print(style.dim("  [not in plan mode]"))
                continue
            plan_mode = False
            print(style.dim("  [plan mode off; the plan stays in the conversation]"))
            continue
        else:
            if user_text == "/plan" or user_text.startswith(("/plan ", "/plan\t")):
                user_text = user_text[len("/plan") :].strip()
                if not user_text:
                    print(style.dim("  [usage: /plan <task>; the model proposes a plan and changes no files]"))
                    continue
                plan = "start"
                print(style.dim("  [plan mode: the model will propose what to implement; no files will be changed]"))
            elif plan_mode and not (
                user_text == "/talk" or user_text.startswith(("/talk ", "/talk\t"))
            ):
                # Ordinary messages while planning are feedback that revises the plan;
                # /talk stays a side question and cannot change files either.
                plan = "update"
                print(style.dim("  [plan mode: this message revises the plan; no files will be changed]"))

        # With --talk every message is read-only; the /talk prefix makes just its
        # message read-only. The prefix is stripped in both cases.
        read_only = args.talk or user_text == "/talk" or user_text.startswith(("/talk ", "/talk\t"))
        if read_only and user_text.startswith("/talk"):
            user_text = user_text[len("/talk") :].strip()
            if not user_text:
                print(style.dim("  [usage: /talk <question>; the model can read files but not change them]"))
                continue
            if not args.talk:
                # With --talk the read-only state was already announced at startup.
                print(style.dim("  [read-only: no files will be changed]"))

        # The pending paste is quoted in front of the prompt at the last moment,
        # after /talk stripping, so whatever was last saved in the editor is what
        # the model sees. This counts as the submission for /plan, plan revisions,
        # /ok and /talk too, and the flag is consumed before sending whatever the
        # turn's outcome, so a failed turn does not re-paste on the next attempt.
        if paste_pending:
            paste_pending = False
            try:
                # The editor may save in a non-UTF-8 encoding; replace any bad
                # bytes rather than losing the paste. .strip() drops editor-added
                # leading/trailing whitespace and CRLF is already normalised.
                paste_text = (TEMP_DIR / PASTE_FILE).read_text(encoding="utf-8", errors="replace").strip()
            except OSError as e:
                print(style.warning(f"  [could not read {TEMP_DIR / PASTE_FILE}: {e}]"))
            else:
                if paste_text:
                    user_text = f"The user pasted this text:\n\n{paste_text}\n\n{user_text}"
                    print(style.dim(f"  [pasted {PASTE_FILE} in front of your message]"))
                else:
                    print(style.dim(f"  [{TEMP_DIR / PASTE_FILE} was empty; nothing pasted]"))

        history_len = len(agent.messages)
        try:
            agent.send(user_text, read_only=read_only, plan=plan)
            # Plan state advances only once the turn fully succeeded, so a failed or
            # interrupted turn leaves the previous state in place.
            if plan == "approved":
                plan_mode = False
            elif plan in ("start", "update"):
                plan_mode = True
                save_plan(agent)
                print(style.dim("  [/ok to approve and implement the plan; anything else revises it]"))
        except KeyboardInterrupt:
            print("\n" + style.warning("  [interrupted]"))
            del agent.messages[history_len:]
        except APIError as e:
            if e.status == 401:
                print(style.error(f"  [authentication failed: check the key in {KEY_FILES[args.provider][0]}]"))
            elif e.status == 429:
                print(style.warning("  [rate limited: wait a moment and try again]"))
            else:
                print(style.error(f"  [API error {e.status}: {e.message}]"))
                if args.think and "think" in e.message.lower():
                    print(style.warning("  [this model may not support thinking: try again with --think off]"))
            del agent.messages[history_len:]
        except APIConnectionError as e:
            print(style.error(f"  [could not reach the API: {e}]"))
            del agent.messages[history_len:]


if __name__ == "__main__":
    main()