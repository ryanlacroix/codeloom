#!/usr/bin/env python3
"""A minimal, locked-down agentic coding harness for Ollama Cloud models.

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

Auth: the harness reads every secret, and its own configuration, from config/ —
a directory inside this repo that is in .gitignore and invisible to the model
(the Workspace is given config/ as a protected directory: nothing in it can be
listed, read, written, or searched, even when this repo is the workspace). Put
your Ollama API key in config/ollama_key.txt (sent as a Bearer token) and, to use
a custom Anthropic-compatible /v1/messages endpoint instead, config/api_config.json: a JSON file with that endpoint's base_url
and api_key (plus optional model, models, headers); --api-config FILE points
elsewhere. config/redact.json is the redaction list (see REDACT_FILE).

Uses only the Python standard library; the API is called directly over HTTPS.
"""

import argparse
import codecs
import difflib
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ui import splash, style
from core.agent import Agent
from core.prompts import CONTINUE_MESSAGE
from core.session import PASTE_FILE, TEMP_DIR, Session, Submit
from api.anthropic_client import AnthropicClient, load_endpoint_config
from api.http_client import APIConnectionError, APIError
from api.models import OLLAMA_DEFAULT_MODEL, OLLAMA_MODELS
from api.ollama_client import OllamaClient
from core.redact import Redactor, load_redactions
from core.workspace import Workspace

try:
    # Importing readline gives input() line editing (arrow keys, history).
    # It is not available on Windows, where the console already provides this.
    import readline  # noqa: F401
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
# The harness's configuration directory, inside this repo: the API keys, the
# --api-config endpoint file, and the redaction list all live here. It is
# passed to the Workspace's protected_dirs below, so nothing under it — not a
# listing, not a read, not a write, not a search — ever reaches the model,
# whichever directory is open as the workspace; and it is in .gitignore, so
# the secrets stay out of git history too.
CONFIG_DIR = SCRIPT_DIR / "config"
# The Ollama API key file, sent to the API as a Bearer token (see the docstring).
KEY_FILE = CONFIG_DIR / "ollama_key.txt"
# The Anthropic-compatible endpoint config; its presence selects that endpoint
# (--api-config points elsewhere), and it is read only when selected.
API_CONFIG_FILE = CONFIG_DIR / "api_config.json"

# Every text sent to the model — user input, /paste buffers, and everything the
# tools return (file contents, search matches, listings, error notes) — passes
# through a redaction filter (core/redact.py, applied in core/agent.py before
# anything joins the conversation): each string in redact.json's "redactList"
# is replaced with the "replaceWith" marker, "<redacted>" by default. The file
# lives in config/ with the keys, so the model can neither read nor edit its
# own redaction list. Read once at startup; edits need a restart.
REDACT_FILE = "redact.json"

# Slash commands typed at the "you>" prompt: each command is one small function
# below (its docstring is its /help text), registered in the COMMANDS table by
# the @command decorator, so /help, the typo suggestion, and the dispatch in
# main() all read from one place. A handler receives (session, args, line) —
# line is the whole input line, word-matched, so trailing text after the
# no-argument commands ("/exit now") is ignored rather than slipping through
# to the model — and returns:
#   None    handled; nothing is sent to the model, keep reading input
#   Submit  a message to send to the model, with its plan/read-only marking
#   EXIT    stop the loop
# The commands that send nothing leave a pending paste pending; any Submit
# consumes it. A new command is one function plus one decorator.
EXIT = object()


@dataclass(frozen=True)
class Command:
    word: str        # what the user types: "/model"
    usage: str       # how /help spells it: "/model [name]"
    help: str        # the /help description, taken from the handler's docstring
    handler: Callable


COMMANDS: dict[str, Command] = {}


def command(word: str, usage: str | None = None, aliases: tuple[str, ...] = ()) -> Callable:
    """Register a prompt command handler under its word and its aliases."""

    def register(handler: Callable) -> Callable:
        entry = Command(word=word, usage=usage or word, help=handler.__doc__.strip(), handler=handler)
        for name in (word, *aliases):
            COMMANDS[name] = entry
        return handler

    return register


def print_commands() -> None:
    """The /help listing: each registered command's usage and description."""
    entries = []
    for entry in COMMANDS.values():  # insertion order; aliases share one entry
        if entry not in entries:
            entries.append(entry)
    width = max(len(entry.usage) for entry in entries)
    print()
    for entry in entries:
        print(f"  {style.bold(entry.usage.ljust(width))}  {entry.help}")


# The names /model lists (and marks as known): Ollama Cloud's by default, or the
# --api-config file's "models" list when that flag selects the Anthropic-compatible
# endpoint. Replaced by main() at startup; any name is accepted either way, /model
# <name> just notes when the name is not in the list.
MODEL_CHOICES: tuple = OLLAMA_MODELS


def print_models(current: str) -> None:
    """The /model list: the models /model can switch to, with the current one marked."""
    print()
    for name in MODEL_CHOICES:
        line = style.model_label(name)
        if name == current:
            line += style.dim("  (current)")
        print(f"  {line}")
    if not MODEL_CHOICES:
        print(style.dim("  [no model list configured; /model <name> accepts any name]"))


def unknown_command_error(word: str) -> str:
    """The rejection message for a prompt line whose first word is not a command.

    Such a line — /page, /jump, a bare "/" — is not sent to the model as
    conversation text (in plan mode it would silently have revised the plan).
    The nearest real command by spelling (difflib) is suggested, so /page
    points at /paste, and the /help pointer lists them all.
    """
    close = difflib.get_close_matches(word, sorted(COMMANDS), n=1, cutoff=0.6)
    hint = f" ~ did you mean {close[0]}?" if close else ""
    return f"command not recognized: {word}{hint} (/help for all commands)"


def read_key() -> str:
    """Read the Ollama API key, exiting with guidance when it is missing or empty."""
    what = "Ollama API key"
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        sys.exit(f"No key file found. Put your {what} in {KEY_FILE}")
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"Could not read {KEY_FILE}: {e}")
    if not key:
        sys.exit(f"{KEY_FILE} is empty. Put your {what} in it.")
    # POSIX-only: Windows st_mode always reports 0666 without reflecting the
    # file's ACLs, so there the check would warn on every launch whatever the
    # real protection; on POSIX the bits mean what they say.
    if sys.platform != "win32" and KEY_FILE.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(style.warning(f"Warning: {KEY_FILE} can be read by other users. Run: chmod 600 {KEY_FILE}") + "\n")
    return key


def make_approver(auto_approve: bool):
    """Build the write approver the harness passes to the Workspace.

    The approver is the one human control over the model's writes: it shows the
    coloured diff of just the areas that changed and asks, or with --yes prints
    an auto-approved note instead of asking. diff_lines is the pre-rendered
    output of diff.render_diff; an empty list means nothing changes at line
    level (e.g. a new empty file). risk_note, when set, flags a write to a file
    that may be executed later (a build or test entry file): the diff it sits
    above is the user's one chance to check it.
    """

    def approve(message: str, diff_lines: list[str], risk_note: str | None = None) -> bool:
        if auto_approve:
            print(f"  {style.dim('[auto-approved]')} {style.sanitize(message)}")
            if risk_note:
                print(f"  {style.warning(risk_note)}")
            return True
        print(f"\n  {style.paint(message, style.BOLD, style.YELLOW)}")
        if risk_note:
            print(f"  {style.warning(risk_note)}")
        for line in diff_lines:
            print(f"  {style.dim('|')} {line}")
        # Like every prompt the user types at, the question is preceded by a blank line.
        print()
        try:
            answer = input(style.prompt("  Allow this write? [y/N] ", style.BOLD)).strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")

    return approve


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


@command("/help")
def cmd_help(session, args, line):
    """show this list of commands"""
    print_commands()


@command("/paste")
def cmd_paste(session, args, line):
    """paste long text into an external editor"""
    # Creates a fresh empty buffer when nothing is pending and only reopens the
    # file when something is, so an unsent draft is kept. Works while a plan is
    # open and sends nothing to the model, so the pending paste stays pending
    # until some message is actually submitted. Anything typed after the word is
    # ignored.
    path = session.open_paste()
    if path:
        open_in_editor(path)
        print(style.dim(f"  [pasting into file: {path}]"))


def _set_thinking(session, on: bool) -> None:
    # "off" stops thinking server-side as well as hiding it; turning it back on
    # restores the startup --think setting, which may be a level
    # (low/medium/high) rather than plain "on". In plan mode the setting is
    # applied and the `plan>` prompt returns with its pending plan intact.
    # Anything typed after the command word is ignored, and nothing is sent to
    # the model.
    session.agent.show_thinking = on
    session.agent.client.think = session.think if on else "off"
    print(style.dim(f"  [thinking {'on' if on else 'off'}]"))


@command("/thinking-on")
def cmd_thinking_on(session, args, line):
    """turn thinking on for all later messages"""
    _set_thinking(session, True)


@command("/thinking-off")
def cmd_thinking_off(session, args, line):
    """turn thinking off for all later messages"""
    _set_thinking(session, False)


@command("/model", usage="/model [name]")
def cmd_model(session, args, line):
    """list the models, or switch to <name> for later messages"""
    # With no argument, list the models it can switch to (the current one
    # marked); with a name, switch for all later messages, accepting any model
    # name, like --model at startup. Also works while a plan is open, and
    # nothing is sent to the model.
    name = line[len("/model") :].strip()
    if not name:
        print_models(session.agent.model)
        return
    session.agent.set_model(name)
    print(style.dim(f"  [model set to {name}]"))
    if name not in MODEL_CHOICES:
        print(style.dim("  [not in the /model list; the name is still sent to the API as given]"))


@command("/talk", usage="/talk <question>")
def cmd_talk(session, args, line):
    """ask the model something without any files being changed"""
    text = line[len("/talk") :].strip()
    if not text:
        print(style.dim("  [usage: /talk <question>; the model can read files but not change them]"))
        return
    if not args.talk:
        # With --talk the read-only state was already announced at startup.
        print(style.dim("  [read-only: no files will be changed]"))
    # Never a plan update: a side question stays a side question.
    return Submit(text, read_only=True)


@command("/plan", usage="/plan <task>")
def cmd_plan(session, args, line):
    """propose an implementation plan for the task, changing no files"""
    text = line[len("/plan") :].strip()
    if not text:
        print(style.dim("  [usage: /plan <task>; the model proposes a plan and changes no files]"))
        return
    print(style.dim("  [plan mode: the model will propose what to implement; no files will be changed]"))
    return Submit(text, plan="start")


@command("/ok")
def cmd_ok(session, args, line):
    """approve the current plan and have the model implement it"""
    if not session.plan_mode:
        print(style.dim("  [no plan to accept — start one with /plan <task>]"))
        return
    print(style.dim("  [plan approved]"))
    return Submit("", plan="approved")  # the approval note is all the model needs


@command("/cancel")
def cmd_cancel(session, args, line):
    """leave plan mode without implementing the plan"""
    if not session.plan_mode:
        print(style.dim("  [not in plan mode]"))
        return
    session.plan_mode = False
    print(style.dim("  [plan mode off; the plan stays in the conversation]"))


@command("/continue")
def cmd_continue(session, args, line):
    """resume after a reply cut off by the output limit"""
    # The fixed nudge replaces the user text; with a plan open it revises the
    # plan instead, and pending paste quoting, error rollback, and usage
    # printing apply as for any message. agent.py's max_tokens handling drops
    # any cut-off tool call first, so the history it resumes from is valid.
    # Anything typed after the command word is ignored.
    submit = Submit(CONTINUE_MESSAGE)
    if session.plan_mode:
        return session.as_plan_update(submit)
    return submit


@command("/reset")
def cmd_reset(session, args, line):
    """clear the conversation and start over"""
    # Also clears the planning state and any pending paste, and reverts the
    # prompt label. Anything typed after the command word is ignored.
    session.reset()


@command("/exit", aliases=("/quit",))
def cmd_exit(session, args, line):
    """leave the harness (also: /quit)"""
    return EXIT


def plain_submit(session, args, line) -> Submit:
    """An ordinary (non-command) message: sent as-is, or as plan feedback while planning.

    With --talk every message is read-only; while a plan is open, ordinary
    messages are feedback that revises it (/talk stays a side question).
    """
    submit = Submit(line, read_only=args.talk)
    if session.plan_mode:
        return session.as_plan_update(submit)
    return submit


def main() -> None:
    # Rebinds the module's MODEL_CHOICES when --api-config supplies its own list.
    global MODEL_CHOICES
    # Clear the console before anything is printed, including argparse usage
    # errors and the key-file warning (which used to precede the splash).
    style.clear_screen()
    parser = argparse.ArgumentParser(description="Locked-down file-only coding agent for Ollama Cloud.")
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
        "--model",
        default=None,
        help=f"model to use; with --api-config the config's \"model\" is the default, "
        f"without it an Ollama Cloud model, which the cloud catalogue decides what exists "
        f"(default: {OLLAMA_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-config",
        metavar="FILE",
        type=Path,
        help="use an Anthropic-compatible /v1/messages endpoint described by this JSON file "
        "(base_url and api_key required; model, models and headers optional) instead of "
        "Ollama Cloud. The default is config/api_config.json next to loom.py, and that "
        "file selects the endpoint when present; an explicitly named file must exist. "
        "The file holds the key, so it lives in config/, hidden from the model",
    )
    parser.add_argument(
        "--think",
        choices=["on", "off", "low", "medium", "high"],
        help="thinking for models that support it. Use a level (low/medium/high) "
        "for gpt-oss models, and 'off' for models that cannot think. With "
        "--api-config the levels set an extended-thinking token budget "
        "(default: on)",
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
    parser.add_argument(
        "--redact-file",
        type=Path,
        default=CONFIG_DIR / REDACT_FILE,
        help=f"the strings listed under 'redactList' are replaced with the file's 'replaceWith' "
        f"marker ('<redacted>' by default) in everything sent to the model "
        f"(default: {REDACT_FILE} in config/, next to loom.py)",
    )
    args = parser.parse_args()

    # --api-config swaps Ollama Cloud for an Anthropic-compatible /v1/messages
    # endpoint. Without the flag, the default file in config/ selects that
    # endpoint by existing — absent, it is the normal opt-out, quiet like
    # redaction's missing file; with the flag, the file the user named must be
    # there (the loader's note says so). The file is read here, before any
    # client exists, because its "model" is the default --model and its
    # "models" is the /model list below; a problem with it stops the harness.
    if not args.api_config and API_CONFIG_FILE.is_file():
        args.api_config = API_CONFIG_FILE
    endpoint_config = {}
    if args.api_config:
        endpoint_config, config_error = load_endpoint_config(args.api_config)
        if config_error:
            sys.exit(config_error)
        if not endpoint_config.get("model"):
            sys.exit(f'{args.api_config} names no model: add a "model" (or a "models" list) or pass --model')
        MODEL_CHOICES = tuple(endpoint_config.get("models") or ())
    args.model = args.model or endpoint_config.get("model") or OLLAMA_DEFAULT_MODEL
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

    # The redaction list is read before anything else touches the model: the
    # filter it configures wraps every text that joins the conversation
    # (core/agent.py). A missing file is the normal opt-out; one that exists
    # but cannot be loaded (unreadable, invalid JSON, wrong shape) is reported
    # next to the splash and text is sent unredacted — never a silent failure.
    redaction_strings, redaction_marker, redaction_error = load_redactions(args.redact_file)
    redactor = Redactor(redaction_strings, redaction_marker)

    workspace = Workspace(
        root,
        approver=make_approver(args.yes),
        fallback_encoding=args.fallback_encoding,
        # search and read_file redact configured strings BEFORE their length cuts
        # (search's 200-char match cap, read_file's page cap), so a secret
        # straddling a truncation boundary is never shown as a partial fragment;
        # agent.py still redacts every tool result again afterward.
        redactor=redactor,
        # The paste buffer is off-limits to the model: it can neither read the
        # pending paste out of the harness's .loom-temp/paste.txt nor plant text
        # there to be quoted into a later "user" message. The buffer lives outside
        # the workspace, so containment already keeps it out of reach; the blocked
        # entry stays as defense in depth for a workspace that contains the
        # harness folder itself (e.g. this repo used as a project). workspace.py
        # resolves every blocked entry with Path.resolve(), which succeeds for a
        # file that does not exist yet, so this holds even before the first /paste
        # creates the file.
        # A --redact-file or --api-config named outside config/ is blocked the
        # same way (its path can be anywhere the user chose); the ones in
        # config/ need no entry, because the whole directory is protected below.
        blocked=(TEMP_DIR / PASTE_FILE, args.redact_file)
        + ((args.api_config,) if args.api_config else ()),
        # config/ holds the API keys and the redaction list, so it is refused
        # outright — not listed, not readable, not writable, not searchable —
        # wherever it sits relative to the workspace: by path, not by name, so
        # a project's own "config" directory stays reachable.
        protected_dirs=(CONFIG_DIR,),
    )
    # The client talks to whichever API was chosen: Ollama Cloud by default, or
    # the --api-config endpoint (whose key comes from its file, so ollama_key.txt
    # is not asked for). Both take the same think setting and expose the same
    # interface to the Agent.
    agent = Agent(
        AnthropicClient(endpoint_config, think=args.think)
        if args.api_config
        else OllamaClient(read_key(), think=args.think),
        workspace,
        model=args.model,
        show_thinking=not args.hide_thinking,
        redactor=redactor,
    )

    splash.print_splash()
    print(f"{style.bold('Workspace:')} {workspace.root}")
    print(f"{style.bold('Model:')}     {style.model_label(args.model)}")
    if args.think:
        print(f"{style.bold('Thinking:')}  {args.think} {style.dim('(toggle with /thinking-on, /thinking-off)')}")
    if redaction_error:
        # The config exists but is unreadable: say so instead of failing to
        # start, but never quietly drop the redaction the user asked for.
        print(f"{style.bold('Redaction:')} off {style.warning(f'({redaction_error}; text is sent unredacted)')}")
    elif len(redactor):
        print(
            f"{style.bold('Redaction:')} {len(redactor)} string{'s' if len(redactor) != 1 else ''}"
            f" replaced with {redaction_marker} {style.dim(f'({args.redact_file.name})')}"
        )
    else:
        state = "no entries yet" if args.redact_file.exists() else "file not found"
        print(f"{style.bold('Redaction:')} none {style.dim(f'({args.redact_file}: {state})')}")
    print()
    print(style.dim("The model can read and write files here only. No commands can be run, and .git is protected."))
    if args.talk:
        print(style.warning("Read-only mode: every message behaves like /talk; no files can be changed."))
    else:
        print(style.dim("Start a message with /talk to ask, or /plan to get a plan to approve, without any files being changed."))
    print(style.dim("Type /help to list commands, /reset to clear the conversation, /exit to quit."))

    # The loop's state: the planning mode and the pending /paste buffer, with
    # the prompt label, the plan file, and the paste quoting all owned by one
    # place (core/session.py).
    session = Session(agent, args.think)

    while True:
        try:
            # A blank line always separates the prompt from whatever came before it
            # (the splash, a command's note, the tail of a model turn), so no other
            # output needs to end with a blank line of its own. The prompt label is
            # rebuilt every iteration so /thinking-on, /thinking-off, /plan and /ok
            # take effect on the very next prompt.
            print()
            user_text = input(session.prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        # The first word decides. A line starting with "/" is a command attempt;
        # if its first word matches no command — /page, /jump, a bare "/" — it
        # is a typo, rejected here instead of being sent to the model as normal
        # conversation text (in plan mode it would silently revise the plan).
        # The note points at /help and suggests the closest command by spelling.
        # Nothing below runs, so a pending paste stays pending.
        word = user_text.split(None, 1)[0]
        if user_text.startswith("/"):
            if word not in COMMANDS:
                print(style.warning(f"  [{unknown_command_error(word)}]"))
                continue
            result = COMMANDS[word].handler(session, args, user_text)
        else:
            result = plain_submit(session, args, user_text)
        if result is EXIT:
            break
        if not isinstance(result, Submit):
            continue
        # The pending paste is quoted in front of the submitted text at the last
        # moment, so whatever was last saved in the editor is what the model
        # sees. This counts as the submission for /plan, plan revisions, /ok and
        # /talk too, and the flag is consumed before sending whatever the turn's
        # outcome, so a failed turn does not re-paste on the next attempt.
        if session.paste_pending:
            session.with_paste(result)
        history_len = len(agent.messages)
        try:
            agent.send(result.text, read_only=result.read_only, plan=result.plan)
            # Plan state advances only once the turn fully succeeded, so a failed
            # or interrupted turn leaves the previous state in place.
            session.advance(result.plan)
        except KeyboardInterrupt:
            print("\n" + style.warning("  [interrupted]"))
            del agent.messages[history_len:]
        except APIError as e:
            if e.status == 401:
                # Which credential source to check depends on the API in use: the
                # --api-config file's key, or the Ollama key file.
                source = args.api_config if args.api_config else KEY_FILE
                print(style.error(f"  [authentication failed: check the API key in {source}]"))
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
