#!/usr/bin/env python3
"""A minimal, locked-down agentic coding harness for Claude or Ollama Cloud models.

The model gets exactly three tools, all confined to a single workspace directory:
  - list_directory: list entries in a directory
  - read_file:      read a text file (UTF-8, or the fallback encoding, cp1252 by default)
  - write_file:     create or overwrite a text file (asks you first); existing files
                    keep their original encoding and line endings

There is deliberately no tool for running commands, and no module here ever
imports subprocess or calls os.system / exec / eval. Claude can only read and
write files inside the workspace; any path that resolves outside it (via "..",
absolute paths, or symlinks) is rejected.

Auth: put your Claude access token in key.txt, or your Ollama API key in ollama_key.txt,
next to these scripts (sent as a Bearer token). The model is never allowed to read or write
either key file, even if the workspace contains it.

Uses only the Python standard library; the API is called directly over HTTPS.
"""

import argparse
import codecs
import stat
import sys
from pathlib import Path

import style
from agent import Agent
from api_client import APIConnectionError, APIError, Client
from models import DEFAULT_MODEL, MODELS, OLLAMA_DEFAULT_MODEL
from ollama_client import OllamaClient
from workspace import Workspace

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

# Plans are saved here, under the project root, so they survive restarts and can be
# read or diffed outside the conversation. The directory is kept out of git by a
# .gitignore entry that is added automatically the first time a plan is saved.
PLAN_DIR = ".slim-harness-temp"
PLAN_FILE = "plan.md"

# Slash commands typed at the "you>" prompt; /help lists these.
COMMANDS = [
    ("/help", "show this list of commands"),
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
    print()


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


def save_plan(root: Path, agent: Agent) -> None:
    """Save the newest plan reply to .slim-harness-temp/plan.md.

    Trusted harness code writes this directly (not through the model's sandboxed
    tools), so saving needs no approval, and the directory is created if missing.
    Failures are reported but never fatal.
    """
    text = agent.last_reply_text()
    if not text.strip():
        print(style.dim("  [the reply had no plan text; nothing saved]"))
        return
    text = text if text.endswith("\n") else text + "\n"
    try:
        (root / PLAN_DIR).mkdir(parents=True, exist_ok=True)
        (root / PLAN_DIR / PLAN_FILE).write_text(text, encoding="utf-8", newline="\n")
    except OSError as e:
        print(style.warning(f"  [could not save the plan to {PLAN_DIR}/{PLAN_FILE}: {e}]"))
        return
    print(style.dim(f"  [plan saved to {PLAN_DIR}/{PLAN_FILE}]"))
    ensure_gitignored(root)


def ensure_gitignored(root: Path) -> None:
    """Add .slim-harness-temp/ to .gitignore unless a spelling of it is already listed.

    The file is created if it does not exist. Existing bytes are kept as they are:
    the entry is appended, so no re-encoding of the rest of the file can corrupt it.
    """
    gitignore = root / ".gitignore"
    try:
        data = gitignore.read_bytes()
    except FileNotFoundError:
        data = b""
    except OSError as e:
        print(style.warning(f"  [could not read {gitignore}: {e}]"))
        return
    # Any spelling counts as present: with or without a leading or trailing slash.
    if any(line.strip().strip("/") == PLAN_DIR for line in data.decode("utf-8", "replace").splitlines()):
        return
    try:
        with gitignore.open("ab") as f:
            f.write((b"" if not data or data.endswith(b"\n") else b"\n") + f"{PLAN_DIR}/\n".encode("utf-8"))
    except OSError as e:
        print(style.warning(f"  [could not update {gitignore}: {e}]"))
        return
    print(style.dim(f"  [.gitignore now ignores {PLAN_DIR}/]"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked-down file-only coding agent for Claude or Ollama Cloud.")
    parser.add_argument("dir", help="project directory the model may access (required)")
    parser.add_argument("--yes", action="store_true", help="approve file writes without asking")
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
        root, auto_approve_writes=args.yes, fallback_encoding=args.fallback_encoding, blocked=tuple(path for path, _ in KEY_FILES.values())
    )
    agent = Agent(
        make_client(args.provider, args.think), workspace, model=args.model, show_thinking=not args.hide_thinking
    )

    print(f"{style.bold('Workspace:')} {workspace.root}")
    print(f"{style.bold('Model:')}     {style.model_label(args.model)} {style.dim(f'({args.provider})')}")
    if args.think:
        print(f"{style.bold('Thinking:')}  {args.think}")
    print(style.dim("The model can read and write files here only. No commands can be run."))
    if args.talk:
        print(style.warning("Read-only mode: every message behaves like /talk; no files can be changed."))
    else:
        print(style.dim("Start a message with /talk to ask, or /plan to get a plan to approve, without any files being changed."))
    print(style.dim("Type /help to list commands, /reset to clear the conversation, /exit to quit.") + "\n")

    # While a plan is being discussed (after /plan, until it is approved or abandoned),
    # ordinary messages revise the plan instead of asking for work, write_file stays
    # disabled, and the prompt is "plan>" to make the mode visible.
    plan_mode = False

    while True:
        try:
            if plan_mode:
                user_text = input(style.prompt("plan> ", style.BOLD, style.MAGENTA)).strip()
            else:
                user_text = input(style.prompt("you> ", style.BOLD, style.GREEN)).strip()
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
            if agent.priced:
                print(style.dim(f"  [conversation cleared; ~${agent.session_usage.cost:.4f} spent this session]"))
            else:
                print(style.dim("  [conversation cleared]"))
            continue
        if user_text == "/help":
            print_commands()
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

        history_len = len(agent.messages)
        try:
            agent.send(user_text, read_only=read_only, plan=plan)
            # Plan state advances only once the turn fully succeeded, so a failed or
            # interrupted turn leaves the previous state in place.
            if plan == "approved":
                plan_mode = False
            elif plan in ("start", "update"):
                plan_mode = True
                save_plan(root, agent)
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