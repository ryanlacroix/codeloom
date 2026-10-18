#!/usr/bin/env python3
"""A minimal, locked-down agentic coding harness for Claude.

Claude gets exactly three tools, all confined to a single workspace directory:
  - list_directory: list entries in a directory
  - read_file:      read a text file (UTF-8, or the fallback encoding, cp1252 by default)
  - write_file:     create or overwrite a text file (asks you first); existing files
                    keep their original encoding and line endings

There is deliberately no tool for running commands, and no module here ever
imports subprocess or calls os.system / exec / eval. Claude can only read and
write files inside the workspace; any path that resolves outside it (via "..",
absolute paths, or symlinks) is rejected.

Auth: put your access token in key.txt next to these scripts (sent as a Bearer token).
Claude is never allowed to read or write key.txt, even if the workspace contains it.

Uses only the Python standard library; the API is called directly over HTTPS.
"""

import argparse
import codecs
import stat
import sys
from pathlib import Path

from agent import Agent
from api_client import APIConnectionError, APIError, Client
from models import DEFAULT_MODEL, MODELS
from workspace import Workspace

KEY_FILE = Path(__file__).resolve().parent / "key.txt"


def make_client() -> Client:
    try:
        token = KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        sys.exit(f"No key file found. Put your Claude access token in {KEY_FILE}")
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"Could not read {KEY_FILE}: {e}")
    if not token:
        sys.exit(f"{KEY_FILE} is empty. Put your Claude access token in it.")
    if KEY_FILE.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(f"Warning: {KEY_FILE} can be read by other users. Run: chmod 600 {KEY_FILE}\n")
    return Client(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked-down file-only coding agent for Claude.")
    parser.add_argument("dir", help="project directory Claude may access (required)")
    parser.add_argument("--yes", action="store_true", help="approve file writes without asking")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--fallback-encoding",
        default="cp1252",
        help="encoding assumed for files that are not valid UTF-8 (default: cp1252)",
    )
    args = parser.parse_args()

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
        root, auto_approve_writes=args.yes, fallback_encoding=args.fallback_encoding, blocked=(KEY_FILE,)
    )
    agent = Agent(make_client(), workspace, model=args.model)

    print(f"Workspace: {workspace.root}")
    print(f"Model:     {args.model}")
    print("Claude can read and write files here only. No commands can be run.")
    print("Type /reset to clear the conversation, /exit to quit.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/reset":
            agent.messages = []
            print(f"  [conversation cleared; ~${agent.session_usage.cost:.4f} spent this session]")
            continue

        history_len = len(agent.messages)
        try:
            agent.send(user_text)
        except KeyboardInterrupt:
            print("\n  [interrupted]")
            del agent.messages[history_len:]
        except APIError as e:
            if e.status == 401:
                print(f"  [authentication failed: check the token in {KEY_FILE}]")
            elif e.status == 429:
                print("  [rate limited: wait a moment and try again]")
            else:
                print(f"  [API error {e.status}: {e.message}]")
            del agent.messages[history_len:]
        except APIConnectionError as e:
            print(f"  [could not reach the API: {e}]")
            del agent.messages[history_len:]


if __name__ == "__main__":
    main()
