"""Terminal output control for the harness: colors, plus the console clear at startup.

Colors are used only when stdout is a terminal, and can be turned off with the
NO_COLOR environment variable (https://no-color.org) or forced on with FORCE_COLOR.
"""

import os
import sys


def colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        return False
    # The classic Windows console shows escape codes literally; Windows Terminal handles them.
    return sys.platform != "win32" or "WT_SESSION" in os.environ


ENABLED = colors_enabled()


def clear_screen() -> None:
    """Wipe the console, used once when the harness starts, before anything else
    is printed, so every run begins on a clean screen.

    Like the modern "clear" command: moves the cursor home, clears the visible
    screen, and erases the scrollback as well (the \\033[3J part; terminals that
    predate it simply ignore that sequence). Skipped unless escape codes are safe
    to write, per the same ENABLED gate as colors: redirected or piped output and
    the classic Windows console stay free of control-code litter."""
    if not ENABLED:
        return
    sys.stdout.write("\033[H\033[2J\033[3J")
    sys.stdout.flush()


def paint(text: str, *codes: str) -> str:
    if not ENABLED or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def prompt(text: str, *codes: str) -> str:
    """Like paint, for input() prompts: readline must be told the escape codes take no space,
    or the cursor lands in the wrong column when editing the line."""
    if not ENABLED:
        return text
    if "readline" not in sys.modules:
        return paint(text, *codes)
    return f"\001\033[{';'.join(codes)}m\002{text}\001\033[0m\002"


BOLD, DIM = "1", "2"
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN = "31", "32", "33", "34", "35", "36"
LIGHT_BLUE = "94"


def bold(text: str) -> str:
    return paint(text, BOLD)


def dim(text: str) -> str:
    return paint(text, DIM)


def error(text: str) -> str:
    return paint(text, RED)


def warning(text: str) -> str:
    return paint(text, YELLOW)


def success(text: str) -> str:
    return paint(text, GREEN)


def model_label(text: str) -> str:
    return paint(text, BOLD, CYAN)


def tool_name(text: str) -> str:
    return paint(text, MAGENTA)
