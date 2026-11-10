"""Terminal output control for the harness: colors, console-bound text sanitizing,
plus the console clear at startup.

Colors are used only when stdout is a terminal, and can be turned off with the
NO_COLOR environment variable (https://no-color.org) or forced on with FORCE_COLOR.
Windows consoles are switched into ANSI mode first (see _enable_vt_processing), so
the classic console host that PowerShell and cmd open in colors up just like
Windows Terminal does; a console that cannot be switched stays in plain text.
"""

import os
import re
import sys

# consoleapi.h's mode bit that makes the Windows console host interpret ANSI escape
# codes (the same ones every other terminal understands) instead of printing them
# as literal text.
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


def colors_enabled() -> bool:
    """Whether escape codes may be written to stdout.

    The switches stack in this order: NO_COLOR beats everything; FORCE_COLOR forces
    past the terminal check, on the user's say-so (the name is the warning); then
    stdout must be a terminal that is not dumb. The only remaining open question is
    the Windows console's, settled by flipping it into ANSI mode and trusting the
    flip (see _enable_vt_processing) — not by allowlisting terminal products, which
    is what used to switch colors off in PowerShell sessions outside Windows
    Terminal, splash included.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        return False
    if sys.platform != "win32":
        return True
    # ConEmu (ConEmuANSI=ON) and ANSICON translate the codes themselves, so there may
    # be no console switch to flip (e.g. on an older Windows): trust them rather than
    # let the probe veto colors they already support.
    if os.environ.get("ConEmuANSI", "").lower() == "on" or os.environ.get("ANSICON"):
        return True
    return _enable_vt_processing()


def _enable_vt_processing() -> bool:
    """Switch the stdout console into ANSI (virtual-terminal) mode; True when escape
    codes can now be trusted on stdout.

    The console Windows gives Python for stdout — the classic conhost behind
    PowerShell and cmd, or the pseudoconsole Windows Terminal and the VS Code
    terminal serve out — honors ANSI escape codes, but ships with the processing
    switch (ENABLE_VIRTUAL_TERMINAL_PROCESSING) off, and Python never flips it
    itself: until it is set, colored output prints its codes as literal text. That
    is why colors used to be allowed on Windows only under Windows Terminal
    (WT_SESSION), and everything else — PowerShell windows, notably — got plain
    text, which suppressed the splash too. Probing the console replaces that
    per-product guess: whatever accepts the switch (every modern host) is declared
    escapable, and whatever rejects it (a Windows-pre-10 conhost; stdout with no
    console behind it) is not.

    Best effort, so callers fall back to plain text on any doubt: False when ctypes
    or windll is missing (a stripped or odd build, not Windows at all), when
    GetConsoleMode refuses the handle (no console behind stdout — redirected or
    piped output, which the isatty gate already covered), or when SetConsoleMode
    refuses the bit. Idempotent: setting a mode that already contains the bit is a
    harmless success.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # windll exists on Windows only; AttributeError -> not Windows
    except (ImportError, AttributeError, OSError):
        return False
    # Explicit handle/mode types: ctypes' int restype would truncate a 64-bit HANDLE.
    kernel32.GetStdHandle.argtypes = (ctypes.c_int,)
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.GetConsoleMode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel32.GetConsoleMode.restype = ctypes.c_int
    kernel32.SetConsoleMode.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetConsoleMode.restype = ctypes.c_int
    conout = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE: the handle prints go through
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(conout, ctypes.byref(mode)):
        return False  # stdout has no console behind it
    return bool(kernel32.SetConsoleMode(conout, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))


ENABLED = colors_enabled()


def clear_screen() -> None:
    """Wipe the console, used once when the harness starts, before anything else
    is printed, so every run begins on a clean screen.

    Like the modern "clear" command: moves the cursor home, clears the visible
    screen, and erases the scrollback as well (the \\033[3J part; terminals that
    predate it simply ignore that sequence). Skipped unless escape codes are safe
    to write, per the same ENABLED gate as colors: redirected or piped output, and
    any console the ANSI-mode switch could not be flipped on (see colors_enabled),
    stay free of control-code litter."""
    if not ENABLED:
        return
    sys.stdout.write("\033[H\033[2J\033[3J")
    sys.stdout.flush()


# Control characters stripped from every console-bound string, wherever
# model-controlled data can reach it (diff lines, tool-call and error lines,
# the streamed reply), plus ESC, which is replaced with its visible symbol
# instead of arriving raw: C0 controls other than \n and \t, C1 controls (some
# terminals honor them without an ESC prefix), and the bidi-override
# characters that can visually reorder text such as a diff.
_CONTROLS = re.compile("\x1b|[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f\x80-\x9f\u202a-\u202e\u2066-\u2069]")
_VISIBLE_REPLACEMENT = {"\x1b": "␛"}


def sanitize(text: str) -> str:
    """Make console-bound text terminal-safe.

    Crafted file content or model output could otherwise carry escape
    sequences that clear the screen, move the cursor, overwrite the clipboard
    (OSC 52), or right-to-left overrides that visually reorder, say, a diff.
    Painted strings sanitize their payload here, and the streaming renderer
    sanitizes where the model's text enters it (MarkdownStream.feed), so a
    control a terminal would act on never reaches the screen. This is
    display-only: text sent to the API and content written to files are never
    altered.
    """
    return _CONTROLS.sub(lambda match: _VISIBLE_REPLACEMENT.get(match.group(), ""), text)


def paint(text: str, *codes: str) -> str:
    """Style one string for the console, sanitizing it first.

    The text is often model-controlled (a diff line, a path, a chunk of the
    streamed reply) and reaches the terminal verbatim whether colors are on or
    off, so it is sanitized here rather than at every call site; the style
    codes are generated by us and are never contaminated.
    """
    text = sanitize(text)
    if not ENABLED or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def prompt(text: str, *codes: str) -> str:
    """Like paint, for input() prompts: readline must be told the escape codes take no space,
    or the cursor lands in the wrong column when editing the line."""
    text = sanitize(text)
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
