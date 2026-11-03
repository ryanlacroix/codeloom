"""Multiline-paste handling for the harness prompt: capture, not fragmentation.

Pasting several lines into a plain input() prompt tears the text apart: the
terminal delivers the whole block at once, input() returns the first line as a
complete message, and the remaining lines are then consumed one per prompt,
each sent (or misread as a slash command) as its own message.

read_prompt() reads the first line with input() as usual, then — when input is
arriving in a burst, which is what a paste looks like — briefly switches the
terminal to cbreak and takes the rest of the input queue as well, returning the
full text as one string. Everything queued is returned as pasted, apart from
terminal escape sequences (bracketed-paste markers and similar CSI sequences),
which are stripped so control litter cannot reach the model. On a terminal the
echoed block is then redrawn as one prompt line ending in [multiline paste]
(PLACEHOLDER), so the transcript shows the marker instead of the pasted text.
read_prompt() returns the message as the terminal shows it plus the mapping
that puts the pasted text back in; the harness applies the mapping just before
a message is sent, so the model receives the text in full.

Only a real terminal sees any of this. When stdin is piped or redirected,
read_prompt() is input() verbatim, and where low-level terminal control is
missing the wrapper degrades to plain input() input rather than failing.
"""

import os
import re
import shutil
import sys
import time
from pathlib import Path

import style

try:
    import termios  # POSIX terminal control
except ImportError:  # Windows and every other non-POSIX platform
    termios = None

# A burst stops counting as one paste once the input queue has been empty for
# this long. Reading is delayed only while bytes are still trickling in after
# the first line, so a normally typed single-line message pays nothing.
QUIET_SECONDS = 0.2

# Hard cap on how long leftover input may keep being collected, so a
# pathological stream of input cannot hold a prompt indefinitely.
CAP_SECONDS = 3.0

# Shown on the terminal in place of a pasted multiline block.
PLACEHOLDER = "[multiline paste]"

# readline tells itself these bytes take no space; they appear in prompt
# strings (never in pasted text) and must go before a prompt is printed
# directly instead of handed to input(), or they show up as visible litter.
_READLINE_MARKS = str.maketrans("", "", "\x01\x02")

# CSI escape sequences: ESC [, parameter bytes, final byte. Terminals wrap a
# paste in "\x1b[200~" ... "\x1b[201~" (bracketed paste mode); readline builds
# that do not understand the mode can let the markers through into the text.
_CSI = re.compile(r"\x1b\[[0-9;:<=>?]*[A-Za-z~]")

# A captured paste is saved here as paste-NN.txt, under the harness's scratch
# directory (loom.py already keeps .portage-temp/ out of git).
PASTES_SUBDIR = "pastes"


def read_prompt(prompt: str, paste_dir: Path | None = None) -> tuple[str, dict[str, str]]:
    """input(), but a paste of several lines comes back as one string.

    The first argument is the same styled prompt string loom.py already
    builds, so this path looks and behaves identically to the user.

    A multiline paste is shown on the prompt line as [multiline paste] (the
    marker the terminal transcript carries) and — with paste_dir given —
    saved as byte-exact text (line endings normalized to \n) to the next free
    paste-NN.txt under it, with a note of where. What comes back is a pair: the message as the terminal
    shows it, plus the mapping capture() made, which the harness applies
    just before sending so the model still receives the text in full.
    """
    line = input(prompt)
    if not is_terminal():
        return line, {}  # scripted or piped input: no queue, no pastes
    # A paste's line endings are whatever the clipboard carried, and readline
    # holds the tty in raw mode for its own reads (with ICRNL cleared), so
    # unlike canonical input nothing on the way in turns \r into \n. A \r- or
    # \r\n-separated paste then contains no \n at all: the checks below never
    # see a multiline paste — no marker is shown, nothing is saved, and the
    # model receives every pasted line glued into one flat string, which is
    # the very result this module exists to prevent. No single-line entry ever
    # legitimately contains \r, so normalizing here can only fix pastes.
    line = line.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in line:
        # readline itself got the whole paste (bracketed paste mode) and
        # returned it in one go: there is nothing left in the queue.
        text = clean(line)
    else:
        fd = stdin_fd()
        if fd is None or not readable(fd, 0.0):
            return clean(line), {}  # nothing queued: ordinary single-line input
        text = clean(line + drain(fd))
    if "\n" in text and text.strip():
        return capture(text, paste_dir, prompt)
    return text, {}  # whitespace-only paste: no message, nothing saved


def capture(raw: str, paste_dir: Path | None, prompt: str = "") -> tuple[str, dict[str, str]]:
    """Reduce a multiline paste on screen to [multiline paste] and save it.

    Returns the message as the terminal will show it together with the
    mapping the harness swaps back in before sending, so the model receives
    the paste in full while the transcript keeps the short marker.

    Normally the whole input is the paste and the visible text is the marker
    alone. When the input begins with a slash command, that first line stays
    visible — and keeps working as a command — while the pasted remainder is
    reduced to the marker. If the paste already contains the marker text,
    collapsing it would swap back at the wrong place, so the message is left
    as typed (it is still saved).

    The paste is on screen by now, echoed by the terminal as it arrived, so
    the marker redraw follows the same rule as the collapse in the end.
    """
    head, sep, tail = raw.partition("\n")
    if PLACEHOLDER in raw:
        visible, pending = raw, {}
    elif head.lstrip().startswith("/"):
        # The typed command line shows as typed; only lines 2..n hide.
        visible, pending = head + PLACEHOLDER, {PLACEHOLDER: sep + tail}
    else:
        visible, pending = PLACEHOLDER, {PLACEHOLDER: raw}
    if pending:
        redraw_echo(prompt, raw)
    if paste_dir is not None:
        _save_paste(raw, paste_dir)
    return visible, pending


def expand(text: str, pending: dict[str, str]) -> str:
    """Swap each placeholder back for its saved paste, just before sending."""
    for token, saved in pending.items():
        text = text.replace(token, saved, 1)
    return text


def is_terminal() -> bool:
    """True when the prompt is interactive: stdin and stdout are both a tty.

    Queue draining and escape-stripping only make sense on a terminal; piped
    or scripted input is passed through untouched.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (OSError, ValueError):
        return False


def stdin_fd() -> int | None:
    """The file descriptor behind sys.stdin, or None if there is not one."""
    try:
        return sys.stdin.fileno()
    except (OSError, ValueError, AttributeError):
        return None


def readable(fd: int, seconds: float) -> bool:
    """Does fd have input queued, waiting at most `seconds` for it?"""
    try:
        import select

        ready, _, _ = select.select([fd], [], [], seconds)
    except (ImportError, OSError, ValueError):  # select cannot watch a console on Windows
        return False
    return bool(ready)


def clean(text: str) -> str:
    """Strip CSI escape sequences: bracketed-paste markers and the like."""
    return _CSI.sub("", text)


def redraw_echo(prompt: str, pasted: str) -> None:
    """Replace the echoed paste on screen with the prompt plus [multiline paste].

    By the time a paste is captured it is already on screen: the terminal
    echoed the burst as it arrived (readline renders it in bracketed-paste
    mode). The redraw erases those lines with escape codes and reprints the
    prompt line with the placeholder instead, so the transcript reads as if a
    single short marker had been typed.

    The height estimate counts wrapping against the terminal width from the
    top; it can be a line off around tabs or double-width characters, which
    only leaves a cosmetic leftover. Like style's colors, the redraw is
    skipped whenever escape codes would be unsafe to write (NO_COLOR, dumb
    terminal, or redirected output).
    """
    if not style.ENABLED:
        return
    try:
        width = shutil.get_terminal_size().columns
    except OSError:
        width = 0
    if width <= 0:
        width = 80
    shown_prompt = prompt.translate(_READLINE_MARKS)
    shown = _visible_width(shown_prompt)
    height = 0
    for i, piece in enumerate(pasted.split("\n")):
        # Only the first pasted line sits on the prompt line; the burst echoed
        # the rest without further prompts.
        start = shown if i == 0 else 0
        height += max(1, (start + len(piece) + width - 1) // width)
    up = f"\033[{height - 1}A" if height > 1 else ""
    sys.stdout.write(f"\r{up}\033[J{shown_prompt}{PLACEHOLDER}\n")
    sys.stdout.flush()


def _visible_width(text: str) -> int:
    """How many screen columns a piece of text would occupy, escape codes aside."""
    return len(_plain(text))


def _plain(text: str) -> str:
    """Text with escape sequences and readline's no-space markers removed."""
    return _CSI.sub("", text.translate(_READLINE_MARKS))


def _save_paste(text: str, paste_dir: Path) -> None:
    """Save a paste's exact text and tell the user where it went."""
    path = _save(text, paste_dir)
    if path is None:
        print(style.warning("  [could not save the multiline paste; it is still sent as typed]"))
    else:
        print(style.dim(f"  [multiline paste saved to {path}]"))


def _save(text: str, paste_dir: Path) -> Path | None:
    """Write the paste to paste_dir/paste-NN.txt with the next free index.

    The "x" open mode keeps two harnesses running at once from clobbering the
    same file, and newline="" writes the bytes with nothing translated, so the
    file is exactly what was captured (read_prompt has normalized the pasted
    line endings to \n by then). Failure is never fatal — the caller
    reports it and the paste has still been captured.
    """
    try:
        paste_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    for n in range(1, 10_000):
        path = paste_dir / f"paste-{n}.txt"
        try:
            with open(path, "x", encoding="utf-8", newline="") as f:
                f.write(text)
            return path
        except FileExistsError:
            continue
        except OSError:
            return None
    return None


def drain(fd: int) -> str:
    """Whatever the terminal still holds from the paste burst, as one string.

    Reading happens with the line discipline switched to cbreak (ECHO and
    ICANON off, VMIN 0 and VTIME 1): canonical mode would block on a final
    pasted line that has no Enter yet — a paste often ends without one — and
    would echo the leftovers a second time. os.read() then always returns
    promptly, with whatever is queued or nothing. The read ends once the
    queue has stayed empty for QUIET_SECONDS; CAP_SECONDS bounds the whole
    thing, so a burst is never split and a fast typist is only ever caught
    while keys are actually arriving. The original terminal settings are
    restored afterwards, and any failure returns what was collected (often "")
    with the terminal left as it was.
    """
    if termios is None or os.name == "nt":
        return drain_windows()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return ""
    cbreak = saved.copy()
    cbreak[3] &= ~(termios.ECHO | termios.ICANON)
    cbreak[6] = list(saved[6])  # the c_cc array is shared with saved; copy it
    try:
        cbreak[6][termios.VMIN] = 0
        cbreak[6][termios.VTIME] = 1  # hundredths of a second
        termios.tcsetattr(fd, termios.TCSANOW, cbreak)
    except (IndexError, termios.error):
        _restore(fd, saved)
        return ""
    data = bytearray()
    deadline = time.monotonic() + CAP_SECONDS
    try:
        while time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if chunk:
                data.extend(chunk)
                continue
            if not readable(fd, QUIET_SECONDS):
                break  # the burst is over: nothing arrived during the quiet window
    finally:
        _restore(fd, saved)
    return data.decode(sys.stdin.encoding or "utf-8", "replace")


def drain_windows() -> str:
    """Leftover console input on Windows, where select() cannot watch a tty.

    The console input buffer already holds the whole paste, so kbhit/getwch
    is enough: take characters while any remain, without waiting. getwch does
    not echo, special keys arrive as a two-character code (a NUL or 0xE0
    prefix followed by a key code), and CRLF line endings become \n, which is
    what the POSIX path gets from the terminal's ICRNL processing.
    """
    try:
        import msvcrt
    except ImportError:
        return ""
    chars: list[str] = []
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()  # discard the key code that follows the prefix
            continue
        chars.append(ch)
    return "".join(chars).replace("\r\n", "\n").replace("\r", "\n")


def _restore(fd: int, attrs: list) -> None:
    """Put back the terminal settings saved before cbreak reading (best effort)."""
    try:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except termios.error:
        pass