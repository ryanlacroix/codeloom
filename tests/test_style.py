"""Tests for ui.style: the sanitize choke points, and the color gate.

Model-controlled text reaches the terminal in the approval diff, in tool-call
and error lines, and in the streamed reply. The sanitize tests pin that ESC,
other C0 controls, C1 controls, and bidi overrides are neutralized at the choke
points — while normal text (and the harness's own style codes) are untouched.
The color-gate tests pin colors_enabled(): the NO_COLOR/FORCE_COLOR switches,
the terminal check, and the Windows console's ANSI-mode switch, run against a
fake windll so every Windows leg executes on any development platform.
"""

import ctypes
import io
import os
import unittest
from unittest import mock

import tests.util  # noqa: F401  turns colors off, so painted assertions see plain text
from ui import style
from ui.stream import StreamPrinter


class SanitizeTest(unittest.TestCase):
    def test_plain_text_passes_through(self) -> None:
        text = "plain words, code(), 中文, and 🙂"
        self.assertEqual(style.sanitize(text), text)

    def test_tabs_and_newlines_are_kept(self) -> None:
        self.assertEqual(style.sanitize("one\ttwo\nthree\n"), "one\ttwo\nthree\n")

    def test_esc_becomes_the_visible_symbol(self) -> None:
        self.assertEqual(style.sanitize("\x1b[2J"), "␛[2J")

    def test_other_c0_controls_are_dropped(self) -> None:
        for bad in "\x00\x01\x07\x08\x0b\x0c\x1a\x1c\x1f\x7f":
            self.assertEqual(style.sanitize(f"a{bad}b"), "ab", repr(bad))

    def test_c1_controls_are_dropped(self) -> None:
        # some terminals honor the raw C1 range (e.g. U+009B as CSI) without ESC
        for bad in "\x80\x90\x9b\x9f":
            self.assertEqual(style.sanitize(f"a{bad}b"), "ab", repr(bad))

    def test_bidi_overrides_are_dropped(self) -> None:
        for bad in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069":
            self.assertEqual(style.sanitize(f"+test{bad}con"), "+testcon", repr(bad))

    def test_paint_never_leaves_a_raw_escape(self) -> None:
        # an OSC 52 clipboard-overwrite attempt, as a file line in a diff could carry it
        out = style.paint("\x1b]52;c;base64payload\x07", style.BOLD)
        self.assertNotIn("\x1b", out)
        self.assertIn("␛", out)
        self.assertEqual(style.paint("clean", style.BOLD), "clean")

    def test_prompt_never_leaves_a_raw_escape(self) -> None:
        self.assertNotIn("\x1b", style.prompt("\x1b]0;spoof\x07", style.BOLD))


class StreamPrinterTest(unittest.TestCase):
    def test_our_own_style_codes_survive_raw_writes(self) -> None:
        # write() is trusted-internal: it carries the harness's own escape
        # codes (labels, rules) and must not have them stripped — that
        # stripping is what printed them as literal ␛ glyphs. Model text is
        # sanitized one level down, where it enters the renderer (the
        # streaming tests below).
        printer = StreamPrinter("model", show_thinking=True)
        # the literal output of style.dim("thinking>") with colors on
        styled = "\x1b[2mthinking>\x1b[0m"
        with mock.patch("sys.stdout", new=io.StringIO()) as fake:
            printer.write(f"{styled} ")
        self.assertEqual(fake.getvalue(), f"{styled} ")

    def test_thinking_path_is_sanitized(self) -> None:
        printer = StreamPrinter("model", show_thinking=True)
        with mock.patch("sys.stdout", new=io.StringIO()) as fake:
            printer.thinking("\x1b]52;c;base64payload\x07 pondering\n")
            printer.end_block()
        self.assertNotIn("\x1b", fake.getvalue())
        self.assertIn("pondering", fake.getvalue())
        self.assertIn("␛", fake.getvalue())

    def test_streamed_reply_path_is_sanitized(self) -> None:
        printer = StreamPrinter("model", show_thinking=True)
        with mock.patch("sys.stdout", new=io.StringIO()) as fake:
            printer.text("\x1b]52;c;base64payload\x07 hi \x1b[2Jthere\n")
            printer.end_block()  # closes the open block and flushes what is held
        out = fake.getvalue()
        self.assertNotIn("\x1b", out)
        self.assertIn("there", out)
        self.assertIn("␛", out)


class ColorGateTest(unittest.TestCase):
    """colors_enabled(): the env switches, the terminal check, the Windows ANSI switch.

    The Windows decision is a mode bit on the console itself
    (ENABLE_VIRTUAL_TERMINAL_PROCESSING), probed directly through kernel32 — so the
    Windows scenarios here run with a fake windll patched over ctypes.windll (which
    exists only on Windows itself), pinning the outcomes on any development
    platform: a console that accepts the switch gets colors, whether it is the
    classic console host behind PowerShell/cmd, Windows Terminal or the VS Code
    terminal; anything that refuses it, has no console behind stdout, or has no
    kernel32 to ask stays plain text; ConEmu and ANSICON translate the codes
    themselves and get colors with nothing to flip.
    """

    def gate(self, env: dict | None = None, isatty: bool = False, windll=None, platform: str = "win32") -> bool:
        """Run colors_enabled() under a controlled environment, stdout, and windll."""
        # The blanks neutralize ambient values that would otherwise decide a branch;
        # the caller's env (merged after them) overrides any of these keys.
        switches = {"NO_COLOR": "", "FORCE_COLOR": "", "ANSICON": "", "ConEmuANSI": "", "TERM": "", **(env or {})}
        stdout = mock.MagicMock()
        stdout.isatty.return_value = isatty
        with mock.patch.dict(os.environ, switches), \
                mock.patch("sys.stdout", new=stdout), \
                mock.patch("sys.platform", platform), \
                mock.patch.object(ctypes, "windll", windll, create=True):
            return style.colors_enabled()

    @staticmethod
    def windll(getmode: int = 1, setmode: int = 1):
        """A fake windll whose console reports itself (getmode) and accepts (setmode=1)
        or refuses (setmode=0) the ANSI switch. .kernel32 stays inspectable."""
        api = mock.MagicMock()
        api.GetConsoleMode.return_value = getmode
        api.SetConsoleMode.return_value = setmode
        return mock.Mock(kernel32=api)

    def test_no_color_wins_over_everything(self) -> None:
        self.assertFalse(self.gate(env={"NO_COLOR": "1", "FORCE_COLOR": "1"}, isatty=True, windll=self.windll()))

    def test_no_color_without_a_terminal(self) -> None:
        self.assertFalse(self.gate(env={"NO_COLOR": "1"}))

    def test_force_color_overrides_a_non_terminal(self) -> None:
        # FORCE_COLOR is an explicit ask; the user chooses what stdout points at.
        self.assertTrue(self.gate(env={"FORCE_COLOR": "1"}))

    def test_no_colors_without_a_terminal(self) -> None:
        self.assertFalse(self.gate())

    def test_posix_terminal_skips_the_probe(self) -> None:
        # Off Windows there is no console switch; a tty understands the codes as-is.
        self.assertTrue(self.gate(isatty=True, platform="linux"))

    def test_windows_colors_once_the_switch_is_accepted(self) -> None:
        # A console that takes the bit — classic conhost, Windows Terminal, VS Code.
        self.assertTrue(self.gate(isatty=True, windll=self.windll()))

    def test_windows_plain_text_when_the_switch_is_refused(self) -> None:
        # A console without the bit (Windows pre-10) prints the codes literally: off.
        self.assertFalse(self.gate(isatty=True, windll=self.windll(setmode=0)))

    def test_windows_plain_text_without_a_console(self) -> None:
        # GetConsoleMode refusing the stdout handle means no console behind it.
        self.assertFalse(self.gate(isatty=True, windll=self.windll(getmode=0)))

    def test_windows_plain_text_without_kernel32(self) -> None:
        # windll=None plays the stripped/non-Windows build the probe must survive.
        self.assertFalse(self.gate(isatty=True, windll=None))

    def test_translating_shells_color_even_when_the_switch_is_refused(self) -> None:
        # ConEmu and ANSICON rewrite the codes themselves: no switch to flip.
        for env in ({"ConEmuANSI": "ON"}, {"ANSICON": "1"}):
            with self.subTest(env=env):
                self.assertTrue(self.gate(env=env, isatty=True, windll=self.windll(setmode=0)))

    def test_the_requested_bit_is_the_virtual_terminal_one(self) -> None:
        shell = self.windll()
        self.gate(isatty=True, windll=shell)
        api = shell.kernel32
        api.GetStdHandle.assert_called_once_with(-11)  # STD_OUTPUT_HANDLE
        # The fake console starts at mode 0, so the request must carry exactly the bit.
        self.assertEqual(api.SetConsoleMode.call_args[0][1], style.ENABLE_VIRTUAL_TERMINAL_PROCESSING)


if __name__ == "__main__":
    unittest.main()
