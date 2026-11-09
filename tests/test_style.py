"""Tests for ui.style.sanitize: control characters never reach the console.

Model-controlled text reaches the terminal in the approval diff, in tool-call
and error lines, and in the streamed reply. These tests pin that ESC, other
C0 controls, C1 controls, and bidi overrides are neutralized at the choke
points — while normal text (and the harness's own style codes) are untouched.
"""

import io
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
    def test_raw_writes_cannot_wipe_the_screen(self) -> None:
        printer = StreamPrinter("model", show_thinking=True)
        with mock.patch("sys.stdout", new=io.StringIO()) as fake:
            printer.write("before\x1b[2Jafter\n")
            printer.end_block()
        self.assertNotIn("\x1b", fake.getvalue())
        self.assertIn("before␛[2Jafter", fake.getvalue())

    def test_streamed_reply_path_is_sanitized(self) -> None:
        printer = StreamPrinter("model", show_thinking=True)
        with mock.patch("sys.stdout", new=io.StringIO()) as fake:
            printer.text("\x1b]52;c;base64payload\x07 hi \x1b[2Jthere\n")
            printer.end_block()  # closes the open block and flushes what is held
        out = fake.getvalue()
        self.assertNotIn("\x1b", out)
        self.assertIn("there", out)
        self.assertIn("␛", out)


if __name__ == "__main__":
    unittest.main()
