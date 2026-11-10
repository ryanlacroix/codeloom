"""Tests for the startup splash: printed exactly when console colors are on.

The splash shares fate with every painted string — it is gated on style.ENABLED,
the one flag they all read — so a colors-off console (NO_COLOR, no terminal, a
Windows console that cannot be switched to ANSI) prints nothing at all. When the
Windows gate went stale, PowerShell sessions lost the splash along with colors;
these pins keep the tie between the two explicit.
"""

import io
import unittest
from unittest import mock

import ui.style
from ui import splash


class SplashTest(unittest.TestCase):
    def test_prints_the_art_when_colors_are_on(self) -> None:
        with mock.patch.object(ui.style, "ENABLED", True), mock.patch("sys.stdout", new=io.StringIO()) as out:
            splash.print_splash()
        printed = out.getvalue()
        for line in splash.ART.splitlines():
            # Centering pads each line, so the art's own (stripped) glyphs are what
            # is contiguous; the pad spaces before them are not part of ART.
            self.assertIn(line.strip(), printed)
        self.assertIn("\x1b[1;36m", printed)  # bold cyan, the art's own style

    def test_prints_nothing_when_colors_are_off(self) -> None:
        with mock.patch.object(ui.style, "ENABLED", False), mock.patch("sys.stdout", new=io.StringIO()) as out:
            splash.print_splash()
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
