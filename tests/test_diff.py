"""Tests for ui.diff: the confirmation diff shown before every write."""

import unittest

import tests.util  # noqa: F401  turns colors off, so painted assertions see plain text
from ui.diff import render_diff


class RenderDiffTest(unittest.TestCase):
    def test_identical_texts_diff_to_nothing(self) -> None:
        self.assertEqual(render_diff("a\nb\n", "a\nb\n"), ([], 0, 0))

    def test_counts_and_markers(self) -> None:
        lines, additions, removals = render_diff("one\ntwo\n", "one\nTWO\n")
        self.assertEqual((additions, removals), (1, 1))
        self.assertTrue(any(line.startswith("-two") for line in lines))
        self.assertTrue(any(line.startswith("+TWO") for line in lines))

    def test_hunk_header_is_kept(self) -> None:
        text = "".join(f"line{i}\n" for i in range(10))
        lines, _, _ = render_diff(text, text.replace("line5", "LINE5"))
        self.assertTrue(any(line.startswith("@@") for line in lines))

    def test_truncation_note_while_counts_stay_exact(self) -> None:
        old = "".join(f"line{i}\n" for i in range(200))
        lines, additions, removals = render_diff(old, "")
        self.assertEqual((additions, removals), (0, 200))
        self.assertEqual(len(lines), 101)  # 100 shown lines + the truncation note
        self.assertTrue(lines[-1].startswith("... (101 more diff lines)"))

    def test_very_long_lines_are_trimmed(self) -> None:
        lines, additions, removals = render_diff("", "x" * 500 + "\n")
        self.assertEqual((additions, removals), (1, 0))
        self.assertIn("(101 more chars)", "".join(lines))


if __name__ == "__main__":
    unittest.main()
