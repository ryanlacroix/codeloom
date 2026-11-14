"""Tests for loom's interactive write approver: Enter is the default yes.

The approver is the one human control over the model's writes, so the answer
grammar is pinned tightly: Enter (the default), y and yes approve; n, no and
anything unrecognised still decline, so a typo can never approve; and a dead
stdin (EOF) declines rather than start approving a model's writes.
"""

import io
import unittest
from unittest import mock
from contextlib import redirect_stdout

import tests.util  # noqa: F401  turns colors off, so the prompt label is plain text
from loom import make_approver


def answer_with(typed: str) -> bool:
    """One prompt answered with `typed` (the text the user would type)."""
    approver = make_approver(auto_approve=False)
    with mock.patch("builtins.input", return_value=typed):
        return approver("a message", ["| a diff line"])


class ApprovePromptTest(unittest.TestCase):
    def test_enter_and_yes_approve(self) -> None:
        for typed in ("", "y", "Y", " yes ", "yEs"):
            with self.subTest(typed=typed):
                self.assertTrue(answer_with(typed))

    def test_no_and_unrecognised_text_decline(self) -> None:
        for typed in ("n", "N", "no", "No", "nope"):
            with self.subTest(typed=typed):
                self.assertFalse(answer_with(typed))

    def test_eof_declines(self) -> None:
        approver = make_approver(auto_approve=False)
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(approver("a message", ["| a diff line"]))

    def test_the_prompt_says_the_default_is_yes(self) -> None:
        approver = make_approver(auto_approve=False)
        with mock.patch("builtins.input", return_value="") as asked:
            approver("a message", [])
        self.assertIn("[Y/n]", asked.call_args[0][0])

    def test_auto_approve_never_asks(self) -> None:
        approver = make_approver(auto_approve=True)
        with redirect_stdout(io.StringIO()) as out, mock.patch("builtins.input") as never:
            self.assertTrue(approver("a message", ["| a diff line"]))
        self.assertFalse(never.called)
        self.assertIn("[auto-approved]", out.getvalue())


if __name__ == "__main__":
    unittest.main()
