"""Tests for core.session: the prompt-loop state machine (plan mode, paste, reset)."""

import tempfile
import unittest
from pathlib import Path

import core.session as session_module
from core.session import PASTE_FILE, PLAN_FILE, Session, Submit


class FakeAgent:
    """Just enough of Agent for Session to steer."""

    def __init__(self):
        self.messages = [{"role": "user", "content": "old"}]
        self.client = type("Client", (), {"think": "on"})()
        self.show_thinking = True

    def last_reply_text(self) -> str:
        return "the plan text"


class SessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Point the harness's scratch dir at the test dir, restoring it after.
        self.saved_temp = session_module.TEMP_DIR
        session_module.TEMP_DIR = Path(self._tmp.name)
        self.addCleanup(setattr, session_module, "TEMP_DIR", self.saved_temp)
        self.agent = FakeAgent()
        self.session = Session(self.agent, think="low")

    def test_prompt_labels_reflect_the_state(self) -> None:
        self.assertIn("you>", self.session.prompt())
        self.session.paste_pending = True
        self.assertIn("you (with pasted text)>", self.session.prompt())
        self.session.plan_mode = True
        self.assertIn("plan>", self.session.prompt())
        self.agent.client.think = "off"
        self.assertIn("(thinking off)", self.session.prompt())

    def test_plan_transitions_and_the_saved_plan_file(self) -> None:
        self.session.advance(None)  # an ordinary message changes nothing
        self.assertFalse(self.session.plan_mode)
        self.session.advance("start")
        self.assertTrue(self.session.plan_mode)
        self.assertEqual((Path(self._tmp.name) / PLAN_FILE).read_text(encoding="utf-8"), "the plan text\n")
        self.session.advance("update")
        self.assertTrue(self.session.plan_mode)
        self.session.advance("approved")
        self.assertFalse(self.session.plan_mode)

    def test_as_plan_update_marks_the_submit(self) -> None:
        result = self.session.as_plan_update(Submit("feedback"))
        self.assertEqual(result.plan, "update")
        self.assertEqual(result.text, "feedback")

    def test_reset_clears_everything(self) -> None:
        self.session.plan_mode = True
        self.session.paste_pending = True
        self.session.reset()
        self.assertFalse(self.session.plan_mode)
        self.assertFalse(self.session.paste_pending)
        self.assertEqual(self.agent.messages, [])

    def test_paste_lifecycle(self) -> None:
        path = self.session.open_paste()
        self.assertTrue(self.session.paste_pending)
        self.assertEqual(path.name, PASTE_FILE)
        self.assertEqual(path.read_bytes(), b"")  # a fresh buffer
        path.write_text("pasted text", encoding="utf-8")
        submit = self.session.with_paste(Submit("my message"))
        self.assertFalse(self.session.paste_pending)
        self.assertIn("pasted text", submit.text)
        self.assertIn("my message", submit.text)

    def test_reopening_a_pending_paste_keeps_the_draft(self) -> None:
        self.session.open_paste()
        (Path(self._tmp.name) / PASTE_FILE).write_text("draft", encoding="utf-8")
        self.session.open_paste()  # already pending: only reopened
        self.assertEqual((Path(self._tmp.name) / PASTE_FILE).read_text(encoding="utf-8"), "draft")

    def test_an_empty_paste_is_consumed_without_quoting(self) -> None:
        self.session.open_paste()
        submit = self.session.with_paste(Submit("my message"))
        self.assertFalse(self.session.paste_pending)
        self.assertEqual(submit.text, "my message")


if __name__ == "__main__":
    unittest.main()
