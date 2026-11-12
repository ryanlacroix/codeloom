"""Tests for core.agent: the turn loop, the mode notes, tool dispatch, and redaction."""

import unittest
from unittest import mock

from api.http_client import APIError
from api.models import Usage
from core.agent import MAX_TOKENS, Agent, cut_off_reason
from core.prompts import COMPACT_NOTE, COMPACT_SUMMARY_NOTE, SYSTEM_PROMPT
from core.redact import Redactor
from core.tools import TOOLS

from tests.util import WorkspaceTest


class FakeClient:
    """Stands in for the API client, returning each canned response in turn."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.bodies = []  # every request body create_message was given

    def create_message(self, body: dict, printer) -> dict:
        self.bodies.append(body)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class CutOffReasonTest(unittest.TestCase):
    def test_reasons_for_each_kind_of_cut_off(self) -> None:
        tool_call = [{"type": "tool_use", "id": "x", "name": "n", "input": {}}]
        self.assertIn("middle of a tool call", cut_off_reason(tool_call))
        self.assertIn("still thinking", cut_off_reason([{"type": "thinking", "thinking": "..."}]))
        self.assertIn("was cut off", cut_off_reason([{"type": "text", "text": "hi"}]))
        self.assertIn("was cut off", cut_off_reason([]))


class AgentTurnTest(WorkspaceTest):
    def make_agent(self, responses: list, **kwargs) -> Agent:
        return Agent(FakeClient(responses), self.ws, model="test-model", **kwargs)

    def test_one_tool_round_trip(self) -> None:
        agent = self.make_agent(
            [
                {
                    "content": [{"type": "tool_use", "id": "t1", "name": "list_directory", "input": {"path": "."}}],
                    "stop_reason": "tool_use",
                },
                {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
            ]
        )
        agent.send("list the root")
        self.assertEqual([m["role"] for m in agent.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(agent.messages[1]["content"][0]["name"], "list_directory")
        tool_result = agent.messages[2]["content"][0]
        self.assertEqual(tool_result["tool_use_id"], "t1")
        self.assertEqual(tool_result["content"], "(empty directory)")
        self.assertEqual(agent.messages[3]["content"][0]["text"], "done")

    def test_read_only_messages_refuse_write_tools(self) -> None:
        agent = self.make_agent(
            [
                {
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "write_file", "input": {"path": "x.txt", "content": "hi"}},
                        {"type": "tool_use", "id": "t2", "name": "list_directory", "input": {"path": "."}},
                    ],
                    "stop_reason": "tool_use",
                },
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
            ]
        )
        agent.send("make a file", read_only=True)
        self.assertTrue(agent.messages[0]["content"].startswith("[Read-only message:"))
        results = agent.messages[2]["content"]
        self.assertTrue(results[0]["is_error"])
        self.assertIn("writes are disabled", results[0]["content"])
        self.assertNotIn("is_error", results[1])  # the read-only tools still ran
        self.assertEqual(results[1]["content"], "(empty directory)")
        self.assertFalse((self.root / "x.txt").exists())

    def test_plan_notes_are_prepended_by_mode(self) -> None:
        for plan, prefix in (
            ("start", "[Plan request:"),
            ("update", "[Plan update:"),
            ("approved", "[Plan approved:"),
        ):
            with self.subTest(plan=plan):
                agent = self.make_agent([{"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}])
                agent.send("task", plan=plan)
                self.assertTrue(agent.messages[0]["content"].startswith(prefix))

    def test_tool_errors_become_failed_results(self) -> None:
        agent = self.make_agent(
            [
                {
                    "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "missing.txt"}}],
                    "stop_reason": "tool_use",
                },
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
            ]
        )
        agent.send("read it")
        result = agent.messages[2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("not a file: missing.txt", result["content"])

    def test_redaction_reaches_user_text_and_tool_results(self) -> None:
        (self.root / "note.txt").write_text("SEKRET value\n")
        agent = Agent(
            FakeClient(
                [
                    {
                        "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "note.txt"}}],
                        "stop_reason": "tool_use",
                    },
                    {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"},
                ]
            ),
            self.ws,
            model="m",
            show_thinking=False,
            redactor=Redactor(["SEKRET"]),
        )
        agent.send("what is in my SEKRET note")
        self.assertEqual(agent.messages[0]["content"], "what is in my <redacted> note")
        result = agent.messages[2]["content"][0]["content"]
        self.assertNotIn("SEKRET", result)
        self.assertIn("<redacted>", result)

    def test_request_body_shape(self) -> None:
        agent = self.make_agent([])
        agent.set_model("other-model")
        body = agent.request_body()
        self.assertEqual(set(body), {"model", "max_tokens", "system", "tools", "messages"})
        self.assertEqual(body["model"], "other-model")
        self.assertEqual(body["max_tokens"], MAX_TOKENS)
        self.assertEqual(body["tools"], TOOLS)
        self.assertTrue(body["system"].startswith("You are a coding assistant"))
        self.assertIs(body["messages"], agent.messages)


class MalformedToolCallTest(WorkspaceTest):
    """A tool call the API did not type-check must not end the session.

    Ollama does not enforce input schemas, so a null, number or missing field
    can arrive where the schema asked for a string; every failure, expected
    or not, becomes a failed tool result and the turn goes on.
    """

    def make_agent(self, responses: list, **kwargs) -> Agent:
        return Agent(FakeClient(responses), self.ws, model="m", **kwargs)

    def test_non_string_paths_and_patterns_become_failed_results(self) -> None:
        agent = self.make_agent(
            [
                {
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "write_file", "input": {"path": 123, "content": "x"}},
                        {"type": "tool_use", "id": "t2", "name": "search", "input": {"pattern": None, "path": "."}},
                        {"type": "tool_use", "input": {"path": "."}},  # no name, no id
                    ],
                    "stop_reason": "tool_use",
                },
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
            ]
        )
        agent.send("go")
        results = agent.messages[2]["content"]
        self.assertTrue(all(result["is_error"] for result in results))
        self.assertIn("path must be a non-empty string", results[0]["content"])
        self.assertIn("pattern must be a non-empty string", results[1]["content"])
        self.assertIn("unknown tool", results[2]["content"])
        self.assertEqual(results[2]["tool_use_id"], "")
        self.assertEqual(agent.messages[3]["content"][0]["text"], "ok")

    def test_an_unexpected_error_becomes_a_failed_result(self) -> None:
        agent = self.make_agent(
            [
                {
                    "content": [{"type": "tool_use", "id": "t1", "name": "list_directory", "input": {"path": "."}}],
                    "stop_reason": "tool_use",
                },
                {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"},
            ]
        )
        with mock.patch.object(self.ws, "list_directory", side_effect=RuntimeError("boom")):
            agent.send("list")
        result = agent.messages[2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("Error: boom.", result["content"])
        self.assertEqual(agent.messages[3]["content"][0]["text"], "ok")


class AgentCompactTest(WorkspaceTest):
    """The compaction: one tool-free call summarizes the history, and only a
    complete summary replaces it — anything less (an empty or cut-off summary,
    a refusal, a stray tool call, or a failed call) leaves the history exactly
    as it was."""

    HISTORY = [
        {"role": "user", "content": "list the root"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "list_directory", "input": {"path": "."}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "(empty directory)"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]

    def make_agent(self, responses: list, **kwargs) -> Agent:
        return Agent(FakeClient(responses), self.ws, model="test-model", **kwargs)

    def test_summary_replaces_the_history(self) -> None:
        agent = self.make_agent(
            [
                {
                    "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "the summary"}],
                    "stop_reason": "end_turn",
                    "usage": Usage(500, 60),
                }
            ]
        )
        agent.messages = [dict(m) for m in self.HISTORY]
        usage = agent.compact()
        self.assertEqual((usage.input, usage.output), (500, 60))
        self.assertEqual(agent.messages, [{"role": "user", "content": COMPACT_SUMMARY_NOTE + "the summary"}])

    def test_the_compact_call_is_tool_free_and_carries_the_focus(self) -> None:
        agent = self.make_agent([{"content": [{"type": "text", "text": "s"}], "stop_reason": "end_turn"}])
        agent.messages = [dict(m) for m in self.HISTORY]
        agent.compact("the failing test")
        body = agent.client.bodies[-1]
        self.assertEqual(set(body), {"model", "max_tokens", "system", "messages"})
        self.assertEqual(body["max_tokens"], MAX_TOKENS)
        self.assertEqual(body["system"], SYSTEM_PROMPT)
        self.assertEqual(body["messages"][:-1], self.HISTORY)
        self.assertEqual(body["messages"][-1]["role"], "user")
        # The note (plus the focus line) is only ever sent, never parked in the
        # live history.
        self.assertEqual(
            body["messages"][-1]["content"], COMPACT_NOTE + "Focus especially on: the failing test\n"
        )

    def test_an_unusable_summary_leaves_the_history_unchanged(self) -> None:
        for response, _why in (
            # only thinking, no text
            ({"content": [{"type": "thinking", "thinking": "hm"}], "stop_reason": "end_turn"}, "no summary text"),
            ({"content": [{"type": "text", "text": "no"}], "stop_reason": "refusal"}, "declined to summarize"),
            ({"content": [{"type": "text", "text": "half a summary"}], "stop_reason": "max_tokens"}, "cut off"),
            (
                {
                    "content": [
                        {"type": "text", "text": "let me look"},
                        {"type": "tool_use", "id": "x", "name": "list_directory", "input": {}},
                    ],
                    "stop_reason": "tool_use",
                },
                "run a tool",
            ),
        ):
            with self.subTest(stop=response["stop_reason"]):
                agent = self.make_agent([response])
                history = [dict(m) for m in self.HISTORY]
                agent.messages = history
                self.assertIsNone(agent.compact())
                self.assertEqual(agent.messages, history)

    def test_a_failed_call_leaves_the_history_unchanged(self) -> None:
        agent = self.make_agent([APIError(500, "boom")])
        agent.messages = [dict(m) for m in self.HISTORY]
        with self.assertRaises(APIError):
            agent.compact()
        self.assertEqual(agent.messages, self.HISTORY)

    def test_an_empty_conversation_refuses_to_compact(self) -> None:
        agent = self.make_agent([])
        with self.assertRaises(ValueError):
            agent.compact()
        self.assertEqual(agent.messages, [])

    def test_the_stored_summary_is_redacted(self) -> None:
        agent = self.make_agent(
            [{"content": [{"type": "text", "text": "the note held SEKRET"}], "stop_reason": "end_turn"}],
            redactor=Redactor(["SEKRET"]),
        )
        agent.messages = [dict(m) for m in self.HISTORY]
        agent.compact()
        self.assertEqual(agent.messages[0]["content"], COMPACT_SUMMARY_NOTE + "the note held <redacted>")


if __name__ == "__main__":
    unittest.main()
