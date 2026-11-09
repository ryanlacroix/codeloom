"""Tests for api.anthropic_client: the endpoint config loader, the request
translation, and the reassembly of a streamed /v1/messages response."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from api import anthropic_client
from api.anthropic_client import (
    AnthropicClient,
    StreamAssembler,
    endpoint_url,
    load_endpoint_config,
    to_anthropic_messages,
    to_anthropic_tools,
    tool_input,
)
from api.http_client import APIConnectionError, APIError
from api.models import Usage


class FakePrinter:
    """Records what a streamed response showed, in arrival order.

    The recordings are named shown_* so they cannot shadow the printer
    methods they stand in for.
    """

    def __init__(self):
        self.shown_thinking = []
        self.shown_text = []
        self.closed = 0

    def thinking(self, piece: str) -> None:
        self.shown_thinking.append(piece)

    def text(self, piece: str) -> None:
        self.shown_text.append(piece)

    def end_block(self) -> None:
        self.closed += 1


class FakeResponse:
    """The little of an HTTP response create_message needs: iterable lines plus
    the context manager open_stream's result is used as."""

    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sse_lines(chunks) -> list:
    """The wire form of one streamed response: event headers, data lines,
    blank separators and a keep-alive comment, as bytes like HTTPResponse yields."""
    lines = []
    for chunk in chunks:
        lines.append(f"event: {chunk['type']}\n")
        lines.append(f"data: {json.dumps(chunk)}\n")
    lines.append("\n")
    lines.append(": keep-alive\n")
    return [line.encode("utf-8") for line in lines]


# One full turn: a thinking block with a signature, a text reply, and a tool
# call whose input arrives as two JSON fragments.
TOOL_TURN = [
    {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 10, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "looking"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"pat'}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": 'tern": "it"}'}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
    {"type": "message_stop"},
]


class EndpointUrlTest(unittest.TestCase):
    def test_joins_v1_messages_and_tolerates_a_full_url(self) -> None:
        self.assertEqual(endpoint_url("https://llm.internal.example.com"), "https://llm.internal.example.com/v1/messages")
        self.assertEqual(endpoint_url("https://x/"), "https://x/v1/messages")
        self.assertEqual(endpoint_url("https://x/api"), "https://x/api/v1/messages")
        self.assertEqual(endpoint_url("https://x/v1/messages"), "https://x/v1/messages")
        self.assertEqual(endpoint_url("https://x/v1/messages/"), "https://x/v1/messages")


class ToolInputTest(unittest.TestCase):
    def test_an_object_assembles_from_fragments(self) -> None:
        self.assertEqual(tool_input('{"pat' + 'tern": "x"}'), {"pattern": "x"})

    def test_empty_partial_or_non_object_input_counts_as_nothing(self) -> None:
        self.assertEqual(tool_input(""), {})
        self.assertEqual(tool_input("  "), {})
        self.assertEqual(tool_input('{"pattern": '), {})
        self.assertEqual(tool_input("[1, 2]"), {})


class ToAnthropicToolsTest(unittest.TestCase):
    def test_tools_drop_the_strict_key(self) -> None:
        tools = [
            {"name": "read_file", "description": "d", "input_schema": {"type": "object"}, "strict": True}
        ]
        self.assertEqual(
            to_anthropic_tools(tools),
            [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}}],
        )


class ToAnthropicMessagesTest(unittest.TestCase):
    def test_history_passes_through_unchanged(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hm", "signature": "sig"},
                    {"type": "tool_use", "id": "1", "name": "search", "input": {"pattern": "it"}},
                ],
            },
        ]
        self.assertIs(to_anthropic_messages(history), history)

    def test_thinking_blocks_are_stripped_when_thinking_is_off(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hm", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "d"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ]
        self.assertEqual(
            to_anthropic_messages(history, strip_thinking=True),
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ],
        )

    def test_an_assistant_turn_of_thinking_only_is_dropped(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}]},
            {"role": "user", "content": "go on"},
        ]
        self.assertEqual(
            to_anthropic_messages(history, strip_thinking=True),
            [{"role": "user", "content": "hi"}, {"role": "user", "content": "go on"}],
        )

    def test_tool_result_lists_are_kept(self) -> None:
        history = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "read_file", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "text", "is_error": True}],
            },
        ]
        self.assertEqual(to_anthropic_messages(history, strip_thinking=True), history)


class LoadEndpointConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "anthropic_config.json"
        self.addCleanup(self._tmp.cleanup)

    def write(self, payload) -> None:
        self.path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")

    def test_valid_config(self) -> None:
        self.write(
            {
                "base_url": "https://llm.internal.example.com/",
                "api_key": " secret ",
                "model": "m1",
                "models": ["m1", "m2"],
                "headers": {"anthropic-version": "2024-10-22"},
            }
        )
        config, error = load_endpoint_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(
            config,
            {
                "base_url": "https://llm.internal.example.com",
                "api_key": "secret",
                "model": "m1",
                "models": ["m1", "m2"],
                "headers": {"anthropic-version": "2024-10-22"},
            },
        )

    def test_first_models_entry_is_the_default_model(self) -> None:
        self.write({"base_url": "https://x", "api_key": "k", "models": ["a", "b"]})
        config, error = load_endpoint_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(config["model"], "a")

    def test_minimal_config_has_no_model(self) -> None:
        self.write({"base_url": "https://x", "api_key": "k"})
        config, error = load_endpoint_config(self.path)
        self.assertIsNone(error)
        self.assertIsNone(config["model"])
        self.assertNotIn("models", config)
        self.assertNotIn("headers", config)

    def test_utf8_bom_is_tolerated(self) -> None:
        self.path.write_bytes(json.dumps({"base_url": "https://x", "api_key": "k"}).encode("utf-8-sig"))
        config, error = load_endpoint_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(config["base_url"], "https://x")

    def test_missing_file(self) -> None:
        config, error = load_endpoint_config(self.path)
        self.assertEqual(config, {})
        self.assertIn("No endpoint config found", error)

    def test_invalid_json(self) -> None:
        self.write("{oops")
        config, error = load_endpoint_config(self.path)
        self.assertEqual(config, {})
        self.assertIn("not valid JSON", error)

    def test_wrong_shapes_are_reported(self) -> None:
        for payload, note in (
            ('["x"]', "must be a JSON object"),
            ('{"api_key": "k"}', 'needs a "base_url"'),
            ('{"base_url": "llm.internal.example.com", "api_key": "k"}', "needs its scheme"),
            ('{"base_url": "https://x"}', 'needs an "api_key"'),
            ('{"base_url": "https://x", "api_key": ""}', 'needs an "api_key"'),
            ('{"base_url": "https://x", "api_key": "k", "model": 3}', '"model" must be a non-empty string'),
            ('{"base_url": "https://x", "api_key": "k", "models": "m"}', '"models" must be a list'),
            ('{"base_url": "https://x", "api_key": "k", "models": [1]}', '"models" must be a list'),
            ('{"base_url": "https://x", "api_key": "k", "headers": {"a": 1}}', '"headers" must be an object'),
        ):
            with self.subTest(payload=payload):
                self.write(payload)
                config, error = load_endpoint_config(self.path)
                self.assertEqual(config, {})
                self.assertIn(note, error)


class StreamAssemblerTest(unittest.TestCase):
    def feed(self, chunks) -> tuple:
        printer = FakePrinter()
        assembler = StreamAssembler(printer)
        for chunk in chunks:
            assembler.feed(chunk)
        return assembler, printer

    def test_a_tool_turn_reassembles_blocks_usage_and_stop_reason(self) -> None:
        assembler, printer = self.feed(TOOL_TURN)
        self.assertTrue(assembler.done)
        self.assertEqual(
            assembler.message(),
            {
                "model": "m",
                "content": [
                    {"type": "thinking", "thinking": "hm", "signature": "sig"},
                    {"type": "text", "text": "looking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"pattern": "it"}},
                ],
                "stop_reason": "tool_use",
                "usage": Usage(input=10, output=5),
            },
        )
        self.assertEqual(printer.shown_thinking, ["hm"])
        self.assertEqual(printer.shown_text, ["looking"])

    def test_stop_reason_defaults_to_end_turn(self) -> None:
        chunks = [
            {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 3}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "message_stop"},
        ]
        assembler, _ = self.feed(chunks)
        self.assertEqual(assembler.message()["stop_reason"], "end_turn")
        self.assertEqual(assembler.message()["usage"], Usage(input=3, output=0))

    def test_text_in_a_block_start_is_shown_and_kept(self) -> None:
        chunks = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "pre"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "fix"}},
        ]
        assembler, printer = self.feed(chunks)
        self.assertEqual(printer.shown_text, ["pre", "fix"])
        self.assertEqual(assembler.message()["content"], [{"type": "text", "text": "prefix"}])

    def test_redacted_thinking_passes_through(self) -> None:
        chunks = [{"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "xyz"}}]
        assembler, _ = self.feed(chunks)
        self.assertEqual(assembler.message()["content"], [{"type": "redacted_thinking", "data": "xyz"}])

    def test_an_unrecognized_block_passes_through_unchanged(self) -> None:
        chunks = [{"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "s1"}}]
        assembler, _ = self.feed(chunks)
        self.assertEqual(assembler.message()["content"], [{"type": "server_tool_use", "id": "s1"}])

    def test_blocks_keep_their_index_order(self) -> None:
        chunks = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "a"}},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": "c"}},
        ]
        assembler, _ = self.feed(chunks)
        self.assertEqual(
            assembler.message()["content"],
            [{"type": "text", "text": "a"}, {"type": "text", "text": "c"}],
        )

    def test_an_error_event_raises(self) -> None:
        with self.assertRaises(APIError):
            StreamAssembler(FakePrinter()).feed(
                {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}
            )

    def test_a_cut_off_tool_input_assembles_to_nothing(self) -> None:
        chunks = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "write_file", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"path": "a' + ".txt"}},
        ]
        assembler, _ = self.feed(chunks)
        self.assertEqual(assembler.message()["content"][0]["input"], {})


class AnthropicClientTest(unittest.TestCase):
    def make(self, think="on") -> AnthropicClient:
        return AnthropicClient({"base_url": "https://llm.internal.example.com", "api_key": "secret"}, think=think)

    def body(self, messages) -> dict:
        return {
            "model": "m",
            "max_tokens": 16000,
            "system": "SYS",
            "tools": [
                {"name": "read_file", "description": "d", "input_schema": {"type": "object"}, "strict": True}
            ],
            "messages": messages,
        }

    def test_headers_carry_the_key_both_ways_and_the_version(self) -> None:
        client = self.make()
        self.assertEqual(client.url, "https://llm.internal.example.com/v1/messages")
        self.assertEqual(client.headers["x-api-key"], "secret")
        self.assertEqual(client.headers["authorization"], "Bearer secret")
        self.assertEqual(client.headers["anthropic-version"], "2023-06-01")
        self.assertEqual(client.headers["content-type"], "application/json")

    def test_config_headers_override_the_defaults(self) -> None:
        client = AnthropicClient(
            {"base_url": "https://x", "api_key": "k", "headers": {"anthropic-version": "2024-10-22", "X-Extra": "1"}},
            think="on",
        )
        self.assertEqual(client.headers["anthropic-version"], "2024-10-22")
        self.assertEqual(client.headers["X-Extra"], "1")

    def test_build_request_maps_the_levels_to_budgets(self) -> None:
        request = self.make().build_request(self.body([{"role": "user", "content": "hi"}]))
        self.assertEqual(request["thinking"], {"type": "enabled", "budget_tokens": 8000})
        self.assertEqual(request["model"], "m")
        self.assertEqual(request["max_tokens"], 16000)
        self.assertEqual(request["system"], "SYS")
        self.assertEqual(request["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(
            request["tools"], [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}}]
        )
        self.assertTrue(request["stream"])

    def test_high_keeps_headroom_and_a_small_window_clamps(self) -> None:
        self.assertEqual(self.make("high").thinking_field(16000)["budget_tokens"], 14000)
        self.assertEqual(self.make("low").thinking_field(16000)["budget_tokens"], 2000)
        self.assertEqual(self.make("on").thinking_field(3000)["budget_tokens"], 1024)
        self.assertEqual(self.make("bogus").thinking_field(16000)["budget_tokens"], 8000)

    def test_thinking_off_strips_the_field_and_the_history(self) -> None:
        client = self.make("off")
        self.assertIsNone(client.thinking_field(16000))
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "ok"}],
            },
        ]
        request = client.build_request(self.body(history))
        self.assertNotIn("thinking", request)
        self.assertEqual(
            request["messages"],
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ],
        )

    def test_no_think_setting_leaves_the_history_untouched(self) -> None:
        request = self.make(None).build_request(self.body([{"role": "user", "content": "hi"}]))
        self.assertNotIn("thinking", request)
        self.assertEqual(request["messages"], [{"role": "user", "content": "hi"}])


class CreateMessageTest(unittest.TestCase):
    def setUp(self):
        self.client = AnthropicClient({"base_url": "https://x", "api_key": "k"}, think="on")
        self.body = {
            "model": "m",
            "max_tokens": 16000,
            "system": "SYS",
            "tools": [],
            "messages": [{"role": "user", "content": "hi"}],
        }

    def run_stream(self, lines) -> tuple:
        printer = FakePrinter()
        captured = {}

        def fake_open_stream(url, headers, request):
            captured["url"] = url
            captured["headers"] = headers
            captured["request"] = request
            return FakeResponse(lines)

        with mock.patch.object(anthropic_client, "open_stream", fake_open_stream):
            result = self.client.create_message(self.body, printer)
        return result, printer, captured

    def test_parses_the_sse_stream_and_closes_the_printer(self) -> None:
        result, printer, captured = self.run_stream(sse_lines(TOOL_TURN))
        self.assertEqual(captured["url"], "https://x/v1/messages")
        self.assertEqual(captured["headers"]["x-api-key"], "k")
        self.assertTrue(captured["request"]["stream"])
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(result["content"][2]["input"], {"pattern": "it"})
        self.assertEqual(result["usage"], Usage(input=10, output=5))
        self.assertEqual(printer.shown_thinking, ["hm"])
        self.assertEqual(printer.shown_text, ["looking"])
        self.assertGreaterEqual(printer.closed, 1)

    def test_a_stream_without_message_stop_is_a_connection_error(self) -> None:
        with self.assertRaises(APIConnectionError):
            self.run_stream(sse_lines(TOOL_TURN[:-1]))

    def test_a_non_json_data_line_is_an_api_error(self) -> None:
        with self.assertRaises(APIError) as caught:
            self.run_stream([b"data: not json\n"])
        self.assertEqual(caught.exception.status, 200)

    def test_a_midstream_error_event_fails_the_request(self) -> None:
        chunks = TOOL_TURN[:3] + [
            {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}
        ]
        printer = FakePrinter()

        def fake_open_stream(url, headers, request):
            return FakeResponse(sse_lines(chunks))

        with mock.patch.object(anthropic_client, "open_stream", fake_open_stream):
            with self.assertRaises(APIError) as caught:
                self.client.create_message(self.body, printer)
        self.assertEqual(caught.exception.message, "overloaded")
        self.assertGreaterEqual(printer.closed, 1)


if __name__ == "__main__":
    unittest.main()
