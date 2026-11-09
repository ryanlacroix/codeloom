"""Tests for api.ollama_client: the translation to and from Ollama's chat API."""

import json
import tempfile
import unittest
from pathlib import Path

from api.http_client import APIError
from api.models import Usage
from api.ollama_client import (
    OLLAMA_DEFAULT_BASE_URL,
    OllamaClient,
    chat_url,
    from_ollama_response,
    load_ollama_config,
    result_text,
    to_ollama_messages,
    to_ollama_tools,
    tool_arguments,
    tool_names_by_id,
)


def history() -> list:
    """A two-turn history with a tool call in it, in the harness's message shape."""
    return [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "looking"},
                {"type": "tool_use", "id": "1", "name": "search", "input": {"pattern": "it"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "found it"}]},
    ]


class ToOllamaMessagesTest(unittest.TestCase):
    def test_system_is_prepended_and_plain_users_pass_through(self) -> None:
        messages = to_ollama_messages("SYS", [{"role": "user", "content": "hi"}])
        self.assertEqual(messages, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])

    def test_assistant_blocks_become_content_thinking_and_tool_calls(self) -> None:
        converted = to_ollama_messages("S", history())
        self.assertEqual(
            converted[2],
            {
                "role": "assistant",
                "content": "looking",
                "thinking": "hmm",
                "tool_calls": [{"function": {"name": "search", "arguments": {"pattern": "it"}}}],
            },
        )

    def test_tool_results_carry_the_tool_name(self) -> None:
        converted = to_ollama_messages("S", history())
        self.assertEqual(converted[3], {"role": "tool", "tool_name": "search", "content": "found it"})

    def test_read_file_note_blocks_join_into_the_result_text(self) -> None:
        converted = to_ollama_messages("S", [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "9", "name": "read_file", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "9", "content": [{"type": "text", "text": "note"}]},
                ],
            },
        ])
        self.assertEqual(converted[-1]["tool_name"], "read_file")
        self.assertEqual(converted[-1]["content"], "note")

    def test_tool_names_by_id(self) -> None:
        self.assertEqual(tool_names_by_id(history()), {"1": "search"})

    def test_result_text_joins_text_blocks(self) -> None:
        self.assertEqual(result_text("plain"), "plain")
        self.assertEqual(
            result_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]), "a\nb"
        )


class FromOllamaResponseTest(unittest.TestCase):
    def test_text_reply_with_usage(self) -> None:
        response = {"model": "m", "message": {"content": "hi"}, "prompt_eval_count": 10, "eval_count": 5}
        result = from_ollama_response(response)
        self.assertEqual(result["model"], "m")
        self.assertEqual(result["content"], [{"type": "text", "text": "hi"}])
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"], Usage(input=10, output=5))

    def test_thinking_precedes_the_text_block(self) -> None:
        result = from_ollama_response({"message": {"thinking": "th", "content": "c"}})
        self.assertEqual(
            result["content"], [{"type": "thinking", "thinking": "th"}, {"type": "text", "text": "c"}]
        )

    def test_tool_calls_set_the_stop_reason_and_parse_json_arguments(self) -> None:
        response = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "search", "arguments": '{"pattern": "x"}'}}],
            }
        }
        result = from_ollama_response(response)
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(result["content"][0]["type"], "tool_use")
        self.assertEqual(result["content"][0]["name"], "search")
        self.assertEqual(result["content"][0]["input"], {"pattern": "x"})
        self.assertTrue(result["content"][0]["id"].startswith("call_"))

    def test_done_reason_length_maps_to_max_tokens(self) -> None:
        response = {"done_reason": "length", "message": {"content": "partial"}}
        self.assertEqual(from_ollama_response(response)["stop_reason"], "max_tokens")

    def test_a_response_without_a_message_raises(self) -> None:
        with self.assertRaises(APIError):
            from_ollama_response({"error": "bad"})

    def test_missing_usage_counts_as_zero(self) -> None:
        self.assertEqual(from_ollama_response({"message": {"content": "x"}})["usage"], Usage())


class HelpersTest(unittest.TestCase):
    def test_tool_arguments_tolerates_json_strings_and_junk(self) -> None:
        self.assertEqual(tool_arguments({"a": 1}), {"a": 1})
        self.assertEqual(tool_arguments('{"a": 1}'), {"a": 1})
        self.assertEqual(tool_arguments("not json"), {})
        self.assertEqual(tool_arguments(None), {})
        self.assertEqual(tool_arguments([1]), {})

    def test_tools_translate_to_function_definitions(self) -> None:
        tools = [{"name": "read_file", "description": "d", "input_schema": {"type": "object"}}]
        self.assertEqual(
            to_ollama_tools(tools),
            [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}}}],
        )


class ChatUrlTest(unittest.TestCase):
    def test_api_chat_is_appended_to_the_base(self) -> None:
        self.assertEqual(chat_url(OLLAMA_DEFAULT_BASE_URL), "https://ollama.com/api/chat")
        self.assertEqual(chat_url("https://ollama.example.com/"), "https://ollama.example.com/api/chat")

    def test_a_base_already_named_the_endpoint_is_used_as_given(self) -> None:
        self.assertEqual(chat_url("https://example.com/api/chat"), "https://example.com/api/chat")


class LoadOllamaConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ollama_config.json"
        self.addCleanup(self._tmp.cleanup)

    def write(self, payload) -> None:
        self.path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")

    def test_valid_config(self) -> None:
        self.write(
            {
                "api_key": " secret ",
                "base_url": "https://ollama.example.com/",
                "model": "m1",
                "models": ["m1", "m2"],
                "headers": {"x-gateway": "v"},
            }
        )
        config, error = load_ollama_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(
            config,
            {
                "api_key": "secret",
                "base_url": "https://ollama.example.com",
                "model": "m1",
                "models": ["m1", "m2"],
                "headers": {"x-gateway": "v"},
            },
        )

    def test_base_url_defaults_to_ollama_cloud(self) -> None:
        self.write({"api_key": "k"})
        config, error = load_ollama_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(config["base_url"], OLLAMA_DEFAULT_BASE_URL)
        self.assertIsNone(config["model"])
        self.assertNotIn("models", config)
        self.assertNotIn("headers", config)

    def test_first_models_entry_is_the_default_model(self) -> None:
        self.write({"api_key": "k", "models": ["a", "b"]})
        config, error = load_ollama_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(config["model"], "a")

    def test_utf8_bom_is_tolerated(self) -> None:
        self.path.write_bytes(json.dumps({"api_key": "k"}).encode("utf-8-sig"))
        config, error = load_ollama_config(self.path)
        self.assertIsNone(error)
        self.assertEqual(config["api_key"], "k")

    def test_missing_file(self) -> None:
        config, error = load_ollama_config(self.path)
        self.assertEqual(config, {})
        self.assertIn("No endpoint config found", error)

    def test_invalid_json(self) -> None:
        self.write("{oops")
        config, error = load_ollama_config(self.path)
        self.assertEqual(config, {})
        self.assertIn("not valid JSON", error)

    def test_wrong_shapes_are_reported(self) -> None:
        for payload, note in (
            ('["x"]', "must be a JSON object"),
            ('{"base_url": "https://x"}', 'needs an "api_key"'),
            ('{"api_key": ""}', 'needs an "api_key"'),
            ('{"api_key": "k", "base_url": ""}', '"base_url" must be a non-empty string'),
            ('{"api_key": "k", "base_url": "ollama.example.com"}', "needs its scheme"),
            ('{"api_key": "k", "model": 3}', '"model" must be a non-empty string'),
            ('{"api_key": "k", "models": "m"}', '"models" must be a list'),
            ('{"api_key": "k", "models": [1]}', '"models" must be a list'),
            ('{"api_key": "k", "headers": {"a": 1}}', '"headers" must be an object'),
        ):
            with self.subTest(payload=payload):
                self.write(payload)
                config, error = load_ollama_config(self.path)
                self.assertEqual(config, {})
                self.assertIn(note, error)


class ClientTest(unittest.TestCase):
    def test_endpoint_key_and_headers_come_from_the_config(self) -> None:
        client = OllamaClient({"api_key": "k", "base_url": "https://x", "headers": {"x-a": "b"}})
        self.assertEqual(client.url, "https://x/api/chat")
        self.assertEqual(
            client.headers,
            {"authorization": "Bearer k", "content-type": "application/json", "x-a": "b"},
        )


if __name__ == "__main__":
    unittest.main()
