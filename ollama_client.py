"""Minimal client for Ollama Cloud's chat API, using only the standard library.

The agent speaks the Claude Messages format. This client translates each
request into Ollama's /api/chat format and translates the reply back, so the
agent loop works unchanged with either provider.
"""

import http.client
import json
import uuid

from api_client import APIConnectionError, APIError, open_stream

API_URL = "https://ollama.com/api/chat"


def tool_names_by_id(messages: list) -> dict:
    """Map each tool_use id in the history to its tool name (Ollama matches results by name)."""
    names = {}
    for message in messages:
        if message["role"] == "assistant" and isinstance(message["content"], list):
            for block in message["content"]:
                if block["type"] == "tool_use":
                    names[block["id"]] = block["name"]
    return names


def result_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block["type"] == "text")


def to_ollama_messages(system: list, messages: list) -> list:
    result = [{"role": "system", "content": "\n\n".join(block["text"] for block in system)}]
    names = tool_names_by_id(messages)
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            result.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            converted = {
                "role": "assistant",
                "content": "".join(b["text"] for b in content if b["type"] == "text"),
            }
            thinking = "".join(b["thinking"] for b in content if b["type"] == "thinking")
            if thinking:
                converted["thinking"] = thinking
            calls = [
                {"function": {"name": b["name"], "arguments": b["input"]}} for b in content if b["type"] == "tool_use"
            ]
            if calls:
                converted["tool_calls"] = calls
            result.append(converted)
        else:
            for block in content:
                if block["type"] == "tool_result":
                    result.append(
                        {
                            "role": "tool",
                            "tool_name": names.get(block["tool_use_id"], ""),
                            "content": result_text(block["content"]),
                        }
                    )
                elif block["type"] == "text":
                    result.append({"role": "user", "content": block["text"]})
    return result


def to_ollama_tools(tools: list) -> list:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


def tool_arguments(arguments) -> dict:
    """Arguments should be an object, but tolerate a JSON-encoded string."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def from_ollama_response(response: dict) -> dict:
    """Convert an /api/chat reply into the shape of a Messages API response."""
    if "message" not in response:
        raise APIError(200, response.get("error") or "the API returned a response with no message")
    message = response["message"]
    content = []
    if message.get("thinking"):
        content.append({"type": "thinking", "thinking": message["thinking"]})
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function.get("name", ""),
                "input": tool_arguments(function.get("arguments")),
            }
        )

    if any(block["type"] == "tool_use" for block in content):
        stop_reason = "tool_use"
    elif response.get("done_reason") == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return {
        "model": response.get("model"),
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": response.get("prompt_eval_count") or 0,
            "output_tokens": response.get("eval_count") or 0,
        },
    }


class OllamaClient:
    """Minimal client for Ollama Cloud, accepting and returning Messages API shapes."""

    def __init__(self, api_key: str, think: str | None = None):
        # None leaves thinking to the model's default; "on" and "off" send true
        # and false; "low", "medium" or "high" set a level (gpt-oss models only
        # accept levels).
        self.think = think
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    def create_message(self, body: dict, printer) -> dict:
        """Stream a response, showing thinking and text through printer as they arrive,
        and return the complete message in the shape of a Messages API response."""
        # Claude-only fields (cache_control, output_config, thinking, fallbacks) are dropped.
        request = {
            "model": body["model"],
            "messages": to_ollama_messages(body.get("system", []), body["messages"]),
            "tools": to_ollama_tools(body.get("tools", [])),
            "stream": True,
            "options": {"num_predict": body["max_tokens"]},
        }
        if self.think:
            request["think"] = {"on": True, "off": False}.get(self.think, self.think)

        # Each line of the stream is a JSON object holding the next piece of the
        # message; the last one has done: true and the token counts.
        thinking, text, tool_calls = [], [], []
        final = None
        with open_stream(API_URL, self.headers, request) as response:
            try:
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise APIError(500, str(chunk["error"]))
                    message = chunk.get("message") or {}
                    if message.get("thinking"):
                        thinking.append(message["thinking"])
                        printer.thinking(message["thinking"])
                    if message.get("content"):
                        text.append(message["content"])
                        printer.text(message["content"])
                    tool_calls.extend(message.get("tool_calls") or [])
                    if chunk.get("done"):
                        final = chunk
                        break
            except ValueError:
                raise APIError(200, "the API sent a streamed line that is not valid JSON") from None
            except (OSError, http.client.HTTPException) as e:
                raise APIConnectionError(f"the connection dropped mid-response: {e}") from None
            finally:
                printer.end_block()
        if final is None:
            raise APIConnectionError("the response ended before it was complete")
        final["message"] = {"thinking": "".join(thinking), "content": "".join(text), "tool_calls": tool_calls}
        return from_ollama_response(final)
