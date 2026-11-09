"""Minimal client for an Ollama /api/chat endpoint, using only the standard library.

The harness keeps its history in a Messages-style shape (role-tagged messages
whose content is text or typed blocks such as tool_use and tool_result). This
client translates each request into Ollama's /api/chat format and translates
the reply back into that shape.

The endpoint is described by a JSON file (load_ollama_config), kept in the
harness's config/ directory — it holds the API key, and that whole directory
is passed to the Workspace's protected_dirs in loom.py, so the model can never
list or read it.
"""

import http.client
import json
import uuid
from pathlib import Path

from api.http_client import APIConnectionError, APIError, open_stream
from api.models import Usage

# The base_url used when the config file gives none: /api/chat is appended by
# chat_url, so a base_url that already ends with the endpoint keeps working.
OLLAMA_DEFAULT_BASE_URL = "https://ollama.com"


def chat_url(base_url: str) -> str:
    """The full chat URL: /api/chat appended to the configured base.

    A base_url that already ends with the endpoint is used as given, so a
    config that pasted the whole URL keeps working.
    """
    base = base_url.rstrip("/")
    return base if base.endswith("/api/chat") else base + "/api/chat"


def load_ollama_config(path: Path) -> tuple[dict, str | None]:
    """Read the Ollama endpoint's config file, returning (config, error note).

    The file describes the Ollama endpoint and holds its API key, so it lives
    in the harness's config/ directory, which the Workspace refuses outright:

        {
          "api_key": "sk-...",
          "base_url": "https://ollama.com",
          "model": "glm-5.3-flash",
          "models": ["glm-5.3", "glm-5.3-flash", "minimax-m3"],
          "headers": {"x-custom": "value"}
        }

    api_key is required — it is the only key source, so it lives in the file
    like the Anthropic config's. base_url defaults to Ollama Cloud when absent
    (/api/chat is appended); model is the default --model, models is the list
    /model offers, and both fall back to api/models.py's defaults when absent;
    headers adds or overrides request headers. Every problem returns an empty
    config plus the note, which loom.py exits with.
    """
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}, f"No endpoint config found at {path}"
    except json.JSONDecodeError as e:
        return {}, f"{path} is not valid JSON: {e}"
    except (OSError, UnicodeDecodeError) as e:
        return {}, f"could not read {path}: {e}"
    if not isinstance(config, dict):
        return {}, f"{path} must be a JSON object"

    key = config.get("api_key")
    if not isinstance(key, str) or not key.strip():
        return {}, f'{path} needs an "api_key" string'
    result = {"api_key": key.strip()}

    url = config.get("base_url")
    if url is not None:
        if not isinstance(url, str) or not url.strip():
            return {}, f'{path}: "base_url" must be a non-empty string when present'
        url = url.strip().rstrip("/")
        if "://" not in url:
            return {}, f'{path}: "base_url" needs its scheme, e.g. {OLLAMA_DEFAULT_BASE_URL}'
        result["base_url"] = url
    result.setdefault("base_url", OLLAMA_DEFAULT_BASE_URL)

    if "model" in config:
        if not isinstance(config["model"], str) or not config["model"].strip():
            return {}, f'{path}: "model" must be a non-empty string'
        result["model"] = config["model"].strip()
    if "models" in config:
        models = config["models"]
        if not isinstance(models, list) or any(not isinstance(m, str) or not m.strip() for m in models):
            return {}, f'{path}: "models" must be a list of model-name strings'
        result["models"] = [m.strip() for m in models]
    # Without an explicit "model", the first "models" entry is the default
    # (api/models.py's default applies when both are absent — loom.py's job).
    result.setdefault("model", (result.get("models") or [None])[0])

    if "headers" in config:
        headers = config["headers"]
        if not isinstance(headers, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()
        ):
            return {}, f'{path}: "headers" must be an object mapping header names to string values'
        result["headers"] = headers
    return result, None


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


def to_ollama_messages(system: str, messages: list) -> list:
    result = [{"role": "system", "content": system}]
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
    """Convert an /api/chat reply into the harness's response shape, usage included."""
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
        "usage": Usage(
            input=response.get("prompt_eval_count") or 0,
            output=response.get("eval_count") or 0,
        ),
    }


class OllamaClient:
    """Minimal client for an Ollama /api/chat endpoint, in the harness's shapes.

    The same interface as AnthropicClient, so the Agent and the /thinking
    commands work with either: create_message(body, printer) takes the body
    the Agent builds and returns the response in the harness's shape, and
    .think holds the thinking setting. The endpoint (base URL, key, extra
    headers) comes from load_ollama_config's file.
    """

    def __init__(self, config: dict, think: str | None = None):
        # None leaves thinking to the model's default; "on" and "off" send true
        # and false; "low", "medium" or "high" set a level (gpt-oss models only
        # accept levels).
        self.think = think
        self.url = chat_url(config["base_url"])
        # The config's "headers" is applied last, so it can override any of
        # these (for gateways that want more than the key).
        self.headers = {
            "authorization": f"Bearer {config['api_key']}",
            "content-type": "application/json",
        }
        self.headers.update(config.get("headers") or {})

    def create_message(self, body: dict, printer) -> dict:
        """Stream a response, showing thinking and text through printer as they arrive,
        and return the complete message in the harness's response shape."""
        request = {
            "model": body["model"],
            "messages": to_ollama_messages(body["system"], body["messages"]),
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
        with open_stream(self.url, self.headers, request) as response:
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
