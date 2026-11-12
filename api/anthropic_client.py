"""Client for an Anthropic-compatible /v1/messages endpoint, using only the standard library.

The harness keeps its history in Anthropic's Messages shape (role-tagged messages whose
content is text or typed blocks such as tool_use and tool_result), and its tool schemas
are Anthropic's too, so a request passes through almost unchanged. What this client adds
is the wire format: the request needs the endpoint's URL and auth headers, thinking is a
token budget rather than an on/off flag, and the reply arrives as server-sent events whose
data lines are reassembled into the harness's response shape.

The endpoint is described by a JSON file (load_endpoint_config), kept in the harness's
config/ directory — it holds the API key, and that whole directory is passed to the
Workspace's protected_dirs in loom.py, so the model can never list or read it.
"""

import http.client
import json
from pathlib import Path

from api.http_client import APIConnectionError, APIError, open_stream
from api.models import Usage

# Thinking budgets by --think level, in tokens. "off" is handled separately
# (it disables the request's thinking field entirely); any unknown level falls
# back to "on"'s budget. Anthropic requires a budget of at least 1024 tokens and
# below max_tokens, so build_request clamps high budgets into that window.
THINK_BUDGETS = {"low": 2000, "on": 8000, "medium": 8000, "high": 14000}
DEFAULT_ANTHROPIC_VERSION = "2026-10-11"


def endpoint_url(base_url: str) -> str:
    """The full messages URL: /v1/messages appended to the configured base.

    A base_url that already ends with the endpoint is used as given, so a
    config that pasted the whole URL keeps working.
    """
    base = base_url.rstrip("/")
    return base if base.endswith("/v1/messages") else base + "/v1/messages"


def load_endpoint_config(path: Path) -> tuple[dict, str | None]:
    """Read the --api-config JSON file, returning (config, error note).

    The file describes one Anthropic-compatible endpoint and holds its API key,
    so it lives in the harness's config/ directory, which the Workspace refuses
    outright:

        {
          "base_url": "https://llm.internal.example.com",
          "api_key": "sk-...",
          "model": "claude-sonnet-4-5",
          "models": ["claude-sonnet-4-5", "claude-haiku-4-5"],
          "headers": {"anthropic-version": "2023-06-01"}
        }

    base_url (the server; /v1/messages is appended) and api_key are required.
    model is the default --model; models is the list /model offers (whose first
    entry becomes the default model when model is absent); headers adds or
    overrides request headers, for gateways that want more than the key. Every
    problem returns an empty config plus the note, which loom.py exits with.
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

    url = config.get("base_url")
    if not isinstance(url, str) or not url.strip():
        return {}, f'{path} needs a "base_url" string (the server; /v1/messages is appended)'
    url = url.strip().rstrip("/")
    if "://" not in url:
        return {}, f'{path}: "base_url" needs its scheme, e.g. https://llm.internal.example.com'
    key = config.get("api_key")
    if not isinstance(key, str) or not key.strip():
        return {}, f'{path} needs an "api_key" string'
    result = {"base_url": url, "api_key": key.strip()}

    if "model" in config:
        if not isinstance(config["model"], str) or not config["model"].strip():
            return {}, f'{path}: "model" must be a non-empty string'
        result["model"] = config["model"].strip()
    if "models" in config:
        models = config["models"]
        if not isinstance(models, list) or any(not isinstance(m, str) or not m.strip() for m in models):
            return {}, f'{path}: "models" must be a list of model-name strings'
        result["models"] = [m.strip() for m in models]
    # Without an explicit "model", the first "models" entry is the default.
    result.setdefault("model", (result.get("models") or [None])[0])

    if "headers" in config:
        headers = config["headers"]
        if not isinstance(headers, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()
        ):
            return {}, f'{path}: "headers" must be an object mapping header names to string values'
        result["headers"] = headers
    return result, None


def to_anthropic_tools(tools: list) -> list:
    """The request's tool definitions: Anthropic's own shape, minus foreign keys.

    core/tools.py already emits name/description/input_schema in Anthropic's
    format; the Ollama-only "strict" key is dropped so a strict server does not
    reject the definitions for an unknown field.
    """
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in tools
    ]


def to_anthropic_messages(messages: list, strip_thinking: bool = False) -> list:
    """The history for the request: passed through as-is unless thinking is off.

    The harness stores messages in Anthropic's shape, including any thinking
    signatures the server sent, so they need no conversion. Only when thinking
    is off server-side do the stored thinking blocks have to go: the API rejects
    a request that carries thinking or redacted_thinking blocks while thinking
    is disabled. An assistant turn whose blocks were all thinking (a reply cut
    off while it was still thinking) would then be left with an empty content
    array, which the API rejects too, so the whole message is dropped: it cannot
    hold a tool_use, so no tool_result is orphaned, and the API merges
    consecutive user turns, so the request stays valid without it.
    """
    if not strip_thinking:
        return messages
    result = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            result.append(message)
            continue
        kept = [b for b in content if b.get("type") not in ("thinking", "redacted_thinking")]
        if kept:
            result.append({"role": message["role"], "content": kept})
    return result


def tool_input(buffer: str) -> dict:
    """The accumulated tool_use input_json_delta string, as an object.

    Empty input counts as {}, and a buffer that is not a JSON object falls back
    to {} as well: a partial buffer only arises when the output limit cut the
    call off mid-stream, and the agent drops tool_use blocks from such a reply
    before the input is ever used.
    """
    if not buffer.strip():
        return {}
    try:
        value = json.loads(buffer)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class StreamAssembler:
    """Reassembles one streamed /v1/messages response into the harness's shape.

    feed() takes each SSE data-line payload in arrival order (message_start,
    content_block_start/delta/stop, message_delta, message_stop); message()
    returns the finished response, thinking and text having been shown through
    the printer as their deltas arrived. An error event raises APIError from
    feed() itself, so the request fails as soon as the server reports it.
    """

    def __init__(self, printer):
        self.printer = printer
        self.blocks = {}  # SSE block index -> {"block": its start payload, plus the pieces}
        self.model = None
        self.stop_reason = "end_turn"
        self.input_tokens = 0
        self.output_tokens = 0
        self.done = False

    def feed(self, chunk: dict) -> None:
        kind = chunk.get("type")
        if kind == "message_start":
            message = chunk.get("message") or {}
            self.model = message.get("model")
            usage = message.get("usage") or {}
            self.input_tokens = usage.get("input_tokens") or 0
        elif kind == "content_block_start":
            block = dict(chunk.get("content_block") or {})
            self.blocks[chunk.get("index", len(self.blocks))] = {
                "block": block,
                "text": "",       # text_delta pieces
                "thinking": "",   # thinking_delta pieces
                "signature": "",  # signature_delta pieces
                "input_json": "",  # input_json_delta pieces
            }
            # Real servers open a block empty; if one opens with text or
            # thinking in hand, show it, like its later deltas.
            if block.get("type") == "text" and block.get("text"):
                self.printer.text(block["text"])
            elif block.get("type") == "thinking" and block.get("thinking"):
                self.printer.thinking(block["thinking"])
        elif kind == "content_block_delta":
            self._delta(chunk.get("index", 0), chunk.get("delta") or {})
        elif kind == "message_delta":
            self.stop_reason = (chunk.get("delta") or {}).get("stop_reason") or self.stop_reason
            usage = chunk.get("usage") or {}
            self.output_tokens = usage.get("output_tokens") or self.output_tokens
            if usage.get("input_tokens"):
                self.input_tokens = usage["input_tokens"]
        elif kind == "message_stop":
            self.done = True
        elif kind == "error":
            error = chunk.get("error") or {}
            raise APIError(
                500, str(error.get("message") or error.get("type") or "the API sent an error event")
            )

    def _delta(self, index, delta: dict) -> None:
        state = self.blocks.get(index)
        if state is None:
            return
        kind = delta.get("type")
        if kind == "text_delta":
            text = delta.get("text") or ""
            state["text"] += text
            self.printer.text(text)
        elif kind == "thinking_delta":
            thinking = delta.get("thinking") or ""
            state["thinking"] += thinking
            self.printer.thinking(thinking)
        elif kind == "signature_delta":
            state["signature"] += delta.get("signature") or ""
        elif kind == "input_json_delta":
            state["input_json"] += delta.get("partial_json") or ""

    def message(self) -> dict:
        """The response in the harness's shape, in the server's block order."""
        content = [self._finish(self.blocks[index]) for index in sorted(self.blocks)]
        return {
            "model": self.model,
            "content": content,
            "stop_reason": self.stop_reason,
            "usage": Usage(input=self.input_tokens, output=self.output_tokens),
        }

    def _finish(self, state: dict) -> dict:
        block = state["block"]
        kind = block.get("type")
        if kind == "text":
            return {"type": "text", "text": block.get("text", "") + state["text"]}
        if kind == "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id") or "",
                "name": block.get("name") or "",
                "input": tool_input(state["input_json"]),
            }
        if kind == "thinking":
            # The signature travels back in the history when the harness keeps
            # the block: the API requires it to accept the thinking again.
            thinking = {"type": "thinking", "thinking": block.get("thinking", "") + state["thinking"]}
            signature = block.get("signature", "") + state["signature"]
            if signature:
                thinking["signature"] = signature
            return thinking
        if kind == "redacted_thinking":
            return {"type": "redacted_thinking", "data": block.get("data", "")}
        return block  # an unrecognized block passes through unchanged


class AnthropicClient:
    """Client for an Anthropic-compatible /v1/messages endpoint, in the harness's shapes.

    The same interface as OllamaClient, so the Agent and the /thinking commands
    work with either: create_message(body, printer) takes the body the Agent
    builds and returns the response in the harness's shape, and .think holds the
    thinking setting ("off", or a level mapped to a token budget). The endpoint
    (base URL, key, extra headers) comes from load_endpoint_config's file.
    """

    def __init__(self, config: dict, think: str | None = None):
        self.think = think
        self.url = endpoint_url(config["base_url"])
        key = config["api_key"]
        # The key goes out as a Bearer token only; x-api-key is not sent. The
        # config's "headers" is applied last, so it can add or override any of
        # these (including anthropic-version for a server pinned to a different
        # one — or x-api-key itself, for an endpoint that wants it).
        self.headers = {
            "authorization": f"Bearer {key}",
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        self.headers.update(config.get("headers") or {})

    def thinking_field(self, max_tokens: int) -> dict | None:
        """The request's thinking field for the current think setting, if any.

        "off" (or no setting) returns None: thinking stays disabled and the
        stored thinking blocks are stripped from the history (see
        to_anthropic_messages). A level maps to a budget kept under max_tokens
        — Anthropic requires that, and 2000 tokens of headroom are left for the
        reply itself — and never below the API's 1024-token minimum. Turning
        thinking back on mid-session can still be refused by a strict server:
        turns produced while it was off carry no thinking block to pair with
        their tool calls. Keep one setting for a conversation, or /reset.
        """
        if not self.think or self.think == "off":
            return None
        budget = THINK_BUDGETS.get(self.think, THINK_BUDGETS["on"])
        budget = min(budget, max(1024, max_tokens - 2000))
        return {"type": "enabled", "budget_tokens": budget}

    def build_request(self, body: dict) -> dict:
        """The POST body for one Agent request, in /v1/messages terms."""
        request = {
            "model": body["model"],
            "max_tokens": body["max_tokens"],
            "system": body["system"],
            "messages": to_anthropic_messages(body["messages"], strip_thinking=self.think == "off"),
            "tools": to_anthropic_tools(body.get("tools", [])),
            "stream": True,
        }
        thinking = self.thinking_field(body["max_tokens"])
        if thinking:
            request["thinking"] = thinking
        return request

    def create_message(self, body: dict, printer) -> dict:
        """Stream a response, showing thinking and text through printer as they
        arrive, and return the complete message in the harness's response shape.

        The reply is server-sent events: each "data:" line holds one JSON object
        whose "type" says what it carries (thinking and reply text arrive as
        deltas and tool inputs as JSON fragments). The lines are fed to a
        StreamAssembler in order; anything that is not a data line — event
        headers, keep-alive comments, blank separators — is skipped.
        """
        request = self.build_request(body)
        assembler = StreamAssembler(printer)
        with open_stream(self.url, self.headers, request) as response:
            try:
                for raw in response:
                    # HTTPResponse iterates bytes; tests may hand over str.
                    line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        raise APIError(200, "the API sent a streamed line that is not valid JSON") from None
                    assembler.feed(chunk)
                    if assembler.done:
                        break
            except (OSError, http.client.HTTPException) as e:
                raise APIConnectionError(f"the connection dropped mid-response: {e}") from None
            finally:
                printer.end_block()
        if not assembler.done:
            raise APIConnectionError("the response ended before it was complete")
        return assembler.message()
