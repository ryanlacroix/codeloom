"""Minimal client for the Claude Messages API, using only the standard library."""

import http.client
import json
import random
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 600  # seconds
MAX_RETRIES = 2
# oauth-2025-04-20: required when authenticating with an OAuth access token.
# server-side-fallback-2026-07-01: lets the API retry a refused turn on a fallback model.
BETAS = ["oauth-2025-04-20", "server-side-fallback-2026-07-01"]


class APIError(Exception):
    """The API returned an error status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class APIConnectionError(Exception):
    """The API could not be reached."""


def is_retryable(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


def retry_delay(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before the next attempt, honouring a retry-after header."""
    try:
        seconds = float(retry_after)
        if 0 <= seconds <= 60:
            return seconds
    except (TypeError, ValueError):
        pass
    return min(8.0, 0.5 * 2**attempt) * (1 - 0.25 * random.random())


def error_message(error: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(error.read())["error"]
        # Claude nests the message in an object; Ollama sends a plain string.
        return detail if isinstance(detail, str) else detail["message"]
    except (ValueError, KeyError, TypeError):
        return str(error.reason)


STREAM_ERROR_STATUS = {"rate_limit_error": 429, "overloaded_error": 529}


def open_stream(url: str, headers: dict, body: dict) -> http.client.HTTPResponse:
    """POST a JSON body and return the open response, to be read as a stream.

    Failures before the response starts are retried. Once it has started the
    caller handles errors: a retry would repeat output already shown.
    """
    data = json.dumps(body).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
        except urllib.error.HTTPError as e:
            if attempt < MAX_RETRIES and is_retryable(e.code):
                time.sleep(retry_delay(attempt, e.headers.get("retry-after")))
                continue
            raise APIError(e.code, error_message(e)) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(retry_delay(attempt, None))
                continue
            raise APIConnectionError(str(getattr(e, "reason", e))) from None


def sse_events(response: http.client.HTTPResponse):
    """Yield the decoded JSON data of each server-sent event."""
    data_lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield json.loads("\n".join(data_lines))
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))


def echoable_content(content: list) -> list:
    """Drop what must not be run or sent back after a mid-stream fallback.

    When a model declines part-way through a streamed reply, its partial
    output stays in the content, followed by a fallback block and the
    fallback model's reply. Only the text before the last fallback block may
    be sent back; its thinking and tool calls must be dropped.
    """
    boundaries = [i for i, block in enumerate(content) if block["type"] == "fallback"]
    if not boundaries:
        return content
    last = boundaries[-1]
    return [block for block in content[:last] if block["type"] in ("text", "fallback")] + content[last:]


class Client:
    """Minimal client for the Claude Messages API."""

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": API_VERSION,
            "anthropic-beta": ",".join(BETAS),
            "content-type": "application/json",
        }

    def create_message(self, body: dict, printer) -> dict:
        """Stream a response, showing thinking and text through printer as they arrive,
        and return the complete message in the shape of a non-streaming response."""
        message = None
        blocks = {}  # content blocks by index
        tool_json = {}  # partial tool input JSON by block index
        finished = False
        with open_stream(API_URL, self.headers, {**body, "stream": True}) as response:
            try:
                for event in sse_events(response):
                    kind = event["type"]
                    if kind == "message_start":
                        message = event["message"]
                    elif kind == "content_block_start":
                        block = blocks[event["index"]] = event["content_block"]
                        if block["type"] == "tool_use":
                            tool_json[event["index"]] = ""
                        elif block["type"] == "fallback":
                            from_model = block.get("from", {}).get("model", "a model")
                            to_model = block.get("to", {}).get("model", "another model")
                            printer.notice(f"{from_model} declined; {to_model} continued")
                    elif kind == "content_block_delta":
                        block, delta = blocks[event["index"]], event["delta"]
                        if delta["type"] == "text_delta":
                            block["text"] = block.get("text", "") + delta["text"]
                            printer.text(delta["text"])
                        elif delta["type"] == "thinking_delta":
                            block["thinking"] = block.get("thinking", "") + delta["thinking"]
                            printer.thinking(delta["thinking"])
                        elif delta["type"] == "signature_delta":
                            block["signature"] = block.get("signature", "") + delta["signature"]
                        elif delta["type"] == "input_json_delta":
                            tool_json[event["index"]] += delta["partial_json"]
                    elif kind == "content_block_stop":
                        printer.end_block()
                        if event["index"] in tool_json:
                            try:
                                blocks[event["index"]]["input"] = json.loads(tool_json.pop(event["index"]) or "{}")
                            except ValueError:
                                # Only a cut-off tool call has incomplete JSON, and those are not run.
                                blocks[event["index"]]["input"] = {}
                    elif kind == "message_delta":
                        message.update(event.get("delta") or {})
                        message["usage"] = {**(message.get("usage") or {}), **(event.get("usage") or {})}
                    elif kind == "message_stop":
                        finished = True
                        break
                    elif kind == "error":
                        error = event.get("error") or {}
                        status = STREAM_ERROR_STATUS.get(error.get("type"), 500)
                        raise APIError(status, error.get("message") or "the API reported an error mid-response")
            except ValueError:
                raise APIError(200, "the API sent a streamed event that is not valid JSON") from None
            except (OSError, http.client.HTTPException) as e:
                raise APIConnectionError(f"the connection dropped mid-response: {e}") from None
            finally:
                printer.end_block()
        if message is None or not finished:
            raise APIConnectionError("the response ended before it was complete")
        message["content"] = echoable_content([blocks[i] for i in sorted(blocks)])
        return message
