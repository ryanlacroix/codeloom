"""Minimal client for the Claude Messages API, using only the standard library."""

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


def post_json(url: str, headers: dict, body: dict) -> dict:
    """POST a JSON body and return the decoded JSON response, retrying transient failures."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if attempt < MAX_RETRIES and is_retryable(e.code):
                time.sleep(retry_delay(attempt, e.headers.get("retry-after")))
                continue
            raise APIError(e.code, error_message(e)) from None
        except ValueError:
            raise APIError(200, "the API returned a response that is not valid JSON") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(retry_delay(attempt, None))
                continue
            raise APIConnectionError(str(getattr(e, "reason", e))) from None


class Client:
    """Minimal client for the Claude Messages API."""

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": API_VERSION,
            "anthropic-beta": ",".join(BETAS),
            "content-type": "application/json",
        }

    def create_message(self, body: dict) -> dict:
        return post_json(API_URL, self.headers, body)
