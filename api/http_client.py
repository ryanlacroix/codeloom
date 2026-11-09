"""Shared HTTP plumbing for the Ollama Cloud client, using only the standard library.

Provides the exception types the API raises, the retry policy for failures
before a response starts, and the streaming POST they come from.
"""

import http.client
import json
import random
import time
import urllib.error
import urllib.request

REQUEST_TIMEOUT = 600  # seconds
MAX_RETRIES = 2


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
    """The API's message from a failed request's JSON body.

    Ollama sends {"error": "<message>"}; a non-string value in that slot falls
    back to its string form, and a body that is not JSON at all falls back to
    the HTTP reason phrase.
    """
    try:
        detail = json.loads(error.read())["error"]
    except (ValueError, KeyError, TypeError):
        return str(error.reason)
    if isinstance(detail, str):
        return detail
    return str(detail.get("message") or detail)


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
