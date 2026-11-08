"""Redaction of the strings a developer lists in a config file.

Every text sent to the model — user messages, /paste buffers, and every tool
result (file contents, search matches, directory listings, error strings) —
passes through this filter, applied in core/agent.py at the two places content
joins the conversation. Each configured string is replaced by the configured
marker ("<redacted>" by default), so a secret appearing anywhere in the
harness's inputs reaches neither the API nor the stored conversation history.

The config lives in redact.json next to loom.py (loom.py's --redact-file moves
it), outside the workspace: the model can neither read nor edit its own
redaction list — the file is on the Workspace's blocked list as well, in case
the harness's own directory is ever the workspace. It is read once at startup,
so edits take effect on the next launch.

redact.json is a JSON object holding one required key and one optional:

    {
      "redactList": [
        "supercalafrajalistic",
        "expialadocious"
      ],
      "replaceWith": "<redacted>"
    }

"redactList" is the array of literal strings to redact, used verbatim with
empty strings dropped (the old '#' comment and blank-line rules are gone with
the line format); "replaceWith" is the marker every entry is replaced with,
"<redacted>" by default. A missing file is the normal opt-out (redaction is
opt-in); a file that exists but is unreadable or wrongly shaped turns
redaction off behind an error note that loom.py prints as a warning that text
goes unredacted — never a silent failure.

Matching is literal (no regular expressions), case-sensitive, and unanchored,
in one scan of each text with the entries longest first — so where one entry
contains another the longer is redacted whole, and because re.sub never
rescans the text it has replaced, no entry can ever rewrite the marker.
"""

import json
import re
from pathlib import Path
from typing import TypeVar

# The marker every configured string is replaced with: redact.json's
# "replaceWith", which defaults to this when the key is absent.
REDACTED = "<redacted>"

# Whatever shape a tool result comes back in: redact_content() preserves it.
_T = TypeVar("_T")


def load_redactions(path: Path) -> tuple[list[str], str, str | None]:
    """Read the redaction config, returning (entries, marker, error note).

    The file is the JSON object described in the module docstring: "replaceWith"
    chooses the marker ("<redacted>" by default) and "redactList" is the array
    of strings to redact, used verbatim with empties dropped. A missing file is
    the normal opt-out (redaction is opt-in) and returns no note, so loom.py
    stays quiet about it; a file that exists but is unreadable, not valid JSON,
    or the wrong shape returns no entries plus the note, which loom.py prints
    as a warning that text goes unredacted.
    """
    marker = REDACTED
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return [], marker, None
    except json.JSONDecodeError as e:
        return [], marker, f"{path} is not valid JSON: {e}"
    except (OSError, UnicodeDecodeError) as e:
        return [], marker, f"could not read {path}: {e}"

    if not isinstance(config, dict):
        return [], marker, f"{path} is not a JSON object"
    if not isinstance(config.get("redactList"), list):
        return [], marker, f"{path} needs a \"redactList\" array of strings"
    if any(not isinstance(entry, str) for entry in config["redactList"]):
        return [], marker, f"{path}: \"redactList\" must be a list of strings"
    if "replaceWith" in config and (
        not isinstance(config["replaceWith"], str) or not config["replaceWith"]
    ):
        return [], marker, f"{path}: \"replaceWith\" must be a non-empty string"
    marker = config.get("replaceWith", marker)
    return [entry for entry in config["redactList"] if entry], marker, None


class Redactor:
    """The filter agent.py applies to everything that enters a conversation.

    Truthiness follows the entry count, so `if redactor:` (or a None redactor)
    skips the pass for an empty config. redact() filters one text;
    redact_content() filters a whole tool result — a plain string, read_file's
    list of note blocks, or whatever other shape a tool hands back.
    """

    def __init__(self, strings: list[str], marker: str = REDACTED):
        self.marker = marker
        entries = sorted(set(strings), key=len, reverse=True)
        # One alternation of escaped literals, longest first. A single scan
        # (C speed, even for a 1 MB read) replaces every hit in place; Python's
        # regex picks the first alternative that matches at each position, so
        # ordering by length makes the most specific entry win, and re.sub
        # keeps scanning the original text only — a replacement is never
        # rescanned — so no entry can rewrite the marker itself.
        self.pattern = re.compile("|".join(re.escape(entry) for entry in entries)) if entries else None
        self.count = len(entries)

    def __len__(self) -> int:
        return self.count

    def redact(self, text: str) -> str:
        if self.pattern is None:
            return text
        # A lambda keeps the replacement literal; a plain replacement string
        # would have its backslashes treated as escapes.
        return self.pattern.sub(lambda _match: self.marker, text)

    def redact_content(self, result: _T) -> _T:
        """Redact every string in one tool result, whatever shape it has.

        Tool results reach the model as a plain string, read_file's list of
        {'type': 'text', 'text': ...} blocks, or whatever a future tool or API
        change returns, so this makes no assumptions about the shape: it walks
        the whole structure and filters every string leaf wherever it sits —
        list and tuple items, and dict values under any key. Containers are
        rebuilt with the same shape, so the API sees exactly what it would
        have seen unredacted. Dict keys are left alone: they are structural
        labels like "type" and "text", and rewriting one would mangle the
        payload rather than hide content. Leaves that are not strings (ints,
        None, ...) cannot carry text; bytes could, but they cannot be
        serialized into a message either, so both pass through unchanged
        rather than being decoded and possibly damaged.
        """
        if isinstance(result, str):
            return self.redact(result)
        if isinstance(result, list):
            return [self.redact_content(item) for item in result]
        if isinstance(result, tuple):
            return tuple(self.redact_content(item) for item in result)
        if isinstance(result, dict):
            return {key: self.redact_content(value) for key, value in result.items()}
        return result
