"""Redaction of the strings a developer lists in a config file.

Every text sent to the model — user messages, /paste buffers, and every tool
result (file contents, search matches, directory listings, error strings) —
passes through this filter, applied in core/agent.py at the two places content
joins the conversation. Each configured string is replaced by the literal
"<redacted>", so a secret appearing anywhere in the harness's inputs reaches
neither the API nor the stored conversation history.

The strings live in redact.txt next to loom.py (loom.py's --redact-file moves
it), outside the workspace: the model can neither read nor edit its own
redaction list — the file is on the Workspace's blocked list as well, in case
the harness's own directory is ever the workspace. It is read once at startup,
so edits take effect on the next launch.

Matching is literal (no regular expressions), case-sensitive, and unanchored,
in one scan of each text with the entries longest first — so where one entry
contains another the longer is redacted whole, and because re.sub never
rescans the text it has replaced, no entry can ever rewrite the marker.
"""

import re
from pathlib import Path

# The replacement text for every configured string.
REDACTED = "<redacted>"


def load_redactions(path: Path) -> tuple[list[str], str | None]:
    """Read the redaction config, returning (strings, error note).

    One literal string per line; blank lines and lines starting with '#' are
    ignored, and surrounding whitespace is trimmed so an entry can never be
    empty. A missing file is the normal opt-out (redaction is opt-in) and
    returns no note, so loom.py stays quiet about it; a file that exists but
    cannot be read returns an empty list plus the note, which loom.py prints
    as a warning that text goes unredacted.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], None
    except (OSError, UnicodeDecodeError) as e:
        return [], f"could not read {path}: {e}"
    strings = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return strings, None


class Redactor:
    """The filter agent.py applies to everything that enters a conversation.

    Truthiness follows the entry count, so `if redactor:` (or a None redactor)
    skips the pass for an empty config. redact() filters one text;
    redact_content() filters a tool result, which is either a plain string or a
    list of the note blocks read_file returns.
    """

    def __init__(self, strings: list[str]):
        entries = sorted(set(strings), key=len, reverse=True)
        # One alternation of escaped literals, longest first. A single scan
        # (C speed, even for a 1 MB read) replaces every hit in place; Python's
        # regex picks the first alternative that matches at each position, so
        # ordering by length makes the most specific entry win, and re.sub
        # keeps scanning the original text only — a replacement is never
        # rescanned — so no entry can rewrite the "<redacted>" marker itself.
        self.pattern = re.compile("|".join(re.escape(entry) for entry in entries)) if entries else None
        self.count = len(entries)

    def __len__(self) -> int:
        return self.count

    def redact(self, text: str) -> str:
        if self.pattern is None:
            return text
        # A lambda keeps the replacement literal; a plain replacement string
        # would have its backslashes treated as escapes.
        return self.pattern.sub(lambda _match: REDACTED, text)

    def redact_content(self, result: str | list) -> str | list:
        """Redact one tool result: a plain string, or a list of text blocks.

        read_file returns a string for a small UTF-8 page and a list of
        {'type': 'text', 'text': ...} blocks when it prefixes notes (encoding,
        page cuts); write tools answer with strings. Only each block's text is
        replaced and every other key is copied unchanged, so the API keeps
        seeing the same shape. Anything else passes through as-is.
        """
        if isinstance(result, str):
            return self.redact(result)
        if isinstance(result, list):
            return [
                {**block, "text": self.redact(block["text"])}
                if isinstance(block, dict) and isinstance(block.get("text"), str)
                else block
                for block in result
            ]
        return result
