"""Sandboxed file access: every path is confined to one project directory.

Also handles text encodings (UTF-8, or a fallback such as cp1252) and line
endings, so files are written back in the format they were read in. Every
write (overwrite, edit, append) is confirmed by an injected approver — in the
harness, a prompt showing a coloured diff of just the areas that changed — and
a write to a file that looks executable by convention (a build or test entry
file) carries an extra warning. Nothing
inside a protected directory such as .git — whose config and hooks are a
command-execution vector — can be read or written, and search never follows a
symlink, so nothing outside the workspace can be reached through the walk.
"""

import codecs
import os
import re
from collections.abc import Callable
from pathlib import Path

from core.redact import Redactor
from ui import style
from ui.diff import render_diff

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULT_CAP = 500     # hard ceiling for search's max_results parameter
MAX_MATCH_CHARS = 200           # matching-line text kept per search result
MAX_SEARCH_FILES = 5_000        # files scanned before search stops with a note
MAX_SEARCH_BYTES = 100_000_000  # bytes scanned before search stops with a note

# Writes to files that look executable by convention get an extra warning line in
# the confirmation prompt: the file may run when the user next builds or tests.
# A deliberately tiny tripwire, not an inventory of runnable files — anything the
# user runs is their own decision, made while reading the diff it points at.
EXEC_RISK_NAMES = (
    "Makefile",
    "conftest.py",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "requirements.txt",
    "Pipfile",
    "Cargo.toml",
)
EXEC_RISK_PARTS = ("__pycache__", "venv", ".venv", "node_modules")

# The one thing a write needs from outside the sandbox: the approver is asked
# before the write happens, given the message, the pre-rendered diff lines,
# and the exec-risk note (or None), and returns True to allow it. The Workspace
# never reads input itself; loom.py injects the interactive prompt (or --yes's
# printed note), and the tests inject whatever behavior they need.
Approver = Callable[[str, list[str], str | None], bool]

# Absolute-path tokens in error text that scrub_paths() rewrites relative to
# the workspace root: a Windows drive letter with its separator, or a leading
# separator — but never one embedded in a URL (every '/' of 'https://…' is
# preceded by ':' or '/'), nor a relative path such as './api/config.py' or
# 'core/redact.py', which do not start with a separator. A token must also
# carry a second separator, so bare slash-words in prose ('/talk', '/plan')
# stay as they are instead of being rewritten as '../talk'.
_ABS_PATH_RE = re.compile(r"""(?<![\w:/\\.-])(?:[A-Za-z]:[\\/]|[\\/])[^\s"'()<>,;]*[\\/][^\s"'()<>,;]*""")


class ToolError(Exception):
    """An error reported back to the model as a failed tool result."""


class Workspace:
    """Confines all file access to one root directory."""

    def __init__(
        self,
        root: Path,
        approver: Approver,
        fallback_encoding: str = "cp1252",
        blocked: tuple[Path, ...] = (),
        protected_parts: tuple[str, ...] = (".git",),
        exec_risk_names: tuple[str, ...] = EXEC_RISK_NAMES,
        exec_risk_parts: tuple[str, ...] = EXEC_RISK_PARTS,
        redactor: Redactor | None = None,
    ):
        self.root = root.resolve()
        # Asked before every write; True allows it (see the Approver note above).
        self.approver = approver
        self.fallback_encoding = fallback_encoding
        # The redaction filter, applied BEFORE any length cut so a configured
        # string straddling a truncation boundary is redacted whole rather than
        # leaking its visible fragment: search redacts each matching line before
        # its 200-char cap, and read_file's windowed read redacts the page
        # before its 1 MB cap (which can cut mid-line). agent.py redacts every
        # tool result a second time afterward — a harmless double pass.
        self.redactor = redactor
        # Files the model may never read or write, even inside the workspace (e.g. the key file).
        self.blocked = {p.resolve() for p in blocked}
        # Path components nothing may sit in or under, compared case-insensitively so a
        # Windows '.GIT' checkout is covered: .git's config and hooks are a
        # command-execution vector if written, so the whole directory is off limits.
        self.protected_parts = frozenset(part.lower() for part in protected_parts)
        # Filenames, and parent directory components, whose writes carry the extra
        # confirmation warning (see EXEC_RISK_NAMES / EXEC_RISK_PARTS above).
        self.exec_risk_names = frozenset(name.lower() for name in exec_risk_names)
        self.exec_risk_parts = frozenset(part.lower() for part in exec_risk_parts)
        # Encoding and line ending of each file the model has read or written,
        # so writes can be saved back in the same format.
        self.formats: dict[Path, tuple[str, str]] = {}

    def _denial(self, resolved: Path) -> str | None:
        """Why a resolved path may not be touched, or None when it may be.

        The one access rule, shared by resolve() (which turns the reason into a
        ToolError naming the path the model asked for) and search's directory
        walk (which silently skips what it refuses), so the two can never drift
        apart: the path must be the workspace root or under it, not one of the
        blocked files, and not inside a protected directory such as .git.
        """
        if resolved != self.root and self.root not in resolved.parents:
            return "outside the project directory"
        if resolved in self.blocked:
            return "a protected file"
        if any(part.lower() in self.protected_parts for part in resolved.relative_to(self.root).parts):
            return "inside a protected directory (git config and hooks are a command-execution vector)"
        return None

    def resolve(self, rel_path: str) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ToolError("path must be a non-empty string")
        if "\x00" in rel_path:
            raise ToolError("path contains a null byte")
        # resolve() follows symlinks and collapses "..", so the containment
        # check below sees where the path really points.
        target = (self.root / rel_path).resolve()
        reason = self._denial(target)
        if reason:
            raise ToolError(f"access denied: '{rel_path}' is {reason}")
        return target

    def display(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

    def scrub_paths(self, text: str) -> str:
        """Rewrite every absolute path in error text relative to the workspace.

        Raw OSError messages carry the resolved absolute path the filesystem
        call was given, and agent.py puts that text into the failed tool result
        — so without this the model would learn the developer's home directory,
        project location, and machine layout from every missing-file or
        permission error. Instead of redacting, which would leave the model no
        way to fix the input, each absolute path becomes its workspace-relative
        spelling: the root's own path turns into '.' (or './' plus the rest
        when it names something inside), and any other absolute path, which
        containment makes unreachable in normal operation, is rewritten with
        os.path.relpath as a '../…' path relative to this root too — with a
        console note, because a path from outside the workspace reaching the
        model deserves a look. The model keeps enough of every path to correct
        the input that caused the error; only the absolute spelling goes.

        ToolError messages need none of this: they name the path as the model
        asked for it, which is already relative, and pass through untouched.
        """
        rooted = str(self.root)
        # (?![\w.\-]) after the root keeps the rewrite off longer names that
        # merely start with the root's path — a workspace 'proj' sitting next
        # to a sibling 'proj.backup' must not become '.backup'; such a sibling
        # is genuinely outside the workspace, so the fallback below rewrites
        # it as a '../…' path instead. The captured trailing separator tells
        # a child from the bare root ('.' vs './' + the rest).
        spellings = [(rooted, os.sep)]
        if os.altsep and os.altsep != os.sep:
            spellings.append((rooted.replace(os.sep, os.altsep), os.altsep))
        for spelling, sep in spellings:
            text = re.sub(
                "(?:" + re.escape(spelling) + r")(?![\w.\-])(" + re.escape(sep) + r")?",
                lambda match: "./" if match.group(1) else ".",
                text,
            )
        named_outside = False

        def to_relative(match: re.Match[str]) -> str:
            nonlocal named_outside
            named_outside = True
            token = match.group()
            try:
                return os.path.relpath(token, rooted)
            except ValueError:
                # os.path.relpath refuses a path on a Windows drive other than
                # the root's: no relative spelling exists, so the token stays
                # and the console note below reports it.
                return token

        text = _ABS_PATH_RE.sub(to_relative, text)
        if named_outside:
            print(style.warning(
                "  [this error text named a path outside the workspace; it was rewritten relative to the root]"
            ))
        return text

    def list_directory(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:MAX_LIST_ENTRIES]]
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"... ({len(entries) - MAX_LIST_ENTRIES} more entries not shown)")
        return "\n".join(lines) if lines else "(empty directory)"

    def decode(self, data: bytes) -> tuple[str, str]:
        """Decode file bytes, returning (text, encoding).

        Valid UTF-8 is unambiguous, so it is tried first. Anything else is
        treated as the fallback encoding, with latin-1 as a last resort
        because it accepts every byte (cp1252 leaves five bytes undefined).
        """
        utf8 = "utf-8-sig" if data.startswith(codecs.BOM_UTF8) else "utf-8"
        for encoding in (utf8, self.fallback_encoding, "latin-1"):
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        raise ToolError("could not decode file")

    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> str | list:
        """Read a text file, optionally as a page of its lines.

        offset is the 1-based number of the first line to return and limit the
        maximum number of lines; both default to 0 (no window). A read never
        returns more than MAX_READ_BYTES of text: larger files (or pages) are
        cut at a line boundary where possible, followed by a note naming the
        line range shown and the offset to continue from, so files of any size
        can be read in slices. Line numbers here match search results. Every
        read records the file's encoding and line endings, so a paged read
        also counts as the read required before editing or overwriting.
        """
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ToolError("offset must be a non-negative integer (1-based first line to read)")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ToolError("limit must be a non-negative integer (maximum number of lines)")
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        data = target.read_bytes()
        # The fallback encodings accept any bytes, so binary files must be
        # caught here rather than by a decode error.
        if b"\x00" in data:
            raise ToolError(f"{path} appears to be a binary file and cannot be read")
        if not offset and not limit and len(data) <= MAX_READ_BYTES:
            return self._read_whole(target, path, data)
        return self._read_window(target, path, data, offset, limit)

    def _read_whole(self, target: Path, path: str, data: bytes) -> str | list:
        """Small read within the one-call cap: the decoded text, byte for byte."""
        text, encoding = self.decode(data)
        newline = "\r\n" if "\r\n" in text else "\n"
        self.formats[target] = (encoding, newline)
        text = text.replace("\r\n", "\n") or "(empty file)"
        if encoding in ("utf-8", "utf-8-sig"):
            return text
        note = (
            f"[Note: this file is encoded as {encoding}, not UTF-8. It will be saved back as "
            f"{encoding}, so only use characters that {encoding} can represent.]"
        )
        return [{"type": "text", "text": note}, {"type": "text", "text": text}]

    def _read_window(self, target: Path, path: str, data: bytes, offset: int, limit: int) -> str | list:
        """Line-windowed read for large files and explicit offset/limit reads.

        Lines are reported without their trailing newlines. The page is capped
        at MAX_READ_BYTES bytes: cut at the last line boundary that fits, or,
        when even one line is longer than that, mid-line (with a note saying
        to use search inside such lines). Notes carry the line range shown;
        line numbers here match search results.
        """
        text, encoding = self.decode(data)
        newline = "\r\n" if "\r\n" in text else "\n"
        self.formats[target] = (encoding, newline)
        notes = []
        if encoding not in ("utf-8", "utf-8-sig"):
            notes.append(
                f"[Note: this file is encoded as {encoding}, not UTF-8. It will be saved back as "
                f"{encoding}, so only use characters that {encoding} can represent.]"
            )
        rows = text.replace("\r\n", "\n").split("\n")
        if rows and rows[-1] == "":
            rows.pop()  # the trailing newline is layout, not a line
        if offset > len(rows):
            raise ToolError(
                f"offset {offset} is past the end of the file, which has {len(rows)} lines "
                "(offset is 1-based, so use 1 for the first line)"
            )
        first = offset or 1  # 1-based number of the first line shown
        chosen = rows[first - 1 :]
        if limit:
            chosen = chosen[:limit]
        page = "\n".join(chosen)
        if self.redactor:
            # Redact before the cap cut below: when one line is longer than the
            # cap the cut is mid-line, and a configured string straddling that
            # cut would otherwise leak its visible fragment (a cut "<reda"
            # reveals nothing). Line numbers stay exact because the marker
            # contains no newline.
            page = self.redactor.redact(page)
        encoded = page.encode(encoding)
        cut = len(encoded) > MAX_READ_BYTES
        if cut:
            head = encoded[:MAX_READ_BYTES]
            end = head.rfind(b"\n")
            if end != -1:
                head = head[:end]
            page = head.decode(encoding, errors="ignore")
        shown_to = first + page.count("\n")
        if cut:
            if end == -1:
                notes.append(
                    f"[Note: line {shown_to} is longer than the {MAX_READ_BYTES:,}-byte page cap, so "
                    "only its first part is shown (use search to find text inside very long lines)"
                )
            else:
                notes.append(f"[Note: the page was cut at {MAX_READ_BYTES:,} bytes]")
        if shown_to < len(rows):
            notes.append(
                f"[Note: showing lines {first}-{shown_to} of {len(rows)} of {path}; continue with "
                f"read_file('{path}', offset={shown_to + 1}" + (f", limit={limit}" if limit else "") + ")]"
            )
        if notes:
            blocks = [{"type": "text", "text": note} for note in notes]
            blocks.append({"type": "text", "text": page or "(no text on these lines)"})
            return blocks
        return page or "(empty file)"

    def _walkable(self, entry: Path) -> bool:
        """Whether search may read one entry from its directory walk.

        Symlinks are never followed, so a link pointing outside the workspace
        cannot be read through the walk (resolve() already refuses the same
        link when it is named directly), and every entry is re-checked with the
        same rule as resolve() — containment, blocked files, protected
        directories — since the walk finds paths itself rather than being
        handed them. The protected-parts filter repeats _denial's check on
        purpose: belt and braces for the one directory whose contents must
        never be searched. Refused entries are skipped silently; binary and
        unreadable files are the only ones a note mentions, below.
        """
        if entry.is_symlink() or not entry.is_file():
            return False
        if any(part.lower() in self.protected_parts for part in entry.relative_to(self.root).parts):
            return False
        return self._denial(entry.resolve()) is None

    def search(self, pattern: str, path: str = ".", ignore_case: bool = False, max_results: int = 50) -> str:
        """Regex-search file contents without reading files into the conversation.

        Scans files under path (default: the whole project; a single-file path
        searches just that file) in sorted order, skips binary files (by a
        null-byte sniff) and symlinks (a link is never followed, so nothing
        outside the workspace can be read through the walk), and reports every
        matching line as 'relpath:line: first 200 characters' with 1-based
        line numbers. Stops at max_results (at most 500), or after
        MAX_SEARCH_FILES files or MAX_SEARCH_BYTES bytes, each with a closing
        note so the caller can narrow the search.
        Reading this way does not count as the read_file required before
        editing a file.
        """
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern must be a non-empty string")
        if not isinstance(ignore_case, bool):
            raise ToolError("ignore_case must be boolean")
        if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results < 1:
            raise ToolError("max_results must be a positive integer")
        result_cap = min(max_results, MAX_SEARCH_RESULT_CAP)
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"invalid pattern: {e}")
        target = self.resolve(path)
        if target.is_file():
            files = [target]
        elif target.is_dir():
            # Every walked entry passes _walkable: the same rule resolve()
            # applies (containment, blocked files, .git), plus never
            # following symlinks. Binary files are skipped by content below.
            files = sorted(entry for entry in target.rglob("*") if self._walkable(entry))
        else:
            raise ToolError(f"not a file or directory: {path}")
        matches: list[str] = []
        notes: list[str] = []
        skipped = 0
        scanned_files = 0
        scanned_bytes = 0
        for file in files:
            if len(matches) >= result_cap:
                break
            if scanned_files >= MAX_SEARCH_FILES or scanned_bytes >= MAX_SEARCH_BYTES:
                notes.append(
                    f"(search stopped after {scanned_files} files and {scanned_bytes:,} bytes; "
                    "narrow the path to search the rest)"
                )
                break
            try:
                data = file.read_bytes()
            except OSError:
                skipped += 1
                continue
            if b"\x00" in data[:8192]:
                skipped += 1
                continue
            scanned_files += 1
            scanned_bytes += len(data)
            text, _encoding = self.decode(data)
            relative = file.relative_to(self.root).as_posix()
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    if self.redactor:
                        # Redact the full line before the cap cut below: a
                        # configured string straddling the truncation boundary
                        # would otherwise leak its visible fragment (a cut
                        # "<reda" reveals nothing). The match above stays on the
                        # raw line, so redaction never changes which lines match.
                        line = self.redactor.redact(line)
                    cut = line[:MAX_MATCH_CHARS] + ("..." if len(line) > MAX_MATCH_CHARS else "")
                    matches.append(f"{relative}:{line_no}: {cut}")
                    if len(matches) >= result_cap:
                        break
        if len(matches) >= result_cap:
            notes.append(
                f"(stopped at the {result_cap}-match cap; more matches may exist, so narrow "
                "the pattern or the path)"
            )
        if skipped:
            notes.append(f"(skipped {skipped} binary or unreadable file{'s' if skipped != 1 else ''})")
        lines = matches if matches else ["(no matches)"]
        return "\n".join(lines + notes)

    def write_file(self, path: str, content: str) -> str:
        """Create a file, or overwrite one with the complete new contents.
        An existing file must have been read first."""
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if "\x00" in content:
            raise ToolError("content contains a null byte; only text files can be written")
        target = self.resolve(path)
        if target == self.root or target.is_dir():
            raise ToolError(f"path is a directory: {path}")
        new_text = content.replace("\r\n", "\n")
        if target.exists():
            if target not in self.formats:
                raise ToolError(f"read {path} before overwriting it, so its encoding and line endings are kept")
            encoding, newline = self.formats[target]
            old_text = self._current_text(target, path)
            if old_text == new_text:
                raise ToolError("nothing to change: the file on disk already has exactly this content")
            return self._finish_write(target, path, old_text, new_text, encoding, newline, "overwrite", "updated")
        return self._finish_write(target, path, "", new_text, "utf-8", "\n", "create", "created")

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace one exact snippet in an existing file with new text.

        Only the snippet travels in the tool call, so an edit costs a fraction
        of the output tokens that rewriting the whole file with write_file
        would. old_text must appear exactly once, or the edit is refused as
        ambiguous; new_text may be empty, deleting the snippet. The file must
        have been read first, and is written back in its original encoding and
        line endings.
        """
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolError("old_text and new_text must be strings")
        if not old_text:
            raise ToolError(
                "old_text is empty: edit_file replaces an existing snippet. Use write_file to "
                "create or replace a file, or append_file to add to its end"
            )
        if "\x00" in old_text or "\x00" in new_text:
            raise ToolError("edit text contains a null byte; only text files can be edited")
        target = self.resolve(path)
        if target == self.root or target.is_dir():
            raise ToolError(f"path is a directory: {path}")
        if not target.exists():
            raise ToolError(f"{path} does not exist; use write_file to create it")
        if target not in self.formats:
            raise ToolError(f"read {path} before editing it, so its encoding and line endings are kept")
        encoding, newline = self.formats[target]
        current = self._current_text(target, path)
        old_text = old_text.replace("\r\n", "\n")
        new_text = new_text.replace("\r\n", "\n")
        count = current.count(old_text)
        if count == 0:
            raise ToolError(
                f"old_text does not appear in {path}; copy it exactly from a fresh read_file "
                "of the file, including its indentation"
            )
        if count > 1:
            raise ToolError(
                f"old_text appears {count} times in {path}; include more of the surrounding "
                "lines so it appears exactly once"
            )
        if old_text == new_text:
            raise ToolError("nothing to change: old_text and new_text are identical")
        result = current.replace(old_text, new_text, 1)
        return self._finish_write(target, path, current, result, encoding, newline, "edit", "edited")

    def append_file(self, path: str, content: str) -> str:
        """Add text at the end of a file, creating the file if it does not exist.

        Appending to an existing file requires reading it first, and it keeps
        its encoding and line endings; if it does not end with a newline, one
        is inserted first so the appended text never joins its last line. A
        long file can be built in chunks across several calls, each fitting in
        one model response.
        """
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if "\x00" in content:
            raise ToolError("content contains a null byte; only text files can be written")
        target = self.resolve(path)
        if target == self.root or target.is_dir():
            raise ToolError(f"path is a directory: {path}")
        content = content.replace("\r\n", "\n")
        if target.exists():
            if target not in self.formats:
                raise ToolError(f"read {path} before appending to it, so its encoding and line endings are kept")
            encoding, newline = self.formats[target]
            current = self._current_text(target, path)
            if current and not current.endswith("\n"):
                content = "\n" + content
            if not content:
                raise ToolError("nothing to change: the content to append is empty")
            return self._finish_write(target, path, current, current + content, encoding, newline, "append to", "appended to")
        return self._finish_write(target, path, "", content, "utf-8", "\n", "create", "created")

    def _current_text(self, target: Path, path: str) -> str:
        """The file's text on disk right now, normalised to '\\n' line endings.

        Decoded with the encoding recorded when it was last read; if it changed
        on disk since and no longer decodes that way, falls back to the usual
        detection. The diff shown for confirmation always compares against
        what is on disk, including any edits made since the model read the file.
        """
        current = target.read_bytes()
        if b"\x00" in current:
            raise ToolError(f"{path} now looks like a binary file on disk; read it again before writing")
        try:
            text = current.decode(self.formats[target][0])
        except UnicodeDecodeError:
            text = self.decode(current)[0]
        return text.replace("\r\n", "\n")

    def exec_risk(self, target: Path) -> str | None:
        """The confirmation warning for a write target that may be executed by
        the user later, or None for an ordinary file.

        Matches by filename (Makefile, conftest.py, pyproject.toml, ...) or by
        any parent directory component (__pycache__, venv, .venv, node_modules),
        case-insensitively. The list is deliberately tiny: a tripwire to make
        the one human control — reading the diff — harder to sleep through, not
        an inventory of every file a build could run.
        """
        parts = [part.lower() for part in target.relative_to(self.root).parts]
        if (parts and parts[-1] in self.exec_risk_names) or any(
            part in self.exec_risk_parts for part in parts[:-1]
        ):
            return "note: this file may be executed when you run builds or tests — review the diff carefully"
        return None

    def _finish_write(
        self,
        target: Path,
        path: str,
        old_text: str,
        new_text: str,
        encoding: str,
        newline: str,
        action: str,
        verb: str,
    ) -> str:
        """Shared tail of the three write tools: size check, coloured diff,
        approval, then the write.

        old_text and new_text are already normalised to '\n'; new_text is
        saved back in the file's newline style and encoding. action names the
        change in the confirmation question ('Model wants to edit ...') and
        verb in the success line ('✓ edited ...').
        """
        text = new_text.replace("\n", newline)
        try:
            data = text.encode(encoding)
        except UnicodeEncodeError as e:
            bad = e.object[e.start]
            raise ToolError(
                f"{path} is encoded as {encoding}, which cannot represent {bad!r} (U+{ord(bad):04X}). "
                f"Rewrite the content without that character."
            )
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"content is too large ({len(data)} bytes, limit {MAX_WRITE_BYTES})")
        diff_lines, additions, removals = render_diff(old_text, new_text)
        if additions or removals:
            changes = f"+{additions}" + (f" -{removals}" if removals else "") + " lines"
        else:
            changes = "no line changes"
        details = f"{changes}, {len(data)} bytes, {encoding}" + (", CRLF" if newline == "\r\n" else "")
        if not self.approver(
            f"Model wants to {action} {self.display(target)} ({details}).", diff_lines, self.exec_risk(target)
        ):
            raise ToolError("the user declined this write")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.formats[target] = (encoding, newline)
        print(f"  {style.success('✓')} {verb} {self.display(target)} ({details})")
        return f"wrote {len(data)} bytes to {self.display(target)}, relative to the project root ({encoding})"