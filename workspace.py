"""Sandboxed file access: every path is confined to one project directory.

Also handles text encodings (UTF-8, or a fallback such as cp1252) and line
endings, so files are written back in the format they were read in. Every
write (overwrite, edit, append) is confirmed with the user, showing a coloured
diff of just the areas that changed.
"""

import codecs
from pathlib import Path

import style
from diff import render_diff

MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000
MAX_LIST_ENTRIES = 500


class ToolError(Exception):
    """An error reported back to Claude as a failed tool result."""


class Workspace:
    """Confines all file access to one root directory."""

    def __init__(
        self,
        root: Path,
        auto_approve_writes: bool = False,
        fallback_encoding: str = "cp1252",
        blocked: tuple[Path, ...] = (),
    ):
        self.root = root.resolve()
        self.auto_approve_writes = auto_approve_writes
        self.fallback_encoding = fallback_encoding
        # Files Claude may never read or write, even inside the workspace (e.g. the key file).
        self.blocked = {p.resolve() for p in blocked}
        # Encoding and line ending of each file Claude has read or written,
        # so writes can be saved back in the same format.
        self.formats: dict[Path, tuple[str, str]] = {}

    def resolve(self, rel_path: str) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ToolError("path must be a non-empty string")
        if "\x00" in rel_path:
            raise ToolError("path contains a null byte")
        # resolve() follows symlinks and collapses "..", so the containment
        # check below sees where the path really points.
        target = (self.root / rel_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"access denied: '{rel_path}' is outside the project directory")
        if target in self.blocked:
            raise ToolError(f"access denied: '{rel_path}' is a protected file")
        return target

    def display(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

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

    def read_file(self, path: str) -> str | list:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(f"file is too large to read ({size} bytes, limit {MAX_READ_BYTES})")
        data = target.read_bytes()
        # The fallback encodings accept any bytes, so binary files must be
        # caught here rather than by a decode error.
        if b"\x00" in data:
            raise ToolError(f"{path} appears to be a binary file and cannot be read")
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
        user confirmation, then the write.

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
        if not self.confirm(f"Model wants to {action} {self.display(target)} ({details}).", diff_lines):
            raise ToolError("the user declined this write")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.formats[target] = (encoding, newline)
        print(f"  {style.success('✓')} {verb} {self.display(target)} ({details})")
        return f"wrote {len(data)} bytes to {self.display(target)}, relative to the project root ({encoding})"

    def confirm(self, message: str, diff_lines: list[str]) -> bool:
        """Ask before writing, showing a coloured diff of the changed areas.

        diff_lines is the pre-rendered output of diff.render_diff; an empty
        list means nothing changes at line level (e.g. a new empty file).
        """
        if self.auto_approve_writes:
            print(f"  {style.dim('[auto-approved]')} {message}")
            return True
        print(f"\n  {style.paint(message, style.BOLD, style.YELLOW)}")
        for line in diff_lines:
            print(f"  {style.dim('|')} {line}")
        try:
            answer = input(style.prompt("  Allow this write? [y/N] ", style.BOLD)).strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")