"""Sandboxed file access: every path is confined to one project directory.

Also handles text encodings (UTF-8, or a fallback such as cp1252) and line
endings, so files are written back in the format they were read in.
"""

import codecs
from pathlib import Path

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
        # so overwrites can be saved back in the same format.
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
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if "\x00" in content:
            raise ToolError("content contains a null byte; only text files can be written")
        target = self.resolve(path)
        if target == self.root or target.is_dir():
            raise ToolError(f"path is a directory: {path}")
        if target.exists():
            if target not in self.formats:
                raise ToolError(f"read {path} before overwriting it, so its encoding and line endings are kept")
            encoding, newline = self.formats[target]
            action = "overwrite"
        else:
            encoding, newline = "utf-8", "\n"
            action = "create"
        text = content.replace("\r\n", "\n").replace("\n", newline)
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
        details = f"{len(data)} bytes, {encoding}" + (", CRLF" if newline == "\r\n" else "")
        if not self.confirm(f"Model wants to {action} {self.display(target)} ({details}).", content):
            raise ToolError("the user declined this write")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.formats[target] = (encoding, newline)
        return f"wrote {len(data)} bytes to {self.display(target)}, relative to the project root ({encoding})"

    def confirm(self, message: str, content: str) -> bool:
        if self.auto_approve_writes:
            print(f"  [auto-approved] {message}")
            return True
        print(f"\n  {message}")
        preview = content.splitlines()
        for line in preview[:20]:
            print(f"  | {line}")
        if len(preview) > 20:
            print(f"  | ... ({len(preview) - 20} more lines)")
        try:
            answer = input("  Allow this write? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")
