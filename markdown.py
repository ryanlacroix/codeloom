"""Renders the model's Markdown as styled terminal text while it streams in.

Text is written as it arrives. Only the start of each line is held back until
its kind is known (heading, list item, quote, code fence, rule, table row), and
within a line only the few characters needed to tell what a marker means, or
an unfinished link. Table rows are held until the table ends, so the columns
can be aligned. With colors off, the text is written exactly as received.
"""

import re
import shutil
import string

import style

ANSI = re.compile(r"\033\[[0-9;]*m")
FENCES = ("```", "~~~")
RULE = re.compile(r"([-*_])( *\1){2,} *")
TABLE_SEPARATOR = re.compile(r":?-+:?")
CODE = "38;5;216"  # light orange, for inline code and code blocks
STYLES = {"bold": "1", "dim": "2", "italic": "3", "strike": "9", "code": CODE, "link": "4;34"}
HEADINGS = {1: ["1", "4"], 2: ["1"]}  # h3 and below: bold italic
BULLETS = ("•", "◦", "▪")


def visible_len(text: str) -> int:
    return len(ANSI.sub("", text))


class Inline:
    """Styles the inline Markdown of one line: bold, italic, strikethrough, code and links."""

    def __init__(self, write, base: list[str], literal: bool = False):
        self.write = write
        self.base = base  # style codes for the whole line, e.g. a heading's
        self.literal = literal  # code block lines are written without styling markers
        self.active = []  # names of the inline styles currently on
        self.buf = ""  # text held back until its meaning is known
        self.prev = ""  # last character written, to tell opening markers from closing ones

    def restyle(self) -> None:
        codes = self.base + [STYLES[name] for name in self.active]
        self.write("\033[0m" + (f"\033[{';'.join(codes)}m" if codes else ""))

    def emit(self, text: str) -> None:
        if text:
            self.write(text)
            self.prev = text[-1]

    def feed(self, text: str, final: bool = False) -> None:
        if self.literal:
            self.emit(text)
            return
        buf, i = self.buf + text, 0
        while i < len(buf):
            c = buf[i]
            if "code" in self.active:
                end = buf.find("`", i)
                if end == i:
                    self.active.remove("code")
                    self.restyle()
                    i += 1
                else:
                    end = len(buf) if end < 0 else end
                    self.emit(buf[i:end])
                    i = end
            elif c == "\\":
                if i + 1 == len(buf) and not final:
                    break
                if buf[i + 1 : i + 2] and buf[i + 1] in string.punctuation:
                    self.emit(buf[i + 1])
                    i += 2
                else:
                    self.emit(c)
                    i += 1
            elif c == "`":
                self.active.append("code")
                self.restyle()
                i += 1
            elif c in "*_~":
                end = i
                while end < len(buf) and buf[end] == c:
                    end += 1
                if end == len(buf) and not final:
                    break  # the run may continue, and the next character decides its meaning
                if not self.emphasis(c, end - i, buf[end] if end < len(buf) else ""):
                    self.emit(buf[i:end])
                i = end
            elif c == "[":
                end = self.link(buf, i, final)
                if end is None:
                    break
                i = end
            else:
                end = i + 1
                while end < len(buf) and buf[end] not in "\\`*_~[":
                    end += 1
                self.emit(buf[i:end])
                i = end
        self.buf = buf[i:]

    def emphasis(self, marker: str, count: int, after: str) -> bool:
        """Turn a style on or off for a run of *, _ or ~. Returns False if the run is plain text."""
        if marker == "~":
            names = ["strike"] if count == 2 else None
        else:
            names = {1: ["italic"], 2: ["bold"], 3: ["bold", "italic"]}.get(count)
        if not names:
            return False

        # Like CommonMark, a marker opens before a word and closes after one; snake_case
        # and "2 * 3" stay as they are.
        def outside_word(ch: str) -> bool:
            return not ch.isalnum()

        opening = after != "" and not after.isspace() and outside_word(self.prev)
        closing = self.prev != "" and not self.prev.isspace() and outside_word(after)
        if closing and all(name in self.active for name in names):
            for name in names:
                self.active.remove(name)
        elif opening and not any(name in self.active for name in names):
            self.active.extend(names)
        else:
            return False
        self.restyle()
        return True

    def link(self, buf: str, start: int, final: bool) -> int | None:
        """Write [text](url) as underlined text and a dim url. Returns where to continue,
        or None to wait for more text."""
        close = buf.find("]", start + 1)
        if close >= 0 and close + 1 < len(buf) and buf[close + 1] != "(":
            self.emit("[")  # not a link, e.g. a[0]
            return start + 1
        end = buf.find(")", close + 2) if close >= 0 else -1
        if end < 0:
            if not final:
                return None
            self.emit("[")
            return start + 1
        text, url = buf[start + 1 : close], buf[close + 2 : end]
        self.active.append("link")
        self.restyle()
        self.emit(text)
        self.active.remove("link")
        if url and url != text:
            self.active.append("dim")
            self.restyle()
            self.emit(f" ({url})")
            self.active.remove("dim")
        self.restyle()
        return end + 1

    def start(self) -> None:
        if self.base:
            self.restyle()

    def end_line(self) -> None:
        self.feed("", final=True)
        if self.base or self.active:
            self.write("\033[0m")
        self.active = []
        self.prev = ""


def render_inline(text: str, base: list[str]) -> str:
    parts = []
    inline = Inline(parts.append, base)
    inline.start()
    inline.feed(text, final=True)
    inline.end_line()
    return "".join(parts)


class MarkdownStream:
    """Renders streamed Markdown text, line by line, as it arrives."""

    def __init__(self, write, base: list[str] | None = None):
        self.raw_write = write
        self.base = base or []  # style codes for all the text, e.g. the thinking color
        self.line = ""  # start of the current line, held until its kind is known
        self.inline = None  # set while the current line is being written
        self.in_fence = False
        self.table = []  # rows of the table being collected
        self.at_line_start = False

    def write(self, text: str) -> None:
        if text:
            self.raw_write(text)
            self.at_line_start = text.endswith("\n")

    def start_own_line(self) -> None:
        """Move to a new line before a block that must start one, e.g. a table
        that is the first thing after the model's label."""
        if not self.at_line_start:
            self.write("\n")

    def feed(self, text: str) -> None:
        if not style.ENABLED:
            self.write(text)
            return
        while text:
            part, newline, text = text.partition("\n")
            if self.inline is not None:
                self.inline.feed(part)
            else:
                self.line += part
                self.classify(complete=False)
            if newline:
                self.end_line()

    def end_line(self) -> None:
        if self.inline is None and self.classify(complete=True) == "done":
            return
        self.inline.end_line()
        self.inline = None
        self.write("\n")

    def finish(self) -> None:
        """Write out whatever is held back, at the end of the text."""
        if self.inline is None and self.line:
            self.classify(complete=True)
        if self.inline is not None:
            self.inline.end_line()
            self.inline = None
        if self.table:
            self.flush_table()
        self.in_fence = False

    def kind(self, s: str, complete: bool) -> tuple[str, int] | None:
        """The kind of a line (with leading spaces removed) and the length of its marker,
        or None if more of the line is needed to tell."""
        if self.in_fence:
            if s.startswith(FENCES):
                return ("fence", 0) if complete else None
            if not complete and len(s) < 3 and any(f.startswith(s) for f in FENCES):
                return None
            return ("code", 0)
        if not s:
            return ("blank", 0) if complete else None
        if s.startswith(FENCES):
            return ("fence", 0) if complete else None
        if s.startswith("|"):
            return ("table", 0) if complete else None
        if not complete and (set(s) <= set("-*_ ") or (len(s) < 3 and any(f.startswith(s) for f in FENCES))):
            return None
        if complete and RULE.fullmatch(s):
            return ("rule", 0)
        if m := re.match(r"(#{1,6}) ", s):
            return (f"h{len(m.group(1))}", m.end())
        if m := re.match(r"[-*+] ", s):
            return ("bullet", m.end())
        if m := re.match(r"\d{1,9}[.)] ", s):
            return ("number", m.end())
        if s.startswith(">"):
            if s == ">" and not complete:
                return None
            return ("quote", 2 if s.startswith("> ") else 1)
        if not complete and re.fullmatch(r"#{1,6}|\d{1,9}[.)]?", s):
            return None
        return ("text", 0)

    def classify(self, complete: bool) -> str:
        """Decide what the held line is, once possible. Returns "pending" (need more
        text), "streaming" (the rest of the line is written as it arrives) or "done"
        (the whole line has been handled)."""
        line = self.line
        s = line.lstrip(" ")
        indent = len(line) - len(s)
        kind = self.kind(s, complete)
        if kind is None:
            return "pending"
        name, marker = kind
        self.line = ""
        if name == "table":
            self.table.append(s)
            return "done"
        if self.table:
            self.flush_table()
        if name == "fence":
            self.in_fence = not self.in_fence
            self.start_own_line()
            self.write(style.paint(line, *self.base, STYLES["dim"]) + "\n")
            return "done"
        if name == "blank":
            self.write("\n")
            return "done"
        if name == "rule":
            width = min(shutil.get_terminal_size().columns, 80)
            self.start_own_line()
            self.write(style.paint("─" * width, *self.base, STYLES["dim"]) + "\n")
            return "done"

        base, prefix, rest = list(self.base), " " * indent, s[marker:]
        if name == "code":
            base, prefix, rest = [CODE], "", line
        elif name.startswith("h"):
            base += HEADINGS.get(int(name[1]), ["1", "3"])
        elif name == "bullet":
            prefix += style.paint(BULLETS[indent // 2 % len(BULLETS)], *self.base) + " "
        elif name == "number":
            prefix += style.paint(s[:marker], *self.base)
        elif name == "quote":
            prefix += style.paint("│", *self.base, STYLES["dim"]) + " "
            base += [STYLES["italic"]]
        self.write(prefix)
        self.inline = Inline(self.write, base, literal=name == "code")
        self.inline.start()
        self.inline.feed(rest)
        return "streaming"

    def flush_table(self) -> None:
        rows = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in self.table]
        self.table = []
        has_header = len(rows) > 1 and all(TABLE_SEPARATOR.fullmatch(cell) for cell in rows[1])
        if has_header:
            del rows[1]
        columns = max(len(row) for row in rows)
        cells = [
            [render_inline(cell, self.base + (["1"] if has_header and r == 0 else [])) for cell in row]
            + [""] * (columns - len(row))
            for r, row in enumerate(rows)
        ]
        widths = [max(visible_len(row[c]) for row in cells) for c in range(columns)]

        def border(left: str, middle: str, right: str) -> str:
            return style.paint(left + middle.join("─" * (w + 2) for w in widths) + right, STYLES["dim"]) + "\n"

        bar = style.paint("│", STYLES["dim"])
        self.start_own_line()
        self.write(border("┌", "┬", "┐"))
        for r, row in enumerate(cells):
            padded = (cell + " " * (widths[c] - visible_len(cell)) for c, cell in enumerate(row))
            self.write(f"{bar} " + f" {bar} ".join(padded) + f" {bar}\n")
            if has_header and r == 0:
                self.write(border("├", "┼", "┤"))
        self.write(border("└", "┴", "┘"))
