"""Coloured unified diffs for the write-permission prompt.

Like Claude Code, render_diff shows only the areas that changed: each change
plus CONTEXT unchanged lines around it, with the unchanged regions between
changes collapsed behind '@@' hunk headers. Removed lines are red, added lines
green, context lines dim, and hunk headers cyan.

New files are diffed against empty text, so the whole file shows as added.
Very large diffs are truncated for display only (a notice reports how many
lines were cut); the write itself replaces the whole file, as the model asked.
"""

import difflib

import style

CONTEXT = 3
MAX_DIFF_LINES = 100
MAX_LINE_CHARS = 400


def paint_line(line: str) -> str:
    """Colour one diff line by its marker, trimming very long lines."""
    if len(line) > MAX_LINE_CHARS:
        line = line[:MAX_LINE_CHARS] + f"... ({len(line) - MAX_LINE_CHARS} more chars)"
    if line.startswith("@@"):
        return style.paint(line, style.CYAN)
    if line.startswith("+"):
        return style.success(line)
    if line.startswith("-"):
        return style.error(line)
    return style.dim(line)


def render_diff(old_text: str, new_text: str) -> tuple[list[str], int, int]:
    """Return the coloured diff lines between two texts, plus line counts.

    Returns (lines, additions, removals). The counts cover the whole diff even
    when the printed lines are truncated to MAX_DIFF_LINES. Identical texts
    give ([], 0, 0).
    """
    raw = list(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm="", n=CONTEXT))
    # difflib starts with a "--- " / "+++ " header pair; the path is shown in
    # the confirm message above the diff instead. Skip it by position, as
    # content lines can legitimately start with "---" (e.g. a YAML separator).
    if raw and raw[0].startswith("--- "):
        raw = raw[2:]
    additions = removals = 0
    for line in raw:
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            removals += 1
    lines = [paint_line(line) for line in raw[:MAX_DIFF_LINES]]
    if len(raw) > MAX_DIFF_LINES:
        lines.append(style.dim(f"... ({len(raw) - MAX_DIFF_LINES} more diff lines)"))
    return lines, additions, removals