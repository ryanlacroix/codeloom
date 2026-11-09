"""Prints the model's thinking and reply as they stream in, under their labels."""

import shutil
import sys

from ui import style
from ui.markdown import MarkdownStream


class StreamPrinter:
    """Prints thinking (light blue) and reply text as they stream in, each under its own
    label, rendering their Markdown. When the thinking block ends and the reply is about
    to start, a rule is drawn across the screen to divide them."""

    def __init__(self, label: str, show_thinking: bool):
        self.label = label
        self.show_thinking = show_thinking
        self.current = None  # "thinking", "text", or None between blocks
        self.last = None  # kind of the block that closed most recently
        self.markdown = None  # renderer for the open block

    def write(self, text: str) -> None:
        # The one raw write path: everything streamed (reply and thinking text,
        # routed here through the Markdown renderer) is sanitized, so control
        # characters in model output cannot reach the terminal.
        sys.stdout.write(style.sanitize(text))
        sys.stdout.flush()

    def start(self, kind: str, text: str) -> str:
        """Open a labelled block if one of this kind isn't already open, and return
        the text to print (without leading whitespace when the block is new)."""
        if self.current == kind:
            return text
        # Ollama switches from thinking to content mid-stream rather than closing
        # the thinking block first, so check both here and in end_block.
        divider = kind == "text" and (self.current == "thinking" or self.last == "thinking")
        self.end_block()
        header = style.dim("thinking>") if kind == "thinking" else style.model_label(self.label + ">")
        if divider:
            # The rule takes the place of the usual blank line after the thinking.
            self.rule()
            self.write(f"{header} ")
        else:
            self.write(f"\n{header} ")
        self.current = kind
        self.markdown = MarkdownStream(self.write, [style.LIGHT_BLUE] if kind == "thinking" else [])
        return text.lstrip()

    def thinking(self, text: str) -> None:
        if self.show_thinking and (text.strip() or self.current == "thinking"):
            text = self.start("thinking", text)
            self.markdown.feed(text)

    def text(self, text: str) -> None:
        if text.strip() or self.current == "text":
            text = self.start("text", text)
            self.markdown.feed(text)

    def end_block(self) -> None:
        if self.current:
            self.markdown.finish()
            if not self.markdown.at_line_start:
                self.write("\n")
            self.last = self.current
            self.current = None
            self.markdown = None

    def rule(self) -> None:
        """A horizontal line across the screen, drawn between thinking and reply.

        Full width, unlike the Markdown rules inside a reply, so the two are easy
        to tell apart."""
        self.write(style.dim("─" * shutil.get_terminal_size().columns) + "\n")

    def notice(self, text: str) -> None:
        self.end_block()
        print(style.warning(f"  [{text}]"))
