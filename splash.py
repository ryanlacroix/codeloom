"""Startup splash: the "Portage" title art shown when the harness starts.

The art is embedded as a constant (pre-generated, originally saved at
.portage-temp/splash-text.txt) rather than read from disk, so the splash works
no matter how or from where the harness is launched. The block is centered
horizontally in the terminal as a whole (never line-by-line, which would break
the figure). Nothing is printed when colors are off (NO_COLOR, non-terminal
output, dumb terminal), keeping that output clean.
"""

import shutil

import style

# "PORTAGE" in the block-shading (▀▄█) figlet style, kept exactly as generated
# (leading blank line and trailing whitespace trimmed; see
# .portage-temp/splash-text.txt).
ART = r"""▄▄▄▄▄▄▄                                        ▄
███▀▀███▄              ██                    ▀▀█▀▀
███▄▄███▀ ▄███▄ ████▄ ▀██▀▀ ▀▀█▄ ▄████ ▄█▀█▄
███▀▀▀▀   ██ ██ ██ ▀▀  ██  ▄█▀██ ██ ██ ██▄█▀
███       ▀███▀ ██     ██  ▀█▄██ ▀████ ▀█▄▄▄
                                    ██
                                  ▀▀▀"""


def print_splash() -> None:
    """Print the title art centered in the terminal, bold cyan, then a blank
    line; nothing if colors are off."""
    if not style.colors_enabled():
        return
    lines = ART.splitlines()
    # The block glyphs (▀▄█) are single-cell in terminals, so len() gives the
    # display width. get_terminal_size() falls back to COLUMNS or 80 when
    # stdout is not a terminal; max(0, ...) covers widths narrower than the art.
    art_width = max(len(line) for line in lines)
    pad = max(0, (shutil.get_terminal_size().columns - art_width) // 2)
    print(style.paint("\n".join(" " * pad + line for line in lines), style.BOLD, style.CYAN))
    print()