"""The model names the harness uses, and the token counts of a turn.

Ollama Cloud serves any model in its cloud catalogue (https://ollama.com/search?c=cloud)
and bills by subscription rather than per token, so usage is counted, not priced.
"""

from dataclasses import dataclass

# The default model, used when --model is not given.
OLLAMA_DEFAULT_MODEL = "glm-5.3-flash"
# The fixed list the /model command prints; /model <name> accepts any Ollama Cloud name.
OLLAMA_MODELS = ("glm-5.3", "glm-5.3-flash", "minimax-m3")


@dataclass
class Usage:
    """Token counts, as returned by one API call or accumulated across calls."""

    input: int = 0
    output: int = 0

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.output += other.output


def format_tokens(count: int) -> str:
    """Render a token count the way the usage footer reports the context: "~12.3k".

    Counts under 1,000 stay exact ("~890"); larger ones get one decimal place
    with a trailing ".0" trimmed ("12,000" -> "~12k"), scaled up to "m" from a
    million. The "~" marks it as an approximation: agent.send() reports the
    newest completed call's prompt, and the message typed next is not counted
    yet.
    """
    for scale, unit in ((1_000_000, "m"), (1_000, "k")):
        if count >= scale:
            tenths = (count * 10 + scale // 2) // scale  # round half up to one decimal
            whole, tenth = divmod(tenths, 10)
            text = str(whole) if tenth == 0 else f"{whole}.{tenth}"
            return f"~{text}{unit}"
    return f"~{count}"
