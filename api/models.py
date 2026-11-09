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
