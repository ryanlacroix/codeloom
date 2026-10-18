"""Models the harness can use, and their prices for the per-turn cost estimate.

Prices are USD per million tokens, taken from Anthropic's published rates as of
October 2026. Check the pricing page in the Claude Console for current rates:
the cost shown after each turn is only an estimate based on these numbers.
"""

from dataclasses import dataclass

DEFAULT_MODEL = "claude-opus-5-5"
# Ollama Cloud accepts any model in its cloud catalogue (https://ollama.com/search?c=cloud).
# It bills by subscription rather than per token, so its models are not in MODELS
# and no cost is estimated for them.
OLLAMA_DEFAULT_MODEL = "gemma4:31b"
# Claude Haiku 5.5 charges a higher rate when a prompt exceeds this many tokens.
LONG_PROMPT_TOKENS = 100_000
# Writing to the 5-minute prompt cache costs 1.25x the normal input price.
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class Rates:
    input: float  # uncached input tokens
    output: float  # output tokens, including thinking
    cache_read: float  # input tokens read from the prompt cache


@dataclass(frozen=True)
class Model:
    rates: Rates
    # Whether the API can retry a declined request on a fallback model.
    fallbacks: bool
    # Rates used instead when the prompt is longer than LONG_PROMPT_TOKENS.
    long_prompt_rates: Rates | None = None


MODELS = {
    "claude-opus-5-5": Model(Rates(input=4.00, output=20.00, cache_read=0.20), fallbacks=True),
    "claude-sonnet-5-5": Model(Rates(input=2.00, output=10.00, cache_read=0.20), fallbacks=True),
    "claude-haiku-5-5": Model(
        Rates(input=0.10, output=0.50, cache_read=0.01),
        fallbacks=False,
        long_prompt_rates=Rates(input=0.50, output=2.50, cache_read=0.05),
    ),
    "claude-fable-5-1": Model(Rates(input=10.00, output=50.00, cache_read=0.25), fallbacks=True),
}


@dataclass
class Usage:
    """Token counts and estimated cost, accumulated across API calls."""

    input: int = 0  # uncached input
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0
    cost: float = 0.0

    @property
    def total_input(self) -> int:
        return self.input + self.cache_write + self.cache_read

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.cache_write += other.cache_write
        self.cache_read += other.cache_read
        self.output += other.output
        self.cost += other.cost


def priced_usage(usage: dict, model_id: str, default_model_id: str) -> Usage:
    """Turn one API usage record into token counts and an estimated cost.

    A model that is not in MODELS (for example a fallback model) is priced at
    the rates of default_model_id. If that is not in MODELS either (an Ollama
    model), the cost is left at zero.
    """
    result = Usage(
        input=usage.get("input_tokens") or 0,
        cache_write=usage.get("cache_creation_input_tokens") or 0,
        cache_read=usage.get("cache_read_input_tokens") or 0,
        output=usage.get("output_tokens") or 0,
    )
    model = MODELS.get(model_id) or MODELS.get(default_model_id)
    if model is None:
        return result
    rates = model.rates
    if model.long_prompt_rates and result.total_input > LONG_PROMPT_TOKENS:
        rates = model.long_prompt_rates
    result.cost = (
        result.input * rates.input
        + result.cache_write * rates.input * CACHE_WRITE_MULTIPLIER
        + result.cache_read * rates.cache_read
        + result.output * rates.output
    ) / 1_000_000
    return result


def response_usage(response: dict, requested_model: str) -> Usage:
    """Total usage for one API response.

    When a fallback model took over, usage.iterations lists every attempt
    separately (each billed at its own model's rates); otherwise the top-level
    usage covers the whole response.
    """
    usage = response.get("usage") or {}
    served_by = response.get("model") or requested_model
    iterations = usage.get("iterations")
    if not iterations:
        return priced_usage(usage, served_by, requested_model)
    total = Usage()
    for attempt in iterations:
        total.add(priced_usage(attempt, attempt.get("model") or served_by, requested_model))
    return total
