"""The conversation loop: sends messages to the model and runs the tools it calls."""

from api_client import Client
from ollama_client import OllamaClient
from models import MODELS, Usage, response_usage
from tools import TOOLS, run_tool
from workspace import ToolError, Workspace

MAX_TOKENS = 16000
# Prompt caching: the marker on the system prompt caches the tool definitions
# and system prompt (tools come first in the request), and the top-level
# marker caches the conversation so far. Each call then pays full price only
# for what is new since the previous call. This only works while the history
# is append-only, so earlier messages must never be edited.
CACHE_CONTROL = {"type": "ephemeral"}

SYSTEM_PROMPT = """You are a coding assistant working inside a single project directory.

You can only list directories, read files, and write files, using the tools provided. \
You cannot run commands, execute code, run tests, install packages, or access the network. \
All paths are relative to the project root. When a task would normally require running \
something (tests, builds, scripts), make the file changes and tell the user what to run \
themselves.

You must read an existing file before overwriting it. write_file replaces the whole file, \
so always write the complete new contents. Existing files are saved back in their original \
encoding and line endings; if read_file says a file is not UTF-8, only use characters that \
encoding can represent. Binary files (for example .xlsx or .dll) cannot be read or written. \
The user must approve every write; if a write is declined, ask what they would like instead."""


class Agent:
    def __init__(self, client: Client | OllamaClient, workspace: Workspace, model: str):
        self.client = client
        self.workspace = workspace
        self.model = model
        # Only Claude models have prices; Ollama Cloud is billed by subscription.
        self.priced = model in MODELS
        self.label = "claude" if self.priced else model
        self.messages = []
        self.session_usage = Usage()

    def send(self, user_text: str) -> None:
        """Run one user turn, then print its token usage and estimated cost.

        The summary is printed even if the turn fails part-way, because the
        API calls that did complete are still billed.
        """
        turn_usage = Usage()
        try:
            self.run_turn(user_text, turn_usage)
        finally:
            self.session_usage.add(turn_usage)
            if (turn_usage.total_input or turn_usage.output) and self.priced:
                print(
                    f"\n  [{turn_usage.total_input:,} input tokens ({turn_usage.cache_read:,} from cache), "
                    f"{turn_usage.output:,} output · ~${turn_usage.cost:.4f} this turn · "
                    f"~${self.session_usage.cost:.4f} this session]"
                )
            elif turn_usage.total_input or turn_usage.output:
                print(f"\n  [{turn_usage.total_input:,} input tokens, {turn_usage.output:,} output]")

    def request_body(self) -> dict:
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": CACHE_CONTROL}],
            "tools": TOOLS,
            "messages": self.messages,
            "output_config": {"effort": "high"},
            "cache_control": CACHE_CONTROL,
        }
        if self.priced and MODELS[self.model].fallbacks:
            body["fallbacks"] = "default"
        return body

    def run_turn(self, user_text: str, turn_usage: Usage) -> None:
        self.messages.append({"role": "user", "content": user_text})
        while True:
            response = self.client.create_message(self.request_body())
            turn_usage.add(response_usage(response, self.model))
            content = response.get("content", [])
            stop_reason = response.get("stop_reason")
            # Append the full content (including thinking blocks) unchanged.
            self.messages.append({"role": "assistant", "content": content})

            for block in content:
                if block["type"] == "text" and block["text"].strip():
                    print(f"\n{self.label}> {block['text']}")
                elif block["type"] == "fallback":
                    from_model = block.get("from", {}).get("model", "a model")
                    to_model = block.get("to", {}).get("model", "another model")
                    print(f"  [{from_model} declined; {to_model} continued]")

            if stop_reason == "refusal":
                print("\n  [The model declined this request.]")
                return
            if stop_reason == "max_tokens":
                print("\n  [Response hit the max_tokens limit and was cut off.]")
                return
            if stop_reason != "tool_use":
                return

            tool_results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                tool_input = block.get("input") or {}
                print(f"  -> {block['name']}({tool_input.get('path', '')})")
                try:
                    result = run_tool(self.workspace, block["name"], tool_input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": result})
                except (ToolError, OSError) as e:
                    print(f"     error: {e}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": f"Error: {e}. This tool call failed and changed nothing.",
                            "is_error": True,
                        }
                    )
            self.messages.append({"role": "user", "content": tool_results})
