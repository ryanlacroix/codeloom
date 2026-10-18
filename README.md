# narrow-harness

A minimal, locked-down coding agent for Claude, or for models on Ollama Cloud. The model can
**only** list, read, and write files inside one project directory. It has no tool for running commands, and the harness
never executes anything.

## Setup

Requires Python 3.10+. No packages to install: it uses only the standard library and calls
the Claude API (or Ollama Cloud's API) directly over HTTPS.

Ollama Cloud is the default provider. Create an API key at <https://ollama.com/settings/keys>,
put it in `ollama_key.txt` in the same folder as `harness.py`, and make it readable only by you:

```bash
echo '<your Ollama API key>' > ollama_key.txt
chmod 600 ollama_key.txt
```

To use Claude instead, put your Claude access token in `key.txt` the same way:

```bash
echo '<your access token>' > key.txt
chmod 600 key.txt
```

Both key files are listed in `.gitignore`. The harness warns at startup if other users can read
them, and the model is never allowed to read or write either one, even when the project folder
contains it.

## Usage

```bash
python3 harness.py /path/to/project                  # Ollama Cloud, gemma4:31b, thinking on
python3 harness.py /path/to/project --yes            # auto-approve writes
python3 harness.py /path/to/project --model <model>  # any Ollama Cloud model
python3 harness.py /path/to/project --provider claude --model claude-sonnet-5-5
```

With Ollama (the default provider), `--model` takes any model name from Ollama Cloud's catalogue
(<https://ollama.com/search?c=cloud>, or `GET https://ollama.com/api/tags`); the default is
`gemma4:31b`. Pick a model that supports tool calling. Requests go to `https://ollama.com/api/chat`.

With `--provider claude`, `--model` accepts `claude-opus-5-5` (default), `claude-sonnet-5-5`,
`claude-haiku-5-5` and `claude-fable-5-1`.

Thinking is on by default for Ollama models, which helps with multi-step or multi-file changes
at the cost of slower turns and more usage. gpt-oss models only accept a level, so use
`--think low|medium|high` with them. A model that can't think rejects the request: use
`--think off` for those. `--think` doesn't apply to Claude, whose reasoning is already on.

```bash
python3 harness.py /path/to/project --model gpt-oss:120b --think high
python3 harness.py /path/to/project --model <non-thinking model> --think off
```

Replies stream in as they're written. The model's thinking is shown first, dimmed under a
`thinking>` label, then the reply under the model's name. Pass `--hide-thinking` to show only the
reply. With Claude, the thinking shown is a summary: the API never returns Claude's full reasoning.

Failed requests are retried twice with backoff on rate limits (429), overloads and server
errors (5xx), timeouts and dropped connections, as long as the reply hasn't started yet. A reply
that fails part-way isn't retried, because its start has already been shown; send the message again.

At the `you>` prompt, type a request. `/reset` clears the conversation, `/exit` quits.

Start a message with `/talk` to ask a question without anything being changed, for example
`/talk how does the retry logic work?`. The model can still list and read files, but any write
it attempts during that message is refused.

Output is colored when it goes to a terminal. Set `NO_COLOR=1` to turn colors off, or
`FORCE_COLOR=1` to keep them when piping output.

## Cost

After each request the harness prints the tokens used and an estimated cost, plus a running
total for the session:

```
  [3,180 input tokens (1,500 from cache), 150 output · ~$0.0117 this turn · ~$0.0117 this session]
```

With Ollama Cloud, which is billed by subscription rather than per token, only the token
counts are shown.

The estimate uses the prices in `models.py` (October 2026); check the Claude Console for current
rates and your actual spend. Prompt caching is on: the tool definitions, system prompt and
conversation so far are cached between calls, so repeated content is billed at the much lower
cache-read rate. Cached content expires after 5 minutes without a request. `/reset` starts a new
conversation, which is cheaper than carrying an old one into an unrelated task.

## Files

| File | Purpose |
|---|---|
| `harness.py` | Entry point: command-line options and the `you>` prompt loop |
| `agent.py` | Conversation loop: model settings, system prompt, tool-call handling |
| `models.py` | Supported Claude models and their prices, used for the cost estimate |
| `tools.py` | Tool definitions sent to Claude, and the allowlist that dispatches them |
| `style.py` | Terminal colors, and when to turn them off |
| `workspace.py` | Sandboxed file access, encodings, line endings, write approval |
| `api_client.py` | HTTPS calls to the Claude API: headers, retries, errors |
| `ollama_client.py` | Ollama Cloud client: translates Claude-format requests and replies to and from `/api/chat` |

## Safety model

- Three tools only: `list_directory`, `read_file`, `write_file`. Any other tool name is refused.
- Every path is resolved (following symlinks and `..`) and rejected if it lands outside the project root.
- Writes need your approval unless you pass `--yes`. Reads and writes are capped at 1 MB.
- Binary files (any file containing a NUL byte, e.g. `.xlsx`, `.dll`) can't be read or overwritten.

## Encodings and line endings

- Files are read as UTF-8 if they're valid UTF-8 (a BOM is kept), otherwise as cp1252, the
  encoding Excel uses when saving plain "CSV" on Windows. Files with one of the five bytes cp1252
  leaves undefined fall back to latin-1. Use `--fallback-encoding <name>` to change cp1252.
- An existing file is saved back in the encoding and line endings (CRLF or LF) it was read with.
  If Claude writes a character that encoding can't hold, the write is refused and Claude is told why.
- Claude must read an existing file before overwriting it. New files are UTF-8 with LF endings.
- No `subprocess`, `os.system`, `exec`, or `eval` anywhere in the harness.
