# narrow-harness

A minimal, locked-down coding agent for Claude. Claude can **only** list, read, and write
files inside one project directory. It has no tool for running commands, and the harness
never executes anything.

## Setup

Requires Python 3.10+. No packages to install: it uses only the standard library and calls
the Claude API directly over HTTPS.

Put your access token in `key.txt`, in the same folder as `harness.py`, and make it
readable only by you:

```bash
echo '<your access token>' > key.txt
chmod 600 key.txt
```

`key.txt` is listed in `.gitignore`. The harness warns at startup if other users can read it,
and Claude is never allowed to read or write it, even when the project folder contains it.

## Usage

```bash
python3 harness.py /path/to/project      # asks before every file write
python3 harness.py /path/to/project --yes  # auto-approve writes
python3 harness.py /path/to/project --model claude-sonnet-5-5
```

`--model` accepts `claude-opus-5-5` (default), `claude-sonnet-5-5`, `claude-haiku-5-5` and
`claude-fable-5-1`.

Failed requests are retried twice with backoff on rate limits (429), overloads and server
errors (5xx), timeouts and dropped connections.

At the `you>` prompt, type a request. `/reset` clears the conversation, `/exit` quits.

## Cost

After each request the harness prints the tokens used and an estimated cost, plus a running
total for the session:

```
  [3,180 input tokens (1,500 from cache), 150 output · ~$0.0117 this turn · ~$0.0117 this session]
```

The estimate uses the prices in `models.py` (October 2026); check the Claude Console for current
rates and your actual spend. Prompt caching is on: the tool definitions, system prompt and
conversation so far are cached between calls, so repeated content is billed at the much lower
cache-read rate. Cached content expires after 5 minutes without a request. `/reset` starts a new
conversation, which is cheaper than carrying an old one into an unrelated task.

## Files

| File | Purpose |
|---|---|
| `harness.py` | Entry point: command-line options and the `you>` prompt loop |
| `agent.py` | Conversation loop with Claude: model settings, system prompt, tool-call handling |
| `models.py` | Supported models and their prices, used for the cost estimate |
| `tools.py` | Tool definitions sent to Claude, and the allowlist that dispatches them |
| `workspace.py` | Sandboxed file access, encodings, line endings, write approval |
| `api_client.py` | HTTPS calls to the Claude API: headers, retries, errors |

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
