# CodeLoom

A minimal, locked-down agentic coding harness for Ollama Cloud models — or any
Anthropic-compatible `/v1/messages` endpoint via `--api-config` — written with only
the Python standard library.

The model gets six file tools (list_directory, read_file, search, write_file,
edit_file, append_file), all confined to one workspace directory — and nothing
else: no shell, no execution, no network. Every write shows the user a diff and
asks first. Slash commands add a read-only side question (`/talk`), a
plan-then-approve flow (`/plan` → `/ok`), model switching (`/model`), pasting
via your editor (`/paste`), and thinking toggles.

## Quick start

    python loom.py <project-dir>            # interactive session (needs config/ollama_config.json)
    python loom.py <dir> --talk            # read-only: nothing can be changed
    python loom.py <dir> --yes             # auto-approve writes (no diff review)
    python -m unittest discover -v         # run the test suite

Every secret and the harness's own configuration live in `config/`, a directory
inside this repo that is in `.gitignore` (so nothing lands in git history) and
completely invisible to the model: it is passed to the Workspace as a protected
directory, so nothing in it can be listed, read, written, or searched, even when
this repo itself is the workspace.

- `config/ollama_config.json` — the Ollama endpoint, the default provider; the
  file is required, because it holds the endpoint's key:

    {
      "api_key": "sk-...",
      "base_url": "https://ollama.com",
      "model": "glm-5.3-flash",
      "models": ["glm-5.3", "glm-5.3-flash", "minimax-m3"]
    }

  `api_key` is required (sent as a Bearer token). `base_url` defaults to Ollama
  Cloud (`/api/chat` is appended to the base), `model` is the default model and
  `models` the list `/model` offers (both fall back to the catalogue in
  `api/models.py`), and `headers` adds or overrides request headers.
- `config/redact.json` — optionally, strings replaced with `<redacted>` in
  everything the model ever sees
- `config/anthropic_config.json` — an Anthropic-compatible `/v1/messages` endpoint, selected only by `--api-config` (normally this file); the shape:

    {
      "base_url": "https://llm.internal.example.com",
      "api_key": "sk-...",
      "model": "claude-sonnet-4-5",
      "models": ["claude-sonnet-4-5", "claude-haiku-4-5"]
    }

`base_url` and `api_key` are required (`/v1/messages` is appended to the base);
`model` is the default model (the first `models` entry if absent), `models` is
the list `/model` offers, and `headers` adds or overrides request headers. The
key is sent as both `x-api-key` and a Bearer token. Only `--api-config FILE`
selects this endpoint — the file's presence alone does nothing. `FILE` points
at a file kept elsewhere, and `--redact-file` overrides the redaction list's
path the same way. Plans are saved to `.loom-temp/plan.md`; the /paste buffer is
`.loom-temp/paste.txt`.

## Layout and dependency direction

    loom.py            the entry point and composition root: argparse, key and
                       redaction config, the command registry, the prompt loop
    core/
      agent.py         the conversation loop: request body, tool dispatch, usage
      prompts.py       every fixed text sent to the model (system prompt, notes)
      session.py       the loop's state: plan mode, pending paste, plan saving
      tools.py         the tool schemas and the dispatch allowlist
      workspace.py     the sandbox and the six tools; takes an injected approver
      redact.py        the redaction filter and its config loader
    api/
      http_client.py   shared HTTP plumbing: errors, retries, the streaming POST
      ollama_client.py the Ollama Cloud chat API client
      anthropic_client.py the Anthropic-compatible /v1/messages client and its config loader
      models.py        the model names and the Usage token counts
    ui/
      style.py         colors, prompt painting, the console clear
      markdown.py      streamed Markdown rendering
      stream.py        the thinking/reply stream printer
      diff.py          the write-confirmation diff
      splash.py        the startup title art
    tests/             the unittest suite (no third-party dependencies)

Dependencies point one way: `core` may import `api` and `ui`; `ui` imports
nothing from `core` or `api`; `api` imports nothing from `core` or `ui`.
loom.py is the only module that ties them together.

## Security model

- One access rule (`Workspace._denial`): a path must resolve inside the root,
  not be a blocked file, and not sit under a protected directory — `.git` by
  name, plus any whole path given in `protected_dirs`. `resolve()`, search's
  directory walk, and list_directory's output all use it, so a named path, the
  walk, and a listing can never disagree.
- The blocked-file and protected-directory comparisons are case-insensitive,
  like `.git`'s: a filesystem can be case-insensitive while case-preserving
  (macOS's default, Windows), and `resolve()` keeps the spelling it was handed
  there, so a model naming `CONFIG/ollama_config.json` must be refused by
  spelling comparison, not left to the filesystem.
- Symlinks are never followed by the search walk; naming one directly resolves
  it first, and one pointing outside the root is refused.
- `.git` is protected: its config and hooks are a command-execution vector.
- `config/` is protected by path: it holds the API keys and the redaction list,
  so nothing in it can be listed, read, written, or searched — completely
  invisible to the model, even when this repo is the workspace. (A project's
  own `config` directory is unaffected; protection is by path, not name.)
- Blocked files (a `--redact-file` or `--api-config` kept outside `config/`,
  the paste buffer) can never be read or written, even inside the workspace.
- No file is ever read into memory whole above the read cap
  (`MAX_FILE_BYTES`, 100 MB): read_file refuses it, search skips it with a
  note, and search's walk stops collecting entries at `MAX_SEARCH_FILES`, so
  neither a giant file nor a giant tree can exhaust memory.
- On Windows, a path component that is a reserved device name (`CON`, `NUL`,
  `COM1`, …) or contains a colon (an alternate-data-stream marker like
  `file.txt:stream`) is refused: an approved write would otherwise land
  somewhere the diff never showed.
- Every write needs the injected approver's yes; the harness's approver shows
  a diff and flags writes to build/test entry files — and, when this repo
  itself sits at or under the workspace, to the harness's own source files,
  which run on the next launch.
- Console output is sanitized (`style.sanitize`): control characters a
  terminal would act on are stripped from every model- or file-controlled
  string, and escape sequences are shown as their visible symbol (␛). Text
  sent to the API is never altered.
- Redaction runs before any truncation (search's 200-char cap, read_file's page
  cap) and before search's match itself, so a configured string is never shown
  as a partial fragment and a search pattern can never probe one's content
  character by character — at most a pattern matches the marker, which a
  reported line shows anyway — and every text that joins the conversation
  passes the filter.
- Error text sent to the model is scrubbed of absolute paths (`scrub_paths`).
- A tool call the API did not type-check, or any unexpected failure while
  running one, becomes a failed tool result rather than a crash: one bad call
  cannot end the session.

## Tests

`python -m unittest discover -v` pins the sandbox invariants (containment,
blocked and protected paths, case-insensitive refusal, redaction-before-
truncation-and-match, line-number agreement, the read caps, encoding
round-trips), the tool dispatch allowlist, the redaction filter, the Ollama
and Anthropic message translations (the Anthropic one with a fake SSE stream),
the agent turn loop (with a fake client, including calls the API did not
type-check), and the session state machine. Run them before and after any
change under `core/` — they are the acceptance criteria for refactors.
