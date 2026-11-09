# CodeLoom

A minimal, locked-down agentic coding harness for Ollama Cloud models, written
with only the Python standard library.

The model gets six file tools (list_directory, read_file, search, write_file,
edit_file, append_file), all confined to one workspace directory — and nothing
else: no shell, no execution, no network. Every write shows the user a diff and
asks first. Slash commands add a read-only side question (`/talk`), a
plan-then-approve flow (`/plan` → `/ok`), model switching (`/model`), pasting
via your editor (`/paste`), and thinking toggles.

## Quick start

    python loom.py <project-dir>            # interactive session
    python loom.py <dir> --talk            # read-only: nothing can be changed
    python loom.py <dir> --yes             # auto-approve writes (no diff review)
    python -m unittest discover -v         # run the test suite

Put your Ollama API key in `ollama_key.txt` next to the harness (it is sent as a
Bearer token). Optionally, `redact.json` next to loom.py lists strings that are
replaced with `<redacted>` in everything the model ever sees. Plans are saved to
`.loom-temp/plan.md`; the /paste buffer is `.loom-temp/paste.txt`.

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
  not be a blocked file, and not sit under a protected directory (`.git`).
  Both `resolve()` and search's directory walk use it, so a named path and the
  walk can never disagree.
- Symlinks are never followed by the search walk; naming one directly resolves
  it first, and one pointing outside the root is refused.
- `.git` is protected: its config and hooks are a command-execution vector.
- Blocked files (the API key, redact.json, the paste buffer) can never be read
  or written, even inside the workspace.
- Every write needs the injected approver's yes; the harness's approver shows
  a diff and flags writes to build/test entry files.
- Redaction runs before any truncation (search's 200-char cap, read_file's page
  cap), so a configured string is never shown as a partial fragment — and every
  text that joins the conversation passes the filter.
- Error text sent to the model is scrubbed of absolute paths (`scrub_paths`).

## Tests

`python -m unittest discover -v` pins the sandbox invariants (containment,
blocked and protected paths, redaction-before-truncation, encoding round-trips),
the redaction filter, the Ollama message translation, the agent turn loop (with
a fake client), and the session state machine. Run them before and after any
change under `core/` — they are the acceptance criteria for refactors.
