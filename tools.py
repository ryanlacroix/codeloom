"""The tools Claude may call. Anything not listed here is refused."""

from workspace import ToolError, Workspace

TOOLS = [
    {
        "name": "list_directory",
        "description": "List the files and subdirectories in a directory inside the project. "
        "Directories are shown with a trailing '/'. Use '.' for the project root.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to the project root."}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read the full contents of a text file inside the project. Line endings "
        "are shown as '\\n'. If the file is not UTF-8, the result starts with a note naming its encoding.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root."}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file inside the project with the given complete "
        "contents. An existing file must be read first, and is saved in its original encoding and "
        "line endings. Parent directories are created as needed. For changes to existing files or "
        "long new files this is the most expensive tool in output tokens: prefer edit_file or "
        "append_file. The user must approve the write.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root."},
                "content": {"type": "string", "description": "The complete new contents of the file."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the single occurrence of an old snippet in an existing text file with "
        "new text, changing nothing else. An existing file must be read first, and is saved in its "
        "original encoding and line endings. old_text is matched literally (not as a regex) and must "
        "appear exactly once, so include enough surrounding lines to make it unique: the edit is "
        "refused if it matches zero or several times. new_text may be empty, deleting the snippet. "
        "Only the snippet travels in the call, so editing is far cheaper in output tokens than "
        "rewriting the file with write_file; prefer this for changes to existing files. "
        "The user must approve the write.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root."},
                "old_text": {
                    "type": "string",
                    "description": "The exact existing snippet to replace, copied from a fresh read "
                    "of the file, including its indentation.",
                },
                "new_text": {"type": "string", "description": "The replacement. May be empty to delete the snippet."},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "append_file",
        "description": "Add text at the end of a text file, creating it if it does not exist yet "
        "(parent directories are created as needed). Appending to an existing file requires reading "
        "it first, and it is saved in its original encoding and line endings; if it does not end "
        "with a newline, one is inserted first so lines never join. Build a long file in several "
        "append_file chunks (roughly one per 150 lines) rather than one huge write_file. "
        "The user must approve the write.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root."},
                "content": {"type": "string", "description": "Text to add at the end of the file."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
]


def run_tool(workspace: Workspace, name: str, tool_input: dict) -> str | list:
    # Explicit allowlist: anything not named here is refused.
    if name == "list_directory":
        return workspace.list_directory(tool_input.get("path"))
    if name == "read_file":
        return workspace.read_file(tool_input.get("path"))
    if name == "write_file":
        return workspace.write_file(tool_input.get("path"), tool_input.get("content"))
    if name == "edit_file":
        return workspace.edit_file(tool_input.get("path"), tool_input.get("old_text"), tool_input.get("new_text"))
    if name == "append_file":
        return workspace.append_file(tool_input.get("path"), tool_input.get("content"))
    raise ToolError(f"unknown tool: {name}")