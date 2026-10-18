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
        "line endings. New files are UTF-8 with '\\n' line endings. Parent directories are created "
        "as needed. The user must approve the write.",
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
]


def run_tool(workspace: Workspace, name: str, tool_input: dict) -> str | list:
    # Explicit allowlist: anything not named here is refused.
    if name == "list_directory":
        return workspace.list_directory(tool_input.get("path"))
    if name == "read_file":
        return workspace.read_file(tool_input.get("path"))
    if name == "write_file":
        return workspace.write_file(tool_input.get("path"), tool_input.get("content"))
    raise ToolError(f"unknown tool: {name}")
