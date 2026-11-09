"""Tests for core.tools.run_tool: the dispatch allowlist and raw pass-through.

run_tool hands the model's input to the workspace untouched: coercing there
(bool("false") is True) would silently flip a tool's meaning, so the workspace
stays the one place that validates, and only the six listed tools dispatch at
all.
"""

import unittest

from core.tools import run_tool
from core.workspace import ToolError

from tests.util import WorkspaceTest


class RunToolTest(WorkspaceTest):
    def test_only_the_listed_tools_are_dispatched(self) -> None:
        with self.assertRaises(ToolError) as cm:
            run_tool(self.ws, "run_command", {"path": "."})
        self.assertIn("unknown tool", str(cm.exception))
        with self.assertRaises(ToolError):
            run_tool(self.ws, "", {})

    def test_missing_arguments_are_validated_by_the_workspace(self) -> None:
        with self.assertRaises(ToolError) as cm:
            run_tool(self.ws, "list_directory", {})
        self.assertIn("path must be a non-empty string", str(cm.exception))
        with self.assertRaises(ToolError):
            run_tool(self.ws, "read_file", {"path": None})

    def test_search_arguments_pass_through_raw(self) -> None:
        # run_tool must not coerce: bool("yes") would silently turn
        # case-insensitivity on, so the workspace refuses the non-boolean.
        (self.root / "a.txt").write_text("alpha\n")
        with self.assertRaises(ToolError) as cm:
            run_tool(self.ws, "search", {"pattern": "alpha", "ignore_case": "yes"})
        self.assertIn("ignore_case must be boolean", str(cm.exception))
        with self.assertRaises(ToolError):
            run_tool(self.ws, "search", {"pattern": "alpha", "max_results": "5"})
        result = run_tool(self.ws, "search", {"pattern": "alpha", "ignore_case": False})
        self.assertEqual(result, "a.txt:1: alpha")


if __name__ == "__main__":
    unittest.main()
