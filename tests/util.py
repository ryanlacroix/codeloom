"""Shared helpers for the tests: temporary workspaces and approval stubs."""

import tempfile
import unittest
from pathlib import Path

import ui.style

from core.workspace import Workspace

# Several tests assert on painted text (diffs, warnings), and whether colors are
# on depends on the environment: a terminal turns them on, so the same run could
# pass plain and fail wrapped. paint() and friends read this flag on every call,
# so forcing it off makes the whole test process see plain strings.
ui.style.ENABLED = False


def approve_all(message: str, diff_lines: list, risk_note: str | None = None) -> bool:
    """An approver that approves every write it is asked about."""
    return True


def make_workspace(root: Path, **overrides) -> Workspace:
    """A Workspace over a temporary root whose writes are approved.

    Approval goes through this one helper, so the tests stay independent of
    how loom.py wires the interactive prompt.
    """
    kwargs = {"approver": approve_all}
    kwargs.update(overrides)
    return Workspace(root, **kwargs)


class WorkspaceTest(unittest.TestCase):
    """Base class for tests that need a fresh workspace on disk."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = make_workspace(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()
