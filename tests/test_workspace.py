"""Tests for core.workspace: the sandbox, the read tools, and the write tools.

These pin the invariants the harness is built on: every path stays inside the
root (or is refused with a reason), .git and blocked files are untouchable,
a directory passed in protected_dirs (the harness's config/) is invisible
to every tool, writes require a prior read and keep encoding and line endings,
and redaction runs before any truncation so a configured string is never shown
as a fragment.
"""

import os
import unittest
from unittest import mock

from core.redact import Redactor
from core.workspace import (
    MAX_WRITE_BYTES,
    ToolError,
    Workspace,
    split_file_lines,
    windows_name_error,
)

from tests.util import WorkspaceTest, approve_all, make_workspace


class ResolveTest(WorkspaceTest):
    def test_inside_paths_resolve_to_their_target(self) -> None:
        target = self.ws.resolve("sub/../a.txt")
        self.assertEqual(target, self.ws.root / "a.txt")

    def test_parent_escape_is_denied(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.resolve("../outside.txt")
        self.assertIn("outside the project directory", str(cm.exception))

    def test_absolute_paths_are_denied(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.resolve("/etc/passwd")
        self.assertIn("outside the project directory", str(cm.exception))

    def test_null_byte_and_empty_paths_are_refused(self) -> None:
        for bad in (None, "   ", "a\x00b"):
            with self.assertRaises(ToolError):
                self.ws.resolve(bad)

    def test_blocked_file_is_denied(self) -> None:
        secret = self.root / "secret.txt"
        secret.write_text("hush")
        blocked_ws = Workspace(self.root, approver=approve_all, blocked=(secret,))
        with self.assertRaises(ToolError) as cm:
            blocked_ws.resolve("secret.txt")
        self.assertIn("a protected file", str(cm.exception))

    def test_case_variant_spellings_of_blocked_files_are_denied(self) -> None:
        # A case-insensitive filesystem resolves the spelling the model typed
        # (Path.resolve keeps it), so exact comparison would miss the file:
        # the key is case-insensitive instead, like .git's part check.
        secret = self.root / "secret.txt"
        secret.write_text("hush")
        blocked_ws = Workspace(self.root, approver=approve_all, blocked=(secret,))
        for name in ("SECRET.TXT", "Secret.Txt", "secret.TXT"):
            with self.assertRaises(ToolError) as cm:
                blocked_ws.resolve(name)
            self.assertIn("a protected file", str(cm.exception))

    def test_git_directory_is_denied_at_any_depth(self) -> None:
        for path in (".git/config", "sub/.git/hooks/pre-commit", ".GIT/config"):
            with self.assertRaises(ToolError) as cm:
                self.ws.resolve(path)
            self.assertIn("protected directory", str(cm.exception))

    def test_symlink_pointing_outside_is_denied(self) -> None:
        outside = self.root.parent / "outside_target.txt"
        outside.write_text("data")
        self.addCleanup(outside.unlink)
        link = self.root / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available on this platform")
        with self.assertRaises(ToolError) as cm:
            self.ws.resolve("link.txt")
        self.assertIn("outside the project directory", str(cm.exception))


class ProtectedDirTest(WorkspaceTest):
    """A whole directory refused by path (the harness's config/, which holds
    the API keys and the redaction list): nothing in it can be resolved,
    listed, read, written, or searched — even when the workspace contains it —
    while a directory of the same name elsewhere stays reachable."""

    def setUp(self) -> None:
        super().setUp()
        (self.root / "config").mkdir()
        (self.root / "config" / "ollama_config.json").write_text("sk-secret\n")
        (self.root / "a.txt").write_text("plain\n")
        self.ws = make_workspace(self.root, protected_dirs=(self.root / "config",))

    def test_the_directory_is_denied_at_any_depth(self) -> None:
        for path in ("config", "config/ollama_config.json", "config/sub/more.txt", "config/../config/ollama_config.json"):
            with self.assertRaises(ToolError) as cm:
                self.ws.resolve(path)
            self.assertIn("protected directory", str(cm.exception))

    def test_reads_inside_the_directory_are_refused(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.read_file("config/ollama_config.json")
        self.assertIn("protected directory", str(cm.exception))

    def test_writes_inside_the_directory_are_refused(self) -> None:
        for write in (
            lambda: self.ws.write_file("config/new.txt", "x\n"),
            lambda: self.ws.append_file("config/new.txt", "x\n"),
        ):
            with self.assertRaises(ToolError) as cm:
                write()
            self.assertIn("protected directory", str(cm.exception))

    def test_the_directory_is_hidden_from_listings(self) -> None:
        # Not merely untouchable: the name never reaches the model, and a
        # blocked file is hidden the same way. Listing the directory itself is
        # refused by resolve(), like naming any file inside it.
        self.assertEqual(self.ws.list_directory("."), "a.txt")
        with self.assertRaises(ToolError):
            self.ws.list_directory("config")
        blocked_ws = make_workspace(
            self.root, protected_dirs=(self.root / "config",), blocked=(self.root / "a.txt",)
        )
        self.assertEqual(blocked_ws.list_directory("."), "(empty directory)")

    def test_the_directory_is_never_searched(self) -> None:
        self.assertEqual(self.ws.search("secret"), "(no matches)")
        self.assertEqual(self.ws.search("plain", path="."), "a.txt:1: plain")

    def test_a_same_named_directory_elsewhere_stays_reachable(self) -> None:
        (self.root / "sub" / "config").mkdir(parents=True)
        (self.root / "sub" / "config" / "settings.txt").write_text("fine\n")
        self.assertEqual(self.ws.list_directory("sub"), "config/")
        self.assertEqual(self.ws.read_file("sub/config/settings.txt"), "fine\n")

    def test_case_variant_spellings_of_the_directory_are_denied(self) -> None:
        # 'CONFIG/...' resolves to the protected directory on a case-preserving
        # filesystem, so the comparison must not depend on spelling either.
        # resolve() needs no existing file, so this holds on every platform.
        for path in ("CONFIG/ollama_config.json", "Config/new.txt", "config/OLLAMA_CONFIG.json"):
            with self.assertRaises(ToolError) as cm:
                self.ws.resolve(path)
            self.assertIn("protected directory", str(cm.exception))
        with self.assertRaises(ToolError):
            self.ws.write_file("Config/new.txt", "x\n")


class ListDirectoryTest(WorkspaceTest):
    def test_directories_first_then_files(self) -> None:
        (self.root / "zdir").mkdir()
        (self.root / "bdir").mkdir()
        (self.root / "afile.txt").write_text("x")
        # the listing travels to the model as text, so it is one newline-joined string
        self.assertEqual(self.ws.list_directory("."), "bdir/\nzdir/\nafile.txt")

    def test_empty_directory(self) -> None:
        (self.root / "empty").mkdir()
        self.assertEqual(self.ws.list_directory("empty"), "(empty directory)")

    def test_not_a_directory(self) -> None:
        (self.root / "afile.txt").write_text("x")
        with self.assertRaises(ToolError):
            self.ws.list_directory("afile.txt")


class ReadFileTest(WorkspaceTest):
    def test_small_file_reads_whole_as_plain_text(self) -> None:
        (self.root / "f.txt").write_bytes(b"one\ntwo\nthree\n")
        self.assertEqual(self.ws.read_file("f.txt"), "one\ntwo\nthree\n")

    def test_empty_file(self) -> None:
        (self.root / "f.txt").write_bytes(b"")
        self.assertEqual(self.ws.read_file("f.txt"), "(empty file)")

    def test_crlf_is_normalised_for_reading(self) -> None:
        (self.root / "f.txt").write_bytes(b"alpha\r\nbeta\r\n")
        self.assertEqual(self.ws.read_file("f.txt"), "alpha\nbeta\n")

    def test_binary_file_is_refused(self) -> None:
        (self.root / "f.bin").write_bytes(b"ab\x00cd")
        with self.assertRaises(ToolError) as cm:
            self.ws.read_file("f.bin")
        self.assertIn("appears to be a binary file", str(cm.exception))

    def test_utf8_bom_is_stripped(self) -> None:
        (self.root / "f.txt").write_bytes(b"\xef\xbb\xbfhello\n")
        self.assertEqual(self.ws.read_file("f.txt"), "hello\n")

    def test_fallback_encoding_carries_a_note(self) -> None:
        (self.root / "f.txt").write_bytes(b"caf\xe9 noir\n")  # cp1252 é
        result = self.ws.read_file("f.txt")
        self.assertEqual([block["type"] for block in result], ["text", "text"])
        self.assertIn("encoded as cp1252", result[0]["text"])
        self.assertEqual(result[1]["text"], "café noir\n")

    def test_latin1_is_the_last_resort(self) -> None:
        (self.root / "f.txt").write_bytes(b"caf\x81e\n")  # \x81 undefined in cp1252
        result = self.ws.read_file("f.txt")
        self.assertIn("encoded as latin-1", result[0]["text"])
        self.assertEqual(result[1]["text"], "caf\x81e\n")

    def test_offset_and_limit_window(self) -> None:
        (self.root / "f.txt").write_bytes(b"one\ntwo\nthree\n")
        result = self.ws.read_file("f.txt", offset=2, limit=1)
        self.assertEqual(result[-1]["text"], "two")
        self.assertIn("showing lines 2-2 of 3 of f.txt", result[0]["text"])
        self.assertIn("read_file('f.txt', offset=3, limit=1)", result[0]["text"])

    def test_offset_past_the_end(self) -> None:
        (self.root / "f.txt").write_bytes(b"one\ntwo\nthree\n")
        with self.assertRaises(ToolError) as cm:
            self.ws.read_file("f.txt", offset=10)
        self.assertIn("offset 10 is past the end of the file, which has 3 lines", str(cm.exception))

    def test_offset_and_limit_must_be_integers(self) -> None:
        for kwargs in ({"offset": -1}, {"offset": True}, {"offset": "2"}, {"limit": -3}, {"limit": "2"}):
            with self.assertRaises(ToolError):
                self.ws.read_file("f.txt", **kwargs)


class SearchTest(WorkspaceTest):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "a.txt").write_text("alpha beta\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.txt").write_text("gamma beta\n")

    def test_matches_report_relative_path_and_line(self) -> None:
        result = self.ws.search("beta")
        self.assertIn("a.txt:1: alpha beta", result)
        self.assertIn("sub/b.txt:1: gamma beta", result)

    def test_single_file_path_and_ignore_case(self) -> None:
        self.assertEqual(self.ws.search("ALPHA", path="a.txt", ignore_case=True), "a.txt:1: alpha beta")

    def test_no_matches(self) -> None:
        self.assertEqual(self.ws.search("nothing-like-this"), "(no matches)")

    def test_max_results_cap_and_note(self) -> None:
        result = self.ws.search("beta", max_results=1)
        self.assertEqual(result.splitlines()[0], "a.txt:1: alpha beta")
        self.assertIn("stopped at the 1-match cap", result)

    def test_long_lines_are_cut_to_200_chars(self) -> None:
        (self.root / "long.txt").write_text("z" * 250 + "\n")
        line = self.ws.search("zzz", path="long.txt").split(": ", 1)[1]
        self.assertEqual(len(line), 200 + len("..."))
        self.assertTrue(line.endswith("..."))

    def test_invalid_pattern(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.search("([")
        self.assertIn("invalid pattern", str(cm.exception))

    def test_argument_validation(self) -> None:
        for kwargs in (
            {"pattern": ""},
            {"pattern": None},
            {"pattern": "x", "max_results": 0},
            {"pattern": "x", "max_results": True},
            {"pattern": "x", "max_results": "5"},
            {"pattern": "x", "ignore_case": "yes"},
        ):
            with self.assertRaises(ToolError):
                self.ws.search(**kwargs)

    def test_binary_files_are_skipped_with_a_note(self) -> None:
        (self.root / "bin.dat").write_bytes(b"needle\x00binary")
        result = self.ws.search("needle")
        self.assertIn("(skipped 1 binary or unreadable file)", result)

    def test_git_is_never_searched(self) -> None:
        (self.root / ".git" / "hooks").mkdir(parents=True)
        (self.root / ".git" / "hooks" / "pre-commit").write_text("needle\n")
        self.assertEqual(self.ws.search("needle"), "(no matches)")

    def test_blocked_files_are_never_searched(self) -> None:
        (self.root / "secret.txt").write_text("needle\n")
        blocked_ws = Workspace(self.root, approver=approve_all, blocked=(self.root / "secret.txt",))
        self.assertEqual(blocked_ws.search("needle"), "(no matches)")

    def test_redaction_happens_before_the_line_cap(self) -> None:
        (self.root / "f.txt").write_text("x" * 190 + "SECRETKEY" + "y" * 60 + "\n")
        self.ws.redactor = Redactor(["SECRETKEY"])
        line = self.ws.search("xxxxx", path="f.txt").split(": ", 1)[1]
        self.assertIn("<redacted>", line)
        self.assertNotIn("SECRET", line)

    def test_search_cannot_probe_redacted_strings(self) -> None:
        # The match runs on the redacted line: a pattern matching the secret
        # itself, or any prefix of it, reports nothing, so match-or-no-match
        # is not an oracle for a configured string's content.
        (self.root / "f.txt").write_text("login abc123 now\n")
        self.ws.redactor = Redactor(["abc123"])
        for pattern in ("abc123", "abc12", "abc", "abc123xyz"):
            self.assertEqual(self.ws.search(pattern), "(no matches)")
        # A clean pattern still reports the line, with the secret redacted.
        result = self.ws.search("login")
        self.assertIn("f.txt:1: login <redacted> now", result)
        self.assertNotIn("abc123", result)

    def test_line_numbers_agree_between_search_and_paged_reads(self) -> None:
        # \f and a lone \r are line breaks for str.splitlines() but not here:
        # search and read_file must number lines identically, or an offset
        # taken from a search hit lands on the wrong line. (The search is
        # path-scoped: setUp's sub/b.txt also contains 'gamma'.)
        (self.root / "f.txt").write_bytes(b"alpha\nbeta\x0cgamma\ndelta\n")
        self.assertEqual(self.ws.search("gamma", path="f.txt"), "f.txt:2: beta\x0cgamma")
        self.assertEqual(self.ws.read_file("f.txt", offset=2, limit=1)[-1]["text"], "beta\x0cgamma")
        (self.root / "old.txt").write_bytes(b"one\rtwo\nthree\n")
        self.assertEqual(self.ws.search("three", path="old.txt"), "old.txt:2: three")
        self.assertEqual(self.ws.read_file("old.txt", offset=2), "three")

    def test_search_does_not_count_as_the_read_before_an_edit(self) -> None:
        self.ws.search("alpha")
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("a.txt", "alpha", "ALPHA")
        self.assertIn("read a.txt before editing it", str(cm.exception))


class WriteFileTest(WorkspaceTest):
    def test_create_builds_parent_directories(self) -> None:
        result = self.ws.write_file("new/sub.txt", "content\n")
        self.assertTrue(result.startswith("wrote 8 bytes to new/sub.txt"))
        self.assertEqual((self.root / "new" / "sub.txt").read_text(), "content\n")

    def test_overwrite_requires_reading_first(self) -> None:
        (self.root / "f.txt").write_text("existing\n")
        with self.assertRaises(ToolError) as cm:
            self.ws.write_file("f.txt", "replacement\n")
        self.assertIn("read f.txt before overwriting it", str(cm.exception))

    def test_identical_content_is_refused(self) -> None:
        (self.root / "f.txt").write_text("same\n")
        self.ws.read_file("f.txt")
        with self.assertRaises(ToolError) as cm:
            self.ws.write_file("f.txt", "same\n")
        self.assertIn("nothing to change", str(cm.exception))

    def test_null_bytes_and_non_strings_are_refused(self) -> None:
        with self.assertRaises(ToolError):
            self.ws.write_file("f.txt", "a\x00b")
        with self.assertRaises(ToolError):
            self.ws.write_file("f.txt", 123)

    def test_directory_path_is_refused(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.write_file(".", "x")
        self.assertIn("path is a directory", str(cm.exception))

    def test_write_size_cap(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.write_file("big.txt", "a" * (MAX_WRITE_BYTES + 1))
        self.assertIn("content is too large", str(cm.exception))
        self.assertFalse((self.root / "big.txt").exists())

    def test_a_declining_approver_leaves_the_file_untouched(self) -> None:
        asked = []

        def decline(message, diff_lines, risk_note=None):
            asked.append((message, diff_lines, risk_note))
            return False

        ws = make_workspace(self.root, approver=decline)
        with self.assertRaises(ToolError) as cm:
            ws.write_file("declined.txt", "content\n")
        self.assertIn("the user declined this write", str(cm.exception))
        self.assertFalse((self.root / "declined.txt").exists())
        message, diff_lines, risk_note = asked[0]
        self.assertIn("declined.txt", message)
        self.assertTrue(any(line.startswith("+") for line in diff_lines))
        self.assertIsNone(risk_note)

    def test_build_entry_writes_carry_the_exec_risk_note(self) -> None:
        seen = {}

        def record(message, diff_lines, risk_note=None):
            seen.update(message=message, risk_note=risk_note)
            return False

        ws = make_workspace(self.root, approver=record)
        with self.assertRaises(ToolError):
            ws.write_file("Makefile", "all:\n")
        self.assertIn("Makefile", seen["message"])
        self.assertIn("may be executed", seen["risk_note"])


class EditFileTest(WorkspaceTest):
    def setUp(self) -> None:
        super().setUp()
        (self.root / "f.txt").write_text("alpha\nbeta\n")
        self.ws.read_file("f.txt")

    def test_unique_snippet_is_replaced(self) -> None:
        self.ws.edit_file("f.txt", "beta", "gamma")
        self.assertEqual((self.root / "f.txt").read_text(), "alpha\ngamma\n")

    def test_crlf_survives_the_round_trip(self) -> None:
        (self.root / "win.txt").write_bytes(b"alpha\r\nbeta\r\n")
        self.ws.read_file("win.txt")
        self.ws.edit_file("win.txt", "beta", "gamma")
        self.assertEqual((self.root / "win.txt").read_bytes(), b"alpha\r\ngamma\r\n")

    def test_cp1252_encoding_is_kept(self) -> None:
        (self.root / "fr.txt").write_bytes(b"caf\xe9 noir\n")
        self.ws.read_file("fr.txt")
        self.ws.edit_file("fr.txt", "noir", "latte")
        self.assertEqual((self.root / "fr.txt").read_bytes(), b"caf\xe9 latte\n")

    def test_unrepresentable_character_is_refused(self) -> None:
        (self.root / "fr.txt").write_bytes(b"caf\xe9 noir\n")
        self.ws.read_file("fr.txt")
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("fr.txt", "noir", "latte →")
        self.assertIn("cannot represent", str(cm.exception))

    def test_must_read_before_editing(self) -> None:
        (self.root / "unread.txt").write_text("data\n")
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("unread.txt", "data", "DATA")
        self.assertIn("read unread.txt before editing it", str(cm.exception))

    def test_missing_file_suggests_write_file(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("gone.txt", "a", "b")
        self.assertIn("does not exist", str(cm.exception))
        self.assertIn("use write_file", str(cm.exception))

    def test_zero_and_multiple_matches_are_refused(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("f.txt", "nope", "x")
        self.assertIn("does not appear in f.txt", str(cm.exception))
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("f.txt", "a", "b")  # twice inside 'alpha', once in 'beta'
        self.assertIn("appears 3 times in f.txt", str(cm.exception))

    def test_empty_and_identical_edits_are_refused(self) -> None:
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("f.txt", "", "x")
        self.assertIn("old_text is empty", str(cm.exception))
        with self.assertRaises(ToolError) as cm:
            self.ws.edit_file("f.txt", "beta", "beta")
        self.assertIn("nothing to change", str(cm.exception))


class AppendFileTest(WorkspaceTest):
    def test_append_creates_and_extends(self) -> None:
        self.ws.append_file("log.txt", "first\n")
        self.ws.append_file("log.txt", "second\n")
        self.assertEqual((self.root / "log.txt").read_text(), "first\nsecond\n")

    def test_a_missing_newline_is_inserted_before_appending(self) -> None:
        (self.root / "f.txt").write_text("no trailing newline")
        self.ws.read_file("f.txt")
        self.ws.append_file("f.txt", "tail\n")
        self.assertEqual((self.root / "f.txt").read_text(), "no trailing newline\ntail\n")

    def test_must_read_before_appending(self) -> None:
        (self.root / "f.txt").write_text("data\n")
        with self.assertRaises(ToolError) as cm:
            self.ws.append_file("f.txt", "more\n")
        self.assertIn("read f.txt before appending to it", str(cm.exception))

    def test_empty_append_is_refused(self) -> None:
        (self.root / "f.txt").write_text("data\n")
        self.ws.read_file("f.txt")
        with self.assertRaises(ToolError) as cm:
            self.ws.append_file("f.txt", "")
        self.assertIn("nothing to change", str(cm.exception))


class ErrorTextTest(WorkspaceTest):
    def test_scrub_paths_rewrites_the_root_and_outside_paths(self) -> None:
        # the workspace's own root is the resolved spelling, so use it here too
        inside = self.ws.scrub_paths(f"[Errno 2] No such file: '{self.ws.root / 'missing.txt'}'")
        self.assertNotIn(str(self.ws.root), inside)
        self.assertIn("./missing.txt", inside)

        sibling = self.ws.root.parent / "outside.txt"
        outside = self.ws.scrub_paths(f"[Errno 2] No such file: '{sibling}'")
        self.assertNotIn(str(sibling), outside)
        self.assertIn(os.path.relpath(sibling, self.ws.root), outside)

    def test_bare_slash_words_in_prose_are_not_treated_as_paths(self) -> None:
        # a single component after the separator ('/talk') is a slash-word of
        # prose, not an absolute path; rewriting it would mangle the message the
        # model sees and print an outside-path warning at the console
        text = "Error: writes are disabled: this message is read-only (sent with /talk or /plan)."
        self.assertEqual(self.ws.scrub_paths(text), text)

    def test_exec_risk_flags_build_entry_files(self) -> None:
        # the note is fixed text; what varies is whether it fires at all
        self.assertIn("may be executed", self.ws.exec_risk(self.ws.root / "Makefile"))
        self.assertIn("may be executed", self.ws.exec_risk(self.ws.root / "venv" / "lib.py"))
        self.assertIn("may be executed", self.ws.exec_risk(self.ws.root / "node_modules" / "x.js"))
        self.assertIsNone(self.ws.exec_risk(self.ws.root / "docs" / "readme.md"))


class BoundedReadsTest(WorkspaceTest):
    """No file larger than the read cap (MAX_FILE_BYTES) is ever loaded whole.

    The byte and page caps bound what is returned, not what is read, so the
    size is checked before the read: search skips such files with a note,
    read_file refuses them, and an edit into a file that has grown past the
    cap on disk is refused too. The cap is patched small here, so no test
    needs a huge file on disk.
    """

    def test_search_skips_oversized_files_with_a_note(self) -> None:
        (self.root / "big.log").write_text("needle\n" + "x" * 200)
        with mock.patch("core.workspace.MAX_FILE_BYTES", 64):
            result = self.ws.search("needle")
        self.assertNotIn("big.log", result)
        self.assertIn("(no matches)", result)
        self.assertIn("(skipped 1 file larger than the 64-byte read cap)", result)

    def test_read_file_refuses_oversized_files(self) -> None:
        (self.root / "big.log").write_text("needle\n" + "x" * 200)
        with mock.patch("core.workspace.MAX_FILE_BYTES", 64):
            with self.assertRaises(ToolError) as cm:
                self.ws.read_file("big.log", offset=1, limit=1)
            self.assertIn("over the 64-byte cap", str(cm.exception))

    def test_an_edit_into_a_file_past_the_cap_is_refused(self) -> None:
        (self.root / "f.txt").write_text("small\n")
        self.ws.read_file("f.txt")
        (self.root / "f.txt").write_text("grown\n" + "x" * 200)
        with mock.patch("core.workspace.MAX_FILE_BYTES", 64):
            with self.assertRaises(ToolError) as cm:
                self.ws.edit_file("f.txt", "grown", "big")
            self.assertIn("has grown to", str(cm.exception))

    def test_the_walk_stops_collecting_at_the_file_cap_with_a_note(self) -> None:
        for i in range(5):
            (self.root / f"f{i}.txt").write_text("needle\n")
        with mock.patch("core.workspace.MAX_SEARCH_FILES", 3):
            result = self.ws.search("needle")
        matched = [line for line in result.splitlines() if ":1: " in line]
        self.assertEqual(len(matched), 3)
        self.assertIn("(the walk stopped at 3 files", result)


class LineSplittingTest(unittest.TestCase):
    """split_file_lines: the one line numbering shared by search and read_file.

    Only the newline breaks lines: str.splitlines() would also break on \r,
    \f, \v and the Unicode separators, and the two tools would then disagree
    about where the lines are.
    """

    def test_only_the_newline_breaks_lines(self) -> None:
        self.assertEqual(split_file_lines("a\r\nb\rc\fd\ne\n"), ["a", "b\rc\x0cd", "e"])
        self.assertEqual(split_file_lines("x"), ["x"])
        self.assertEqual(split_file_lines(""), [])

    def test_the_trailing_newline_is_layout_not_a_line(self) -> None:
        self.assertEqual(split_file_lines("one\n\n"), ["one", ""])


class WindowsNameTest(unittest.TestCase):
    """windows_name_error: the device-name and alternate-stream refusals.

    resolve() applies these to every path on Windows (os.name == 'nt'), where
    naming CON, NUL or 'file.txt:stream' would make an approved write land
    somewhere the diff did not show; the helper is tested directly so the
    suite passes on every platform.
    """

    def test_reserved_device_names_are_refused_in_any_spelling(self) -> None:
        for part in ("CON", "con", "NUL.txt", "COM1 ", "aux.dat."):
            self.assertIn("reserved Windows device name", windows_name_error((part,)))

    def test_colons_are_refused(self) -> None:
        self.assertIn("alternate-data-stream", windows_name_error(("a.txt:hidden",)))

    def test_ordinary_names_pass(self) -> None:
        self.assertIsNone(windows_name_error(("readme.md", "sub", "setup.py")))
        self.assertIsNone(windows_name_error(()))


if __name__ == "__main__":
    unittest.main()
