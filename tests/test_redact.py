"""Tests for core.redact: the redaction config loader and the filter itself."""

import json
import tempfile
import unittest
from pathlib import Path

from core.redact import REDACTED, Redactor, load_redactions


class LoadRedactionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "redact.json"
        self.addCleanup(self._tmp.cleanup)

    def write(self, payload) -> None:
        self.path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")

    def test_missing_file_is_the_silent_opt_out(self) -> None:
        self.assertEqual(load_redactions(self.path), ([], REDACTED, None))

    def test_valid_config(self) -> None:
        self.write({"redactList": ["s1", "", "s2"], "replaceWith": "[X]"})
        self.assertEqual(load_redactions(self.path), (["s1", "s2"], "[X]", None))

    def test_marker_defaults_and_empty_strings_are_dropped(self) -> None:
        self.write({"redactList": ["a", ""]})
        entries, marker, error = load_redactions(self.path)
        self.assertEqual((entries, marker, error), (["a"], REDACTED, None))

    def test_utf8_bom_is_tolerated(self) -> None:
        self.path.write_bytes(json.dumps({"redactList": ["a"]}).encode("utf-8-sig"))
        self.assertEqual(load_redactions(self.path), (["a"], REDACTED, None))

    def test_invalid_json(self) -> None:
        self.write("{oops")
        entries, marker, error = load_redactions(self.path)
        self.assertEqual((entries, marker), ([], REDACTED))
        self.assertIn("not valid JSON", error)

    def test_wrong_shapes_are_reported(self) -> None:
        for payload, note in (
            ('["a"]', "is not a JSON object"),
            ("{}", 'needs a "redactList"'),
            ('{"redactList": "a"}', 'needs a "redactList"'),
            ('{"redactList": [1]}', "must be a list of strings"),
            ('{"redactList": [], "replaceWith": ""}', "must be a non-empty string"),
        ):
            with self.subTest(payload=payload):
                self.write(payload)
                entries, _marker, error = load_redactions(self.path)
                self.assertEqual(entries, [])
                self.assertIn(note, error)


class RedactorTest(unittest.TestCase):
    def test_replaces_each_entry_with_the_marker(self) -> None:
        self.assertEqual(Redactor(["cat"]).redact("the cat sat"), "the <redacted> sat")

    def test_longest_entry_wins(self) -> None:
        self.assertEqual(Redactor(["key", "keyword"]).redact("a keyword and a key"), "a <redacted> and a <redacted>")

    def test_the_marker_is_never_rescanned(self) -> None:
        self.assertEqual(Redactor(["redacted"]).redact("my redacted words"), "my <redacted> words")

    def test_custom_marker(self) -> None:
        self.assertEqual(Redactor(["x"], "[GONE]").redact("x y x"), "[GONE] y [GONE]")

    def test_matching_is_case_sensitive(self) -> None:
        self.assertEqual(Redactor(["secret"]).redact("Secret secret"), "Secret <redacted>")

    def test_an_empty_redactor_passes_text_through(self) -> None:
        empty = Redactor([])
        self.assertEqual(len(empty), 0)
        self.assertFalse(empty)
        self.assertEqual(empty.redact("anything"), "anything")

    def test_duplicate_entries_are_counted_once(self) -> None:
        self.assertEqual(len(Redactor(["a", "a", "b"])), 2)

    def test_redact_content_walks_every_shape(self) -> None:
        red = Redactor(["tok"])
        self.assertEqual(red.redact_content("a tok"), "a <redacted>")
        self.assertEqual(red.redact_content(["tok", 1, None]), ["<redacted>", 1, None])
        self.assertEqual(red.redact_content(("tok",)), ("<redacted>",))
        # dict values are filtered wherever they sit; dict keys are structural
        # labels ("type", "text") and stay untouched
        self.assertEqual(
            red.redact_content({"type": "tok", "text": "tok"}),
            {"type": "<redacted>", "text": "<redacted>"},
        )
        self.assertEqual(red.redact_content({"tok": "safe"}), {"tok": "safe"})
        self.assertEqual(
            red.redact_content({"a": ["tok", {"b": "tok"}]}), {"a": ["<redacted>", {"b": "<redacted>"}]}
        )
        # non-string leaves cannot carry text and pass through untouched
        self.assertEqual(red.redact_content(b"tok"), b"tok")


if __name__ == "__main__":
    unittest.main()
