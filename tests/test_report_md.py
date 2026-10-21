"""Tests for Markdown report generation."""
import os
import unittest

from zhrreport.hallucination import detect_hallucinations
from zhrreport.models import TimeRange, TranscriptResult, TranscriptSegment
from zhrreport.report_md import write_markdown_reports
from zhrreport.utils import ensure_dir

_OUT = os.path.join(os.path.dirname(__file__), "_md_test_output")


def _make_result():
    segs = [
        TranscriptSegment(0.0,   6.0,  "Alice", "Hello everyone, let's start.", 0.95),
        TranscriptSegment(7.0,  14.0,  "Bob",   "Good morning. Ready.", 0.92),
        TranscriptSegment(15.0, 22.0,  "Alice", "First item on the agenda.", 0.93),
        TranscriptSegment(23.0, 30.0,  "Bob",   "Thank you for watching.", 0.55),
    ]
    flags = detect_hallucinations(segs)
    return TranscriptResult(
        segments=segs, source_file="meeting.wav",
        duration=30.0, backend="fixture", flags=flags,
    )


class TestWriteMarkdownReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_dir(_OUT)

    def test_returns_file_list(self):
        result = _make_result()
        paths = write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_files_created_on_disk(self):
        result = _make_result()
        paths = write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f"Missing: {p}")

    def test_per_speaker_files_present(self):
        result = _make_result()
        paths = write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        names = [os.path.basename(p) for p in paths]
        self.assertIn("Alice.md", names)
        self.assertIn("Bob.md", names)

    def test_summary_file_present(self):
        result = _make_result()
        paths = write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        names = [os.path.basename(p) for p in paths]
        self.assertIn("_summary.md", names)

    def test_alice_report_contains_transcript(self):
        result = _make_result()
        write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        alice_path = os.path.join(_OUT, "Alice.md")
        with open(alice_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Hello everyone", content)
        self.assertIn("First item", content)

    def test_flag_section_in_report(self):
        result = _make_result()
        write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        bob_path = os.path.join(_OUT, "Bob.md")
        with open(bob_path, encoding="utf-8") as f:
            content = f.read()
        # Bob's "Thank you for watching" should be flagged
        self.assertIn("Quality Flags", content)

    def test_summary_contains_all_speakers(self):
        result = _make_result()
        write_markdown_reports(result, TimeRange(0.0, 30.0), _OUT)
        summary_path = os.path.join(_OUT, "_summary.md")
        with open(summary_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Alice", content)
        self.assertIn("Bob", content)

    def test_non_ascii_text_handled(self):
        segs = [
            TranscriptSegment(0.0, 5.0, "张三", "早上好，我们开始吧。", 0.9),
        ]
        result = TranscriptResult(
            segments=segs, source_file="test.wav",
            duration=5.0, backend="fixture",
        )
        paths = write_markdown_reports(result, TimeRange(0.0, 5.0), _OUT)
        self.assertTrue(any(os.path.isfile(p) for p in paths))


if __name__ == "__main__":
    unittest.main()
