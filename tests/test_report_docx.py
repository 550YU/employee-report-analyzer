"""Tests for DOCX report generation and ZIP validity."""
import os
import unittest
import zipfile

from zhrreport.hallucination import detect_hallucinations
from zhrreport.models import TimeRange, TranscriptResult, TranscriptSegment
from zhrreport.report_docx import write_docx_reports, _validate_docx
from zhrreport.utils import ensure_dir

_OUT = os.path.join(os.path.dirname(__file__), "_docx_test_output")

_REQUIRED_PARTS = {
    "[Content_Types].xml",
    "_rels/.rels",
    "word/document.xml",
    "word/_rels/document.xml.rels",
    "word/styles.xml",
}


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


class TestWriteDocxReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_dir(_OUT)

    def test_returns_file_list(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_files_created_on_disk(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f"Missing: {p}")

    def test_per_speaker_docx_present(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        names = [os.path.basename(p) for p in paths]
        self.assertIn("Alice.docx", names)
        self.assertIn("Bob.docx", names)

    def test_summary_docx_present(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        names = [os.path.basename(p) for p in paths]
        self.assertIn("_summary.docx", names)

    def test_docx_are_valid_zip(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        for p in paths:
            self.assertTrue(zipfile.is_zipfile(p), f"Not a zip: {p}")

    def test_docx_contain_required_parts(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        for p in paths:
            with zipfile.ZipFile(p, "r") as zf:
                names = set(zf.namelist())
            missing = _REQUIRED_PARTS - names
            self.assertEqual(
                missing, set(),
                f"{os.path.basename(p)} missing: {missing}"
            )

    def test_document_xml_is_valid_xml(self):
        import xml.etree.ElementTree as ET
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        for p in paths:
            with zipfile.ZipFile(p, "r") as zf:
                doc_xml = zf.read("word/document.xml").decode("utf-8")
            # Should parse without error
            try:
                ET.fromstring(doc_xml)
            except ET.ParseError as exc:
                self.fail(f"word/document.xml in {os.path.basename(p)} is not valid XML: {exc}")

    def test_validate_docx_passes(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        for p in paths:
            # _validate_docx must not raise
            _validate_docx(p)

    def test_validate_docx_bad_zip_raises(self):
        bad_path = os.path.join(_OUT, "bad.docx")
        with open(bad_path, "wb") as f:
            f.write(b"not a zip file at all")
        with self.assertRaises(RuntimeError):
            _validate_docx(bad_path)

    def test_content_in_document_xml(self):
        result = _make_result()
        paths = write_docx_reports(result, TimeRange(0.0, 30.0), _OUT)
        alice_path = next(p for p in paths if os.path.basename(p) == "Alice.docx")
        with zipfile.ZipFile(alice_path, "r") as zf:
            doc_xml = zf.read("word/document.xml").decode("utf-8")
        self.assertIn("Hello everyone", doc_xml)

    def test_non_ascii_speaker_name(self):
        segs = [
            TranscriptSegment(0.0, 5.0, "李四", "大家好，今天开会。", 0.9),
        ]
        result = TranscriptResult(
            segments=segs, source_file="test.wav",
            duration=5.0, backend="fixture",
        )
        paths = write_docx_reports(result, TimeRange(0.0, 5.0), _OUT)
        self.assertTrue(any(os.path.isfile(p) for p in paths))
        for p in paths:
            _validate_docx(p)

    def test_special_xml_chars_escaped(self):
        segs = [
            TranscriptSegment(0.0, 5.0, "Alice",
                              'The <tag> & "quote" isn\'t &real;', 0.9),
        ]
        result = TranscriptResult(
            segments=segs, source_file="test.wav",
            duration=5.0, backend="fixture",
        )
        paths = write_docx_reports(result, TimeRange(0.0, 5.0), _OUT)
        for p in paths:
            with zipfile.ZipFile(p, "r") as zf:
                doc_xml = zf.read("word/document.xml").decode("utf-8")
            import xml.etree.ElementTree as ET
            ET.fromstring(doc_xml)  # Must not raise


if __name__ == "__main__":
    unittest.main()
