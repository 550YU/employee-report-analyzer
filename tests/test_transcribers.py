"""Tests for transcription backends."""
import json
import os
import tempfile
import unittest

from zhrreport.models import TimeRange, TranscriptResult
from zhrreport.transcribers import get_backend, BACKENDS
from zhrreport.transcribers.fixture import FixtureTranscriber
from zhrreport.transcribers.json_backend import JsonTranscriber


class TestFixtureTranscriber(unittest.TestCase):
    def setUp(self):
        self.backend = FixtureTranscriber()

    def test_backend_name(self):
        self.assertEqual(self.backend.backend_name, "fixture")

    def test_returns_segments(self):
        segs = self.backend.transcribe_chunk("ignored_source")
        self.assertGreater(len(segs), 0)

    def test_all_segments_have_required_fields(self):
        segs = self.backend.transcribe_chunk("x")
        for seg in segs:
            self.assertIsInstance(seg.start, float)
            self.assertIsInstance(seg.end, float)
            self.assertIsInstance(seg.speaker, str)
            self.assertIsInstance(seg.text, str)
            self.assertGreaterEqual(seg.confidence, 0.0)
            self.assertLessEqual(seg.confidence, 1.0)

    def test_time_range_filter(self):
        segs_all = self.backend.transcribe_chunk("x")
        tr = TimeRange(0.0, 100.0)
        segs_filtered = self.backend.transcribe_chunk("x", time_range=tr)
        self.assertLess(len(segs_filtered), len(segs_all))
        for seg in segs_filtered:
            self.assertLess(seg.start, tr.end)
            self.assertGreater(seg.end, tr.start)

    def test_chunk_id_set(self):
        segs = self.backend.transcribe_chunk("x", chunk_id=42)
        self.assertTrue(all(s.chunk_id == 42 for s in segs))

    def test_deterministic(self):
        segs1 = self.backend.transcribe_chunk("x")
        segs2 = self.backend.transcribe_chunk("x")
        self.assertEqual(
            [(s.start, s.text) for s in segs1],
            [(s.start, s.text) for s in segs2],
        )


class TestJsonTranscriber(unittest.TestCase):
    def _write_json(self, data: dict) -> str:
        """Write JSON to a temp file, return path."""
        # Use a temp file in the current dir to avoid /tmp
        path = os.path.join(
            os.path.dirname(__file__), "_test_transcript_tmp.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def tearDown(self):
        path = os.path.join(
            os.path.dirname(__file__), "_test_transcript_tmp.json"
        )
        if os.path.exists(path):
            os.remove(path)

    def test_reads_segments(self):
        data = {
            "source_file": "test.wav", "duration": 10.0, "backend": "test",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "A",
                 "text": "hello", "confidence": 0.9},
                {"start": 5.0, "end": 10.0, "speaker": "B",
                 "text": "world", "confidence": 0.85},
            ]
        }
        path = self._write_json(data)
        backend = JsonTranscriber()
        segs = backend.transcribe_chunk(source=path)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].text, "hello")
        self.assertEqual(segs[1].speaker, "B")

    def test_time_range_filter(self):
        data = {
            "source_file": "x.wav", "duration": 20.0, "backend": "t",
            "segments": [
                {"start": 0.0,  "end": 5.0,  "speaker": "A", "text": "a", "confidence": 1.0},
                {"start": 5.0,  "end": 10.0, "speaker": "A", "text": "b", "confidence": 1.0},
                {"start": 10.0, "end": 15.0, "speaker": "A", "text": "c", "confidence": 1.0},
                {"start": 15.0, "end": 20.0, "speaker": "A", "text": "d", "confidence": 1.0},
            ]
        }
        path = self._write_json(data)
        backend = JsonTranscriber()
        segs = backend.transcribe_chunk(path, time_range=TimeRange(4.0, 12.0))
        texts = [s.text for s in segs]
        self.assertIn("a", texts)
        self.assertIn("b", texts)
        self.assertIn("c", texts)
        self.assertNotIn("d", texts)

    def test_missing_file_raises(self):
        backend = JsonTranscriber()
        with self.assertRaises(FileNotFoundError):
            backend.transcribe_chunk("nonexistent_xyz_123.json")

    def test_invalid_json_raises(self):
        path = os.path.join(os.path.dirname(__file__), "_bad_json_tmp.json")
        with open(path, "w") as f:
            f.write("{invalid json}")
        try:
            backend = JsonTranscriber()
            with self.assertRaises((ValueError, Exception)):
                backend.transcribe_chunk(path)
        finally:
            os.remove(path)


class TestGetBackend(unittest.TestCase):
    def test_fixture_backend(self):
        b = get_backend("fixture")
        self.assertIsInstance(b, FixtureTranscriber)

    def test_json_backend(self):
        b = get_backend("json")
        self.assertIsInstance(b, JsonTranscriber)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_backend("nonexistent_backend_xyz")
        self.assertIn("nonexistent_backend_xyz", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
