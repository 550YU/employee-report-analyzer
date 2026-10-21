"""Tests for ChunkedTranscriber (including hard-timeout kill)."""
import subprocess
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from zhrreport.chunker import ChunkedTranscriber
from zhrreport.models import TimeRange


class TestChunkedTranscriber(unittest.TestCase):
    def test_fixture_full_run(self):
        """Full chunked run with fixture backend returns ordered segments."""
        ct = ChunkedTranscriber(
            backend_name="fixture",
            chunk_duration=200.0,
            timeout=60.0,
        )
        result = ct.run(source="ignored", full_duration=780.0)
        self.assertGreater(len(result.segments), 0)
        # Segments should be in ascending time order
        starts = [s.start for s in result.segments]
        self.assertEqual(starts, sorted(starts))

    def test_fixture_with_time_range(self):
        """ChunkedTranscriber respects a time range."""
        ct = ChunkedTranscriber(
            backend_name="fixture",
            chunk_duration=100.0,
            timeout=60.0,
        )
        tr = TimeRange(0.0, 100.0)
        result = ct.run(source="ignored", full_duration=780.0, time_range=tr)
        for seg in result.segments:
            self.assertLess(seg.start, tr.end)

    def test_make_chunks_exact_division(self):
        ct = ChunkedTranscriber("fixture", chunk_duration=300.0)
        chunks = ct._make_chunks(TimeRange(0.0, 600.0))
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0].start, 0.0)
        self.assertAlmostEqual(chunks[0].end, 300.0)
        self.assertAlmostEqual(chunks[1].start, 300.0)
        self.assertAlmostEqual(chunks[1].end, 600.0)

    def test_make_chunks_remainder(self):
        ct = ChunkedTranscriber("fixture", chunk_duration=300.0)
        chunks = ct._make_chunks(TimeRange(0.0, 700.0))
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[2].end, 700.0)

    def test_make_chunks_smaller_than_chunk_dur(self):
        ct = ChunkedTranscriber("fixture", chunk_duration=300.0)
        chunks = ct._make_chunks(TimeRange(0.0, 100.0))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].end, 100.0)

    def test_timeout_kills_worker(self):
        """The actual chunk worker timeout path must kill and fail closed."""
        from zhrreport.chunker import ChunkTranscriptionError

        original_popen = subprocess.Popen

        def slow_worker(*_args, **kwargs):
            return original_popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=kwargs.get("stdout"),
                stderr=kwargs.get("stderr"),
                creationflags=kwargs.get("creationflags", 0),
                start_new_session=kwargs.get("start_new_session", False),
            )

        ct = ChunkedTranscriber("fixture", chunk_duration=10, timeout=0.1)
        started = time.monotonic()
        with patch("zhrreport.chunker.subprocess.Popen", side_effect=slow_worker):
            with self.assertRaises(ChunkTranscriptionError):
                ct.run("ignored", full_duration=10)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_backend_name_propagated(self):
        ct = ChunkedTranscriber("fixture")
        self.assertEqual(ct.backend_name, "fixture")

    def test_killed_chunk_cleanup_is_scoped(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                temp_dir = os.path.join(tmp, ".zhr_tmp")
                os.makedirs(temp_dir)
                owned = os.path.join(temp_dir, "run_abc_chunk_3_10_123.wav")
                unrelated = os.path.join(temp_dir, "run_xyz_chunk_3_10_123.wav")
                with open(owned, "wb") as fh:
                    fh.write(b"owned")
                with open(unrelated, "wb") as fh:
                    fh.write(b"unrelated")

                ChunkedTranscriber._cleanup_killed_chunk_temp_files("abc", 3)

                self.assertFalse(os.path.exists(owned))
                self.assertTrue(os.path.exists(unrelated))
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
