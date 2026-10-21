"""Integration tests for real-audio transcription and diarization.

These tests require:
- The real MP3 file at AUDIO_PATH
- faster-whisper installed (pip install faster-whisper)
- numpy installed (for diarization)
- imageio-ffmpeg or ffmpeg on PATH

All tests are **skipped** automatically when their prerequisites are absent,
with explicit skip messages explaining what to install.
"""
from __future__ import annotations

import os
import sys
import time
import unittest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

AUDIO_PATH = os.environ.get(
    "ZHRREPORT_TEST_AUDIO",
    os.path.join(os.path.dirname(__file__), "fixtures", "integration_audio.mp3"),
)

def _has_audio() -> bool:
    return os.path.isfile(AUDIO_PATH)

def _has_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False

def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False

def _has_ffmpeg() -> bool:
    try:
        from zhrreport.utils import find_ffmpeg
        find_ffmpeg()
        return True
    except Exception:
        return False

_SKIP_NO_AUDIO = unittest.skipUnless(_has_audio(), f"Audio file absent: {AUDIO_PATH}")
_SKIP_NO_FW = unittest.skipUnless(_has_faster_whisper(),
    "faster-whisper not installed. Run: pip install faster-whisper")
_SKIP_NO_NUMPY = unittest.skipUnless(_has_numpy(),
    "numpy not installed. Run: pip install numpy")
_SKIP_NO_FFMPEG = unittest.skipUnless(_has_ffmpeg(),
    "ffmpeg/imageio-ffmpeg not available. Run: pip install imageio-ffmpeg")

_OUT = os.path.join(os.path.dirname(__file__), "_real_audio_output")


# ---------------------------------------------------------------------------
# Helper to run CLI
# ---------------------------------------------------------------------------

def _cli(*argv):
    from zhrreport.cli import main
    return main(list(argv))


# ---------------------------------------------------------------------------
# Test: diarization on transition clip (18:30-20:30)
# ---------------------------------------------------------------------------

@_SKIP_NO_AUDIO
@_SKIP_NO_NUMPY
@_SKIP_NO_FFMPEG
class TestDiarizationTransitionClip(unittest.TestCase):
    """Run local diarization on the KY→TF transition region."""

    OUT = os.path.join(_OUT, "diarization")

    def test_diarize_transition_clip(self):
        """Diarize 18:30–20:30 and persist timeline."""
        from zhrreport.utils import ensure_dir
        ensure_dir(self.OUT)

        t0 = time.monotonic()
        rc = _cli(
            "diarize", AUDIO_PATH,
            "--range", "18:30-20:30",
            "--diarization-backend", "numpy_acoustic",
            "--num-speakers", "2",
            "--output-dir", self.OUT,
        )
        elapsed = time.monotonic() - t0

        self.assertEqual(rc, 0, "diarize command returned non-zero")
        tl_path = os.path.join(self.OUT, "diarization_timeline.txt")
        self.assertTrue(os.path.isfile(tl_path), "Timeline not saved")
        content = open(tl_path, encoding="utf-8").read()
        self.assertIn("SPEAKER", content)
        print(f"\n  Diarization elapsed: {elapsed:.1f}s")
        print(f"  Timeline saved: {tl_path}")
        print(f"\n{content}\n")

    def test_diarize_returns_entries(self):
        """NumpyAcousticDiarizer returns at least 1 turn for a 2-min clip."""
        from zhrreport.local_diarizer import NumpyAcousticDiarizer
        d = NumpyAcousticDiarizer()
        entries = d.diarize(AUDIO_PATH, num_speakers=2, time_range=(1110.0, 1230.0))
        self.assertGreater(len(entries), 0)
        for e in entries:
            self.assertTrue(e.speaker.startswith("SPEAKER"))
            self.assertLess(e.start, e.end)


# ---------------------------------------------------------------------------
# Test: faster-whisper transcription KY range
# ---------------------------------------------------------------------------

@_SKIP_NO_AUDIO
@_SKIP_NO_FW
@_SKIP_NO_FFMPEG
class TestFasterWhisperKY(unittest.TestCase):
    """Transcribe the KY section (05:26–07:26 = 2 min sub-range)."""

    OUT = os.path.join(_OUT, "KY")
    RANGE = "05:26-07:26"
    _elapsed = 0.0
    _rc = -1

    @classmethod
    def setUpClass(cls):
        from zhrreport.utils import ensure_dir
        ensure_dir(cls.OUT)
        # Run transcription once for the whole class
        t0 = time.monotonic()
        cls._rc = _cli(
            "report", AUDIO_PATH,
            "--backend", "faster-whisper",
            "--model", "base",
            "--range", cls.RANGE,
            "--speaker-name", "KY",
            "--chunk-duration", "45",
            "--timeout", "120",
            "--output-dir", cls.OUT,
            "--skip-hallucination",
        )
        cls._elapsed = time.monotonic() - t0
        print(f"\n  KY transcription elapsed: {cls._elapsed:.1f}s")

    def test_ky_transcription_produces_reports(self):
        self.assertEqual(self._rc, 0, f"CLI returned {self._rc} for KY range")

    def test_ky_md_report_exists(self):
        path = os.path.join(self.OUT, "KY.md")
        self.assertTrue(os.path.isfile(path), f"KY.md not found: {path}")
        content = open(path, encoding="utf-8").read()
        self.assertIn("KY", content)
        print(f"\n  KY.md size: {len(content)} chars")

    def test_ky_docx_report_exists(self):
        import zipfile
        path = os.path.join(self.OUT, "KY.docx")
        self.assertTrue(os.path.isfile(path), f"KY.docx not found: {path}")
        self.assertTrue(zipfile.is_zipfile(path), "KY.docx is not a valid ZIP")

    def test_ky_has_transcript_segments(self):
        md_path = os.path.join(self.OUT, "KY.md")
        if not os.path.isfile(md_path):
            self.skipTest("KY.md not present")
        content = open(md_path, encoding="utf-8").read()
        self.assertGreater(len(content), 500, "KY.md has too little content")

    def test_ky_elapsed_reasonable(self):
        """2-minute transcription should finish in under 120 seconds."""
        self.assertLess(self._elapsed, 120.0,
                        f"KY transcription took {self._elapsed:.1f}s > 120s")


# ---------------------------------------------------------------------------
# Test: faster-whisper transcription TF range
# ---------------------------------------------------------------------------

@_SKIP_NO_AUDIO
@_SKIP_NO_FW
@_SKIP_NO_FFMPEG
class TestFasterWhisperTF(unittest.TestCase):
    """Transcribe the TF section (19:32–21:32 = 2 min sub-range)."""

    OUT = os.path.join(_OUT, "TF")
    RANGE = "19:32-21:32"
    _elapsed = 0.0
    _rc = -1

    @classmethod
    def setUpClass(cls):
        from zhrreport.utils import ensure_dir
        ensure_dir(cls.OUT)
        t0 = time.monotonic()
        cls._rc = _cli(
            "report", AUDIO_PATH,
            "--backend", "faster-whisper",
            "--model", "base",
            "--range", cls.RANGE,
            "--speaker-name", "TF",
            "--chunk-duration", "45",
            "--timeout", "120",
            "--output-dir", cls.OUT,
            "--skip-hallucination",
        )
        cls._elapsed = time.monotonic() - t0
        print(f"\n  TF transcription elapsed: {cls._elapsed:.1f}s")

    def test_tf_transcription_produces_reports(self):
        self.assertEqual(self._rc, 0, f"CLI returned {self._rc} for TF range")

    def test_tf_md_report_exists(self):
        path = os.path.join(self.OUT, "TF.md")
        self.assertTrue(os.path.isfile(path), f"TF.md not found: {path}")
        content = open(path, encoding="utf-8").read()
        self.assertIn("TF", content)

    def test_tf_docx_report_valid(self):
        import zipfile
        path = os.path.join(self.OUT, "TF.docx")
        self.assertTrue(os.path.isfile(path), f"TF.docx not found: {path}")
        self.assertTrue(zipfile.is_zipfile(path))

    def test_tf_has_transcript_segments(self):
        md_path = os.path.join(self.OUT, "TF.md")
        if not os.path.isfile(md_path):
            self.skipTest("TF.md not present")
        content = open(md_path, encoding="utf-8").read()
        self.assertGreater(len(content), 500, "TF.md has too little content")

    def test_tf_elapsed_reasonable(self):
        self.assertLess(self._elapsed, 120.0,
                        f"TF transcription took {self._elapsed:.1f}s > 120s")


# ---------------------------------------------------------------------------
# Test: faster-whisper backend unit (temp WAV cleanup)
# ---------------------------------------------------------------------------

@_SKIP_NO_AUDIO
@_SKIP_NO_FW
@_SKIP_NO_FFMPEG
class TestFasterWhisperBackendUnit(unittest.TestCase):
    """Unit tests for FasterWhisperTranscriber."""

    def test_transcribe_chunk_returns_segments(self):
        from zhrreport.transcribers.faster_whisper_backend import FasterWhisperTranscriber
        from zhrreport.models import TimeRange
        b = FasterWhisperTranscriber(model="base", speaker_name="TESTSP")
        # Small 30-second slice
        segs = b.transcribe_chunk(AUDIO_PATH, time_range=TimeRange(326.0, 356.0), chunk_id=0)
        self.assertIsInstance(segs, list)
        # All segments should be within [326, 356] range
        for s in segs:
            self.assertGreaterEqual(s.start, 325.0)
            self.assertLessEqual(s.end, 360.0)
            self.assertEqual(s.speaker, "TESTSP")

    def test_temp_wav_cleaned_up(self):
        """No stale .zhr_tmp WAV files left after transcription."""
        from zhrreport.transcribers.faster_whisper_backend import FasterWhisperTranscriber
        from zhrreport.models import TimeRange
        tmp_dir = os.path.join(_PROJ, ".zhr_tmp")
        before = set(os.listdir(tmp_dir)) if os.path.exists(tmp_dir) else set()

        b = FasterWhisperTranscriber(model="base")
        b.transcribe_chunk(AUDIO_PATH, time_range=TimeRange(326.0, 341.0), chunk_id=99)

        after = set(os.listdir(tmp_dir)) if os.path.exists(tmp_dir) else set()
        leftover = after - before
        self.assertEqual(leftover, set(), f"Stale temp files: {leftover}")

    def test_hallucination_dedup(self):
        """Hallucination dedup removes known bad phrases and consecutive identical."""
        from zhrreport.transcribers.faster_whisper_backend import (
            _dedup_segments, _segment_is_hallucination
        )
        from zhrreport.models import TranscriptSegment

        # Check hallucination detection
        self.assertTrue(_segment_is_hallucination("thank you for watching", 3.0, 0.1))
        self.assertFalse(_segment_is_hallucination("Hello everyone", 5.0, 0.05))
        # High no_speech_prob → hallucination regardless of text
        self.assertTrue(_segment_is_hallucination("some words", 3.0, 0.9))

        # _dedup_segments removes CONSECUTIVE identical segments
        segs = [
            TranscriptSegment(0.0, 5.0, "SP", "Hello everyone.", 0.9),
            TranscriptSegment(5.0, 8.0, "SP", "Hello everyone.", 0.9),  # consecutive dup
            TranscriptSegment(8.0, 11.0, "SP", "Welcome to the meeting.", 0.9),
            TranscriptSegment(11.0, 14.0, "SP", "Hello everyone.", 0.9),  # not consecutive
        ]
        result = _dedup_segments(segs)
        texts = [s.text for s in result]
        # First "Hello everyone." kept, second (consecutive) removed
        # Third "Hello everyone." kept (not consecutive with first)
        self.assertEqual(texts.count("Hello everyone."), 2)
        self.assertIn("Welcome to the meeting.", texts)

    def test_fully_rejected_primary_escalates(self):
        from zhrreport.transcribers.faster_whisper_backend import FasterWhisperTranscriber
        backend = FasterWhisperTranscriber(model="base")
        self.assertTrue(backend._needs_escalation([], rejected_count=2))
        self.assertFalse(backend._needs_escalation([], rejected_count=0))


# ---------------------------------------------------------------------------
# Test: CLI diarize sub-command (no-range flow)
# ---------------------------------------------------------------------------

@_SKIP_NO_AUDIO
@_SKIP_NO_NUMPY
@_SKIP_NO_FFMPEG
class TestCLIDiarizeCommand(unittest.TestCase):
    """Test the `diarize` CLI sub-command."""

    OUT = os.path.join(_OUT, "diar_cmd")

    def test_diarize_command_exit_zero(self):
        from zhrreport.utils import ensure_dir
        ensure_dir(self.OUT)
        rc = _cli(
            "diarize", AUDIO_PATH,
            "--range", "18:30-20:30",
            "--output-dir", self.OUT,
        )
        self.assertEqual(rc, 0)

    def test_timeline_file_saved(self):
        tl = os.path.join(self.OUT, "diarization_timeline.txt")
        self.assertTrue(os.path.isfile(tl), f"Timeline not saved: {tl}")
        content = open(tl, encoding="utf-8").read()
        self.assertIn("SPEAKER", content)
        self.assertIn("Backend", content)


# ---------------------------------------------------------------------------
# Test: diarizer list
# ---------------------------------------------------------------------------

class TestDiarizerList(unittest.TestCase):
    """Smoke test: list_diarizers returns expected keys."""

    def test_list_diarizers(self):
        from zhrreport.local_diarizer import list_diarizers
        status = list_diarizers()
        self.assertIn("numpy_acoustic", status)
        self.assertIn("speechbrain", status)
        self.assertIn("pyannote", status)
        # numpy_acoustic must be available (numpy is installed)
        if _has_numpy():
            self.assertTrue(status["numpy_acoustic"]["available"])

    def test_get_diarizer_numpy(self):
        from zhrreport.local_diarizer import get_diarizer, NumpyAcousticDiarizer
        if not _has_numpy():
            self.skipTest("numpy not installed")
        d = get_diarizer("numpy_acoustic")
        self.assertIsInstance(d, NumpyAcousticDiarizer)

    def test_get_diarizer_unknown_raises(self):
        from zhrreport.local_diarizer import get_diarizer
        with self.assertRaises(ValueError):
            get_diarizer("nonexistent_xyz_backend")

    def test_speechbrain_missing_raises(self):
        """SpeechBrainDiarizer raises ImportError with install hint when deps absent."""
        try:
            import speechbrain  # noqa: F401
            self.skipTest("speechbrain is installed — skip absence test")
        except ImportError:
            pass
        from zhrreport.local_diarizer import SpeechBrainDiarizer
        d = SpeechBrainDiarizer()
        with self.assertRaises(ImportError) as ctx:
            d.diarize("dummy.wav")
        self.assertIn("pip install", str(ctx.exception))


# ---------------------------------------------------------------------------
# Test: faster-whisper backend (unit, no real audio)
# ---------------------------------------------------------------------------

class TestFasterWhisperUnit(unittest.TestCase):
    """Unit tests that don't require the real audio file."""

    def test_backend_registered(self):
        from zhrreport.transcribers import BACKENDS
        self.assertIn("faster-whisper", BACKENDS)

    def test_backend_describe(self):
        from zhrreport.transcribers.faster_whisper_backend import FasterWhisperTranscriber
        b = FasterWhisperTranscriber(model="tiny")
        desc = b.describe()
        self.assertIn("tiny", desc)
        self.assertIn("int8", desc)

    def test_backends_command_lists_faster_whisper(self):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            from zhrreport.cli import main
            main(["backends"])
        self.assertIn("faster-whisper", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
