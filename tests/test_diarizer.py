"""Tests for diarizer module."""
import unittest

from zhrreport.diarizer import build_diarization, format_timeline
from zhrreport.local_diarizer import (
    AudioDiarizationEntry,
    PyAnnoteDiarizer,
    aggregate_speaker_ranges,
    format_diarization_timeline,
)
from zhrreport.models import TranscriptSegment


def _segs():
    return [
        TranscriptSegment(0.0,   5.0,  "Alice", "Hi",      0.9),
        TranscriptSegment(6.0,  10.0,  "Bob",   "Hello",   0.9),
        TranscriptSegment(11.0, 20.0,  "Alice", "How are", 0.9),
        TranscriptSegment(21.0, 30.0,  "Bob",   "Good",    0.9),
    ]


class TestBuildDiarization(unittest.TestCase):
    def test_correct_speaker_count(self):
        entries = build_diarization(_segs())
        self.assertEqual(len(entries), 2)

    def test_first_seen(self):
        entries = build_diarization(_segs())
        alice = next(e for e in entries if e.speaker == "Alice")
        self.assertAlmostEqual(alice.first_seen, 0.0)

    def test_last_seen(self):
        entries = build_diarization(_segs())
        alice = next(e for e in entries if e.speaker == "Alice")
        self.assertAlmostEqual(alice.last_seen, 20.0)

    def test_total_speaking_duration(self):
        entries = build_diarization(_segs())
        alice = next(e for e in entries if e.speaker == "Alice")
        # Segments: 0-5 (5s) + 11-20 (9s) = 14s
        self.assertAlmostEqual(alice.total_speaking_duration, 14.0)

    def test_segment_count(self):
        entries = build_diarization(_segs())
        bob = next(e for e in entries if e.speaker == "Bob")
        self.assertEqual(bob.segment_count, 2)

    def test_sorted_by_first_seen(self):
        entries = build_diarization(_segs())
        first_seens = [e.first_seen for e in entries]
        self.assertEqual(first_seens, sorted(first_seens))

    def test_empty_segments(self):
        entries = build_diarization([])
        self.assertEqual(entries, [])


class TestFormatTimeline(unittest.TestCase):
    def test_returns_string(self):
        entries = build_diarization(_segs())
        output = format_timeline(entries, 30.0)
        self.assertIsInstance(output, str)

    def test_contains_speaker_names(self):
        entries = build_diarization(_segs())
        output = format_timeline(entries, 30.0)
        self.assertIn("Alice", output)
        self.assertIn("Bob", output)

    def test_contains_duration(self):
        entries = build_diarization(_segs())
        output = format_timeline(entries, 30.0)
        # Duration 30s → 00:00:30
        self.assertIn("00:00:30", output)

    def test_suggested_ranges_present(self):
        entries = build_diarization(_segs())
        output = format_timeline(entries, 30.0)
        self.assertIn("Full recording", output)


class TestFormatAudioDiarizationTimeline(unittest.TestCase):
    def test_speaker_ranges_match_displayed_choice_order(self):
        entries = [
            AudioDiarizationEntry("SPEAKER_00", 0.0, 2.0),
            AudioDiarizationEntry("SPEAKER_00", 5.0, 8.0),
            AudioDiarizationEntry("SPEAKER_01", 3.0, 4.0),
        ]
        ranges = aggregate_speaker_ranges(entries)
        self.assertEqual([r.speaker for r in ranges], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual((ranges[0].start, ranges[0].end), (0.0, 8.0))
        self.assertEqual((ranges[1].start, ranges[1].end), (3.0, 4.0))

    def test_subrange_uses_absolute_bounds(self):
        entries = [AudioDiarizationEntry("SPEAKER_00", 1110.0, 1230.0)]
        output = format_diarization_timeline(
            entries,
            total_duration=120.0,
            backend="numpy_acoustic",
            range_start=1110.0,
        )
        self.assertIn("Analyzed range", output)
        self.assertIn("00:18:30 - 00:20:30", output)

    def test_pyannote_reads_environment_token(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "env-token"}):
            self.assertEqual(PyAnnoteDiarizer()._token, "env-token")
            self.assertEqual(PyAnnoteDiarizer("explicit")._token, "explicit")


if __name__ == "__main__":
    unittest.main()
