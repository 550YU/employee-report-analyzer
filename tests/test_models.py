"""Tests for zhrreport.models"""
import unittest
from zhrreport.models import (
    TranscriptSegment, TimeRange, HallucinationFlag,
    TranscriptResult, SpeakerDiarizationEntry,
)


class TestTranscriptSegment(unittest.TestCase):
    def _make(self, start=0.0, end=5.0, speaker="A", text="Hello world",
              confidence=0.9):
        return TranscriptSegment(start=start, end=end, speaker=speaker,
                                 text=text, confidence=confidence)

    def test_duration(self):
        seg = self._make(start=1.0, end=4.0)
        self.assertAlmostEqual(seg.duration, 3.0)

    def test_duration_negative_clamped(self):
        seg = self._make(start=5.0, end=3.0)
        self.assertEqual(seg.duration, 0.0)

    def test_word_count(self):
        seg = self._make(text="one two three four")
        self.assertEqual(seg.word_count, 4)

    def test_words_per_second(self):
        seg = self._make(start=0.0, end=2.0, text="a b c d")  # 4 words / 2s = 2.0
        self.assertAlmostEqual(seg.words_per_second, 2.0)

    def test_words_per_second_zero_duration(self):
        seg = self._make(start=1.0, end=1.0, text="word")
        self.assertEqual(seg.words_per_second, 0.0)

    def test_roundtrip_dict(self):
        seg = self._make()
        d = seg.to_dict()
        seg2 = TranscriptSegment.from_dict(d)
        self.assertEqual(seg, seg2)

    def test_extra_keys_ignored(self):
        d = {"start": 0.0, "end": 1.0, "speaker": "X", "text": "hi",
             "confidence": 0.9, "unknown_key": "ignored"}
        seg = TranscriptSegment.from_dict(d)
        self.assertEqual(seg.text, "hi")


class TestTimeRange(unittest.TestCase):
    def test_contains(self):
        tr = TimeRange(0.0, 10.0)
        self.assertTrue(tr.contains(5.0))
        self.assertTrue(tr.contains(0.0))
        self.assertTrue(tr.contains(10.0))
        self.assertFalse(tr.contains(10.1))

    def test_overlaps(self):
        tr = TimeRange(5.0, 15.0)
        self.assertTrue(tr.overlaps(3.0, 7.0))   # start before, end inside
        self.assertTrue(tr.overlaps(6.0, 9.0))   # fully inside
        self.assertTrue(tr.overlaps(10.0, 20.0)) # start inside, end after
        self.assertFalse(tr.overlaps(0.0, 5.0))  # ends exactly at start
        self.assertFalse(tr.overlaps(15.0, 20.0))# starts exactly at end

    def test_duration(self):
        tr = TimeRange(10.0, 70.0)
        self.assertAlmostEqual(tr.duration, 60.0)

    def test_from_string_hms(self):
        tr = TimeRange.from_string("00:05:00-00:10:00")
        self.assertAlmostEqual(tr.start, 300.0)
        self.assertAlmostEqual(tr.end, 600.0)

    def test_from_string_ms(self):
        tr = TimeRange.from_string("05:00-10:00")
        self.assertAlmostEqual(tr.start, 300.0)
        self.assertAlmostEqual(tr.end, 600.0)

    def test_from_string_seconds(self):
        tr = TimeRange.from_string("300-600")
        self.assertAlmostEqual(tr.start, 300.0)
        self.assertAlmostEqual(tr.end, 600.0)

    def test_from_string_invalid(self):
        with self.assertRaises(ValueError):
            TimeRange.from_string("not-a-range")

    def test_str(self):
        tr = TimeRange(0.0, 3661.0)
        self.assertEqual(str(tr), "00:00:00-01:01:01")


class TestTranscriptResult(unittest.TestCase):
    def _make_result(self):
        segs = [
            TranscriptSegment(0.0, 5.0, "Alice", "Hello", 0.9),
            TranscriptSegment(5.0, 10.0, "Bob", "World", 0.8),
            TranscriptSegment(10.0, 15.0, "Alice", "Bye", 0.95),
        ]
        return TranscriptResult(
            segments=segs, source_file="test.wav",
            duration=15.0, backend="fixture",
        )

    def test_speakers(self):
        result = self._make_result()
        self.assertEqual(result.speakers(), ["Alice", "Bob"])

    def test_segments_for_speaker(self):
        result = self._make_result()
        alice_segs = result.segments_for_speaker("Alice")
        self.assertEqual(len(alice_segs), 2)
        self.assertTrue(all(s.speaker == "Alice" for s in alice_segs))

    def test_filter_by_range(self):
        result = self._make_result()
        filtered = result.filter_by_range(TimeRange(4.0, 11.0))
        # seg [5,10] fully inside; seg [0,5] overlaps at 4<5 and 5>4 ✔; [10,15] overlaps
        starts = [s.start for s in filtered.segments]
        self.assertIn(0.0, starts)  # overlaps [4,11]
        self.assertIn(5.0, starts)
        self.assertIn(10.0, starts)

    def test_roundtrip_json(self):
        result = self._make_result()
        json_str = result.to_json()
        result2 = TranscriptResult.from_dict(__import__("json").loads(json_str))
        self.assertEqual(len(result.segments), len(result2.segments))
        self.assertEqual(result.duration, result2.duration)


if __name__ == "__main__":
    unittest.main()
