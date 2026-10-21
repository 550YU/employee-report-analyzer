"""Tests for hallucination detection."""
import unittest

from zhrreport.hallucination import detect_hallucinations, summarise_flags
from zhrreport.models import TranscriptSegment


def _seg(start, end, speaker, text, confidence=0.9):
    return TranscriptSegment(start=start, end=end, speaker=speaker,
                             text=text, confidence=confidence)


class TestDetectHallucinations(unittest.TestCase):
    def test_no_flags_on_clean_transcript(self):
        segs = [
            _seg(0.0,  5.0,  "Alice", "Good morning everyone.", 0.95),
            _seg(6.0,  12.0, "Bob",   "Hello Alice, ready to start.", 0.93),
            _seg(13.0, 20.0, "Alice", "Let us begin the review.", 0.96),
        ]
        flags = detect_hallucinations(segs)
        # May have 'uniform_confidence' flag but no high-severity flags
        high_flags = [f for f in flags if f.severity == "high"]
        self.assertEqual(high_flags, [])

    def test_known_phrase_detected(self):
        segs = [
            _seg(0.0, 5.0, "Alice", "Thank you for watching.", 0.65),
        ]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertIn("known_phrase", types)

    def test_known_phrase_case_insensitive(self):
        segs = [
            _seg(0.0, 5.0, "Bob", "THANK YOU FOR WATCHING.", 0.65),
        ]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertIn("known_phrase", types)

    def test_high_word_rate_detected(self):
        # 20 words in 0.5 seconds = 40 wps >> MAX_WORDS_PER_SECOND (5.0)
        text = " ".join(["word"] * 20)
        segs = [_seg(0.0, 0.5, "Alice", text, 0.9)]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertIn("high_word_rate", types)

    def test_normal_word_rate_ok(self):
        # 10 words in 5 seconds = 2 wps — OK
        text = " ".join(["word"] * 10)
        segs = [_seg(0.0, 5.0, "Alice", text, 0.9)]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertNotIn("high_word_rate", types)

    def test_low_confidence_flagged(self):
        segs = [_seg(0.0, 5.0, "Alice", "Some text.", 0.50)]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertIn("low_confidence", types)

    def test_confidence_at_threshold_not_flagged(self):
        segs = [_seg(0.0, 5.0, "Alice", "Some text.", 0.70)]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertNotIn("low_confidence", types)

    def test_repeated_segment_detected(self):
        segs = [
            _seg(0.0,  5.0,  "Bob", "Thank you very much.",  0.9),
            _seg(5.0,  10.0, "Bob", "Thank you very much.",  0.9),
        ]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertIn("repeated_segment", types)

    def test_repeated_segment_different_speakers_not_flagged(self):
        segs = [
            _seg(0.0,  5.0,  "Alice", "Same text here.",  0.9),
            _seg(5.0,  10.0, "Bob",   "Same text here.",  0.9),
        ]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertNotIn("repeated_segment", types)

    def test_uniform_confidence_flagged(self):
        segs = [
            _seg(0.0,  5.0,  "Alice", "Hello.", 1.0),
            _seg(5.0,  10.0, "Bob",   "World.", 1.0),
            _seg(10.0, 15.0, "Alice", "Bye.",   1.0),
        ]
        flags = detect_hallucinations(segs)
        types = [f.flag_type for f in flags]
        self.assertIn("uniform_confidence", types)

    def test_empty_segments(self):
        flags = detect_hallucinations([])
        self.assertEqual(flags, [])

    def test_severity_values(self):
        segs = [_seg(0.0, 5.0, "Alice", "Thank you for watching.", 0.4)]
        flags = detect_hallucinations(segs)
        for f in flags:
            self.assertIn(f.severity, ("low", "medium", "high"))

    def test_flags_sorted_by_start(self):
        segs = [
            _seg(10.0, 11.0, "Bob", "Thank you for watching.", 0.4),
            _seg(0.0,  0.2,  "Alice", " ".join(["word"] * 30), 0.9),
        ]
        flags = detect_hallucinations(segs)
        if len(flags) >= 2:
            starts = [f.start for f in flags]
            self.assertEqual(starts, sorted(starts))


class TestSummariseFlags(unittest.TestCase):
    def test_no_flags(self):
        msg = summarise_flags([])
        self.assertIn("No hallucination", msg)

    def test_with_flags(self):
        segs = [
            _seg(0.0, 5.0, "Alice", "Thank you for watching.", 0.4),
        ]
        flags = detect_hallucinations(segs)
        msg = summarise_flags(flags)
        self.assertIn("flag(s)", msg)


if __name__ == "__main__":
    unittest.main()
