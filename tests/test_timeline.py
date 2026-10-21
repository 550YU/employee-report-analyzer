"""Tests for timeline confirmation workflow."""
import io
import sys
import unittest

from zhrreport.models import TimeRange, TranscriptSegment
from zhrreport.timeline import TimeRangeRequired, confirm_time_range


def _segs():
    return [
        TranscriptSegment(0.0,   5.0,  "Alice", "Hi",    0.9),
        TranscriptSegment(6.0,  10.0,  "Bob",   "Hello", 0.9),
        TranscriptSegment(11.0, 20.0,  "Alice", "Bye",   0.9),
    ]


class TestConfirmTimeRange(unittest.TestCase):
    def test_explicit_range_str(self):
        tr = confirm_time_range(
            segments=_segs(),
            recording_duration=20.0,
            range_str="00:00:00-00:00:10",
        )
        self.assertAlmostEqual(tr.start, 0.0)
        self.assertAlmostEqual(tr.end, 10.0)

    def test_confirm_full(self):
        tr = confirm_time_range(
            segments=_segs(),
            recording_duration=20.0,
            confirm_full=True,
        )
        self.assertAlmostEqual(tr.start, 0.0)
        self.assertAlmostEqual(tr.end, 20.0)

    def test_no_range_non_interactive_raises(self):
        """With no range and non-interactive session, must raise TimeRangeRequired."""
        with self.assertRaises(TimeRangeRequired) as ctx:
            confirm_time_range(
                segments=_segs(),
                recording_duration=20.0,
                interactive=False,
            )
        self.assertIn("--confirm-full", str(ctx.exception))

    def test_range_str_takes_priority_over_confirm_full(self):
        tr = confirm_time_range(
            segments=_segs(),
            recording_duration=20.0,
            range_str="5-15",
            confirm_full=True,
        )
        self.assertAlmostEqual(tr.start, 5.0)
        self.assertAlmostEqual(tr.end, 15.0)

    def test_interactive_full_choice(self):
        """Simulate user typing 'A' to select full recording."""
        original_stdin = sys.stdin
        try:
            sys.stdin = io.TextIOWrapper(
                io.BytesIO(b"A\n"), encoding="utf-8"
            )
            tr = confirm_time_range(
                segments=_segs(),
                recording_duration=20.0,
                interactive=True,
            )
            self.assertAlmostEqual(tr.start, 0.0)
            self.assertAlmostEqual(tr.end, 20.0)
        finally:
            sys.stdin = original_stdin

    def test_interactive_custom_range(self):
        """Simulate user typing a custom range string."""
        original_stdin = sys.stdin
        try:
            sys.stdin = io.TextIOWrapper(
                io.BytesIO(b"5-15\n"), encoding="utf-8"
            )
            tr = confirm_time_range(
                segments=_segs(),
                recording_duration=20.0,
                interactive=True,
            )
            self.assertAlmostEqual(tr.start, 5.0)
            self.assertAlmostEqual(tr.end, 15.0)
        finally:
            sys.stdin = original_stdin


if __name__ == "__main__":
    unittest.main()
