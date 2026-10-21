"""Time-range confirmation workflow.

Implements the mandatory step before report generation:

1. If ``--range`` was supplied on the CLI → validate and use it.
2. If ``--confirm-full`` was supplied → use the full recording range.
3. Otherwise → display the diarization-based candidate timeline and:
   a. In interactive mode (TTY) → prompt the user.
   b. In non-interactive mode (no TTY / piped) → raise an explicit
      :class:`TimeRangeRequired` exception so the caller can surface a
      clear error message.  We never silently choose a range.

The prompt accepts:
* ``A`` / ``a`` or ``full`` → full recording.
* A letter from the list (``B``, ``C``, …) → that speaker's range.
* An explicit range string such as ``00:05:00-00:25:00``.
"""
from __future__ import annotations

import sys
from typing import List, Optional

from zhrreport.diarizer import build_diarization, format_timeline
from zhrreport.models import TimeRange, TranscriptSegment


class TimeRangeRequired(RuntimeError):
    """Raised when no time range is provided and stdin is not a TTY."""


def confirm_time_range(
    segments: List[TranscriptSegment],
    recording_duration: float,
    range_str: Optional[str] = None,
    confirm_full: bool = False,
    interactive: Optional[bool] = None,
) -> TimeRange:
    """Return a confirmed :class:`~zhrreport.models.TimeRange`.

    Parameters
    ----------
    segments:
        Transcript segments (used to build diarization).
    recording_duration:
        Total recording duration in seconds.
    range_str:
        Explicit range supplied via ``--range``.
    confirm_full:
        If *True*, use the full recording without prompting.
    interactive:
        Override TTY detection.  ``None`` → detect from ``sys.stdin``.

    Raises
    ------
    TimeRangeRequired
        When no range is provided and the session is non-interactive.
    ValueError
        When *range_str* cannot be parsed.
    """
    # --- Fast path: explicit range supplied ---
    if range_str:
        return TimeRange.from_string(range_str)

    if confirm_full:
        return TimeRange(0.0, recording_duration)

    # --- Need to show timeline and prompt/fail ---
    entries = build_diarization(segments)
    timeline_text = format_timeline(entries, recording_duration)

    print(timeline_text)

    is_tty = sys.stdin.isatty() if interactive is None else interactive

    if not is_tty:
        raise TimeRangeRequired(
            "\nNo time range specified and stdin is not a TTY.\n"
            "Either supply --range START-END  (e.g. --range 00:00:00-00:30:00)\n"
            "or pass --confirm-full to process the entire recording.\n"
            "Example:\n"
            "  python -m zhrreport.cli report meeting.json --confirm-full\n"
            "  python -m zhrreport.cli report meeting.json --range 00:00-30:00"
        )

    # --- Interactive prompt ---
    print()
    print("Enter a range to process:")
    print("  [A] Full recording")
    for i, e in enumerate(entries, ord('B')):
        label = chr(i)
        print(f"  [{label}] {e.speaker} ({e.first_seen:.0f}s – {e.last_seen:.0f}s)")
    print("  Or type a custom range, e.g.  00:05:00-00:25:00")
    print()

    while True:
        try:
            answer = input("Your choice: ").strip()
        except EOFError:
            raise TimeRangeRequired("EOF on stdin — cannot prompt for time range.")

        if not answer:
            continue

        upper = answer.upper()
        if upper in ("A", "FULL"):
            return TimeRange(0.0, recording_duration)

        # Letter choice: B → entries[0], C → entries[1], …
        if len(upper) == 1 and upper.isalpha():
            idx = ord(upper) - ord('B')
            if 0 <= idx < len(entries):
                e = entries[idx]
                return TimeRange(e.first_seen, e.last_seen)

        # Try to parse as a range string
        try:
            return TimeRange.from_string(answer)
        except ValueError as exc:
            print(f"  Cannot parse {answer!r}: {exc}. Try again.")
