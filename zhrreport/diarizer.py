"""Speaker diarization analysis.

Takes a list of :class:`~zhrreport.models.TranscriptSegment` objects and
produces per-speaker statistics (first/last appearance, total speaking
duration, segment count) plus a formatted candidate timeline for the
time-range confirmation workflow.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from zhrreport.models import SpeakerDiarizationEntry, TranscriptSegment
from zhrreport.utils import fmt_duration, fmt_sec


def build_diarization(
    segments: List[TranscriptSegment],
) -> List[SpeakerDiarizationEntry]:
    """Compute per-speaker statistics from *segments*.

    Parameters
    ----------
    segments:
        All transcript segments (any time range).

    Returns
    -------
    List[SpeakerDiarizationEntry]
        One entry per speaker, sorted by first appearance.
    """
    first_seen: Dict[str, float] = {}
    last_seen: Dict[str, float] = {}
    total_dur: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)

    for seg in segments:
        sp = seg.speaker
        if sp not in first_seen:
            first_seen[sp] = seg.start
        last_seen[sp] = max(last_seen.get(sp, 0.0), seg.end)
        total_dur[sp] += seg.duration
        count[sp] += 1

    entries = [
        SpeakerDiarizationEntry(
            speaker=sp,
            first_seen=first_seen[sp],
            last_seen=last_seen[sp],
            total_speaking_duration=total_dur[sp],
            segment_count=count[sp],
        )
        for sp in first_seen
    ]
    # Sort by first appearance
    entries.sort(key=lambda e: e.first_seen)
    return entries


def format_timeline(
    entries: List[SpeakerDiarizationEntry],
    recording_duration: float,
) -> str:
    """Return a human-readable timeline string for display to the user.

    Example output::

        ╔══════════════════════════════════════════════════════════════╗
        ║  Recording duration : 00:30:00                               ║
        ║  Speakers detected  : 2                                      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  #  Speaker      First seen   Last seen   Speaking   Segs   ║
        ║  1  Alice        00:00:00     00:30:00    14m 20s    12      ║
        ║  2  Bob          00:00:06     00:30:00    12m 45s    10      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Suggested ranges                                            ║
        ║   A) Full recording     00:00:00 - 00:30:00                  ║
        ║   B) Alice only         00:00:00 - 00:30:00                  ║
        ║   C) Bob only           00:00:06 - 00:30:00                  ║
        ╚══════════════════════════════════════════════════════════════╝
    """
    W = 66  # box width (chars)
    border_top = "╔" + "═" * W + "╗"
    border_mid = "╠" + "═" * W + "╣"
    border_bot = "╚" + "═" * W + "╝"

    def row(text: str) -> str:
        padded = f"  {text}"
        return "║" + padded.ljust(W) + "║"

    lines = [border_top]
    lines.append(row(f"Recording duration : {fmt_sec(recording_duration)}"))
    lines.append(row(f"Speakers detected  : {len(entries)}"))
    lines.append(border_mid)
    lines.append(row(
        f"{'#':<4}{'Speaker':<16}{'First seen':<14}{'Last seen':<14}"
        f"{'Speaking':<12}{'Segs'}"
    ))
    for i, e in enumerate(entries, 1):
        lines.append(row(
            f"{i:<4}{e.speaker:<16}"
            f"{fmt_sec(e.first_seen):<14}"
            f"{fmt_sec(e.last_seen):<14}"
            f"{fmt_duration(e.total_speaking_duration):<12}"
            f"{e.segment_count}"
        ))
    lines.append(border_mid)
    lines.append(row("Suggested ranges"))
    lines.append(row(
        f"  A) Full recording   "
        f"00:00:00 - {fmt_sec(recording_duration)}"
    ))
    for i, e in enumerate(entries, ord('B')):
        lines.append(row(
            f"  {chr(i)}) {e.speaker:<17}"
            f"{fmt_sec(e.first_seen)} - {fmt_sec(e.last_seen)}"
        ))
    lines.append(border_bot)
    return "\n".join(lines)
