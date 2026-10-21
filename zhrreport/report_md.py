"""Markdown report generator.

Produces one ``{speaker}.md`` file per speaker, plus a
``_summary.md`` that lists all speakers with statistics.
"""
from __future__ import annotations

import datetime
import os
from typing import List, Optional

from zhrreport.diarizer import build_diarization
from zhrreport.models import HallucinationFlag, TimeRange, TranscriptResult, TranscriptSegment
from zhrreport.utils import ensure_dir, fmt_duration, fmt_sec, safe_filename

_SEVERITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🔵"}


def _speaker_report(
    speaker: str,
    segments: List[TranscriptSegment],
    all_flags: List[HallucinationFlag],
    time_range: TimeRange,
    source_file: str,
    backend: str,
    generated_at: datetime.datetime,
) -> str:
    """Build the full Markdown report string for a single speaker."""
    speaker_flags = [f for f in all_flags if f.speaker == speaker or f.speaker == "*"]
    total_speaking = sum(s.duration for s in segments)
    word_count = sum(s.word_count for s in segments)

    lines: List[str] = []

    # ---- Header ----
    lines.append(f"# Performance Report — {speaker}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Generated | {generated_at.strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| Source | `{os.path.basename(source_file)}` |")
    lines.append(f"| Backend | `{backend}` |")
    lines.append(f"| Time range | {time_range} |")
    lines.append(f"| Speaking time | {fmt_duration(total_speaking)} |")
    lines.append(f"| Word count | {word_count:,} |")
    lines.append(f"| Segments | {len(segments)} |")
    lines.append(f"| Quality flags | {len(speaker_flags)} |")
    lines.append("")

    # ---- Quality notice ----
    if speaker_flags:
        high = sum(1 for f in speaker_flags if f.severity == "high")
        med  = sum(1 for f in speaker_flags if f.severity == "medium")
        low  = sum(1 for f in speaker_flags if f.severity == "low")
        lines.append("> **⚠️ Quality Notice**")
        parts = []
        if high: parts.append(f"{high} HIGH")
        if med:  parts.append(f"{med} MEDIUM")
        if low:  parts.append(f"{low} LOW")
        lines.append(f"> This transcript has {len(speaker_flags)} flag(s): "
                     + ", ".join(parts) + ".")
        lines.append("> Flagged segments are marked with ⚠️ in the transcript below.")
        lines.append("")

    # ---- Transcript ----
    lines.append("## Transcript")
    lines.append("")

    flagged_idxs = {f.segment_idx for f in speaker_flags
                    if f.flag_type not in ("uniform_confidence",)}

    # Build a map from segment idx in *all* segments to flag messages
    # (segment_idx in flags refers to position in the full filtered list,
    # not just this speaker's segments)
    flag_by_idx: dict = {}
    for f in speaker_flags:
        flag_by_idx.setdefault(f.segment_idx, []).append(f)

    # We need the global index of each speaker segment to match flags
    # Re-derive by matching start time
    start_to_flags: dict = {}
    for f in speaker_flags:
        start_to_flags.setdefault(f.start, []).append(f)

    for seg in segments:
        timestamp = f"**`{fmt_sec(seg.start)}`**"
        seg_flags = start_to_flags.get(seg.start, [])
        if seg_flags:
            flag_icons = " ".join(
                _SEVERITY_EMOJI.get(f.severity, "⚠️") for f in seg_flags
            )
            lines.append(f"### {timestamp} {flag_icons}")
        else:
            lines.append(f"### {timestamp}")
        lines.append("")
        lines.append(f"> {seg.text}")
        if seg.confidence < 0.80:
            lines.append(f"> *(confidence: {seg.confidence:.0%})*")
        lines.append("")

    # ---- Hallucination flags ----
    if speaker_flags:
        lines.append("## Quality Flags")
        lines.append("")
        for f in speaker_flags:
            icon = _SEVERITY_EMOJI.get(f.severity, "⚠️")
            lines.append(
                f"- {icon} **[{f.severity.upper()}]** `{fmt_sec(f.start)}` "
                f"— `{f.flag_type}`: {f.message}"
            )
        lines.append("")

    return "\n".join(lines)


def _summary_report(
    result: TranscriptResult,
    time_range: TimeRange,
    generated_at: datetime.datetime,
) -> str:
    """Build the summary Markdown for all speakers."""
    entries = build_diarization(result.segments)
    lines: List[str] = []

    lines.append("# Meeting Report — Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Generated | {generated_at.strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| Source | `{os.path.basename(result.source_file)}` |")
    lines.append(f"| Backend | `{result.backend}` |")
    lines.append(f"| Time range | {time_range} |")
    lines.append(f"| Duration processed | {fmt_duration(time_range.duration)} |")
    lines.append(f"| Speakers | {len(entries)} |")
    lines.append(f"| Total flags | {len(result.flags)} |")
    lines.append("")

    lines.append("## Speaker Breakdown")
    lines.append("")
    lines.append(
        "| Speaker | Speaking time | Word count | Segments | Flags |"
    )
    lines.append("|---------|---------------|------------|----------|-------|")
    for e in entries:
        segs = result.segments_for_speaker(e.speaker)
        words = sum(s.word_count for s in segs)
        n_flags = sum(1 for f in result.flags if f.speaker == e.speaker)
        lines.append(
            f"| {e.speaker} "
            f"| {fmt_duration(e.total_speaking_duration)} "
            f"| {words:,} "
            f"| {e.segment_count} "
            f"| {n_flags} |"
        )
    lines.append("")

    if result.flags:
        lines.append("## Quality Flags (all speakers)")
        lines.append("")
        for f in result.flags:
            icon = _SEVERITY_EMOJI.get(f.severity, "⚠️")
            lines.append(
                f"- {icon} **[{f.severity.upper()}]** `{fmt_sec(f.start)}` "
                f"{f.speaker}: `{f.flag_type}` — {f.message}"
            )
        lines.append("")

    return "\n".join(lines)


def write_markdown_reports(
    result: TranscriptResult,
    time_range: TimeRange,
    output_dir: str,
) -> List[str]:
    """Write per-speaker + summary Markdown reports to *output_dir*.

    Returns
    -------
    List[str]
        Paths of the files written.
    """
    ensure_dir(output_dir)
    now = datetime.datetime.now()
    written: List[str] = []

    # Per-speaker reports
    for speaker in result.speakers():
        segs = result.segments_for_speaker(speaker)
        content = _speaker_report(
            speaker=speaker,
            segments=segs,
            all_flags=result.flags,
            time_range=time_range,
            source_file=result.source_file,
            backend=result.backend,
            generated_at=now,
        )
        fname = f"{safe_filename(speaker)}.md"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(path)

    # Summary report
    summary = _summary_report(result, time_range, now)
    summary_path = os.path.join(output_dir, "_summary.md")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    written.append(summary_path)

    return written
