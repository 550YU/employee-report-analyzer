"""Transcript hallucination detection.

Detects common failure modes in ASR transcripts (especially Whisper):

``repeated_segment``
    The same text appears consecutively two or more times.

``known_phrase``
    A well-known hallucination phrase ("Thank you for watching", etc.).

``high_word_rate``
    Segment claims more than *MAX_WORDS_PER_SECOND* words per second —
    physically impossible for normal speech.

``short_duration``
    Segment audio duration is below *MIN_SEGMENT_DURATION* — too short
    for reliable transcription.

``low_confidence``
    Segment confidence below *LOW_CONFIDENCE_THRESHOLD*.

``uniform_confidence``
    Every confidence value is exactly 1.0 — a sign the backend did not
    actually score confidence (treat all as suspect).

Each flag has a ``severity`` field (``'low'``, ``'medium'``, ``'high'``)
to help callers decide whether to suppress the segment or just warn.
"""
from __future__ import annotations

from typing import List

from zhrreport.models import HallucinationFlag, TranscriptSegment

# ---------------------------------------------------------------------------
# Tunable thresholds (override as needed)
# ---------------------------------------------------------------------------

MAX_WORDS_PER_SECOND: float = 5.0   # > this → physically suspect
MIN_SEGMENT_DURATION: float = 0.3   # < this → too short to trust
LOW_CONFIDENCE_THRESHOLD: float = 0.70
REPEAT_WINDOW: int = 3              # consecutive segments to compare

# Known Whisper / generic ASR hallucination phrases (lower-cased)
_KNOWN_BAD_PHRASES: List[str] = [
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "subscribe to my channel",
    "music plays",
    "inaudible",
    "blank audio",
    "silence",
    "[silence]",
    "[blank_audio]",
    "[music]",
    "(music)",
    "(applause)",
    "[ silence ]",
    "[ music ]",
    "subtitles by",
    "transcribed by",
    "www.",
    "http",
]


def detect_hallucinations(
    segments: List[TranscriptSegment],
) -> List[HallucinationFlag]:
    """Scan *segments* and return a list of :class:`HallucinationFlag`.

    Parameters
    ----------
    segments:
        All transcript segments (already filtered to the desired range).

    Returns
    -------
    List[HallucinationFlag]
        Zero or more flags, in segment order.
    """
    flags: List[HallucinationFlag] = []

    # --- Pre-compute: are all confidences exactly 1.0? ---
    confidences = [s.confidence for s in segments if s.confidence is not None]
    all_uniform = len(confidences) > 1 and all(c == 1.0 for c in confidences)

    if all_uniform:
        # One global flag on segment 0
        flags.append(HallucinationFlag(
            segment_idx=0,
            start=segments[0].start if segments else 0.0,
            end=segments[-1].end if segments else 0.0,
            text="(all segments)",
            speaker="*",
            flag_type="uniform_confidence",
            message=(
                "All confidence scores are exactly 1.0 — the backend likely "
                "did not score confidence.  Treat transcript with caution."
            ),
            severity="low",
        ))

    for idx, seg in enumerate(segments):
        text_lower = seg.text.lower().strip()

        # --- Known hallucination phrases ---
        for phrase in _KNOWN_BAD_PHRASES:
            if phrase in text_lower:
                flags.append(HallucinationFlag(
                    segment_idx=idx,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=seg.speaker,
                    flag_type="known_phrase",
                    message=(
                        f"Segment contains known hallucination phrase: "
                        f"{phrase!r}"
                    ),
                    severity="high",
                ))
                break  # one flag per segment for this type

        # --- High word rate ---
        if seg.duration > 0 and seg.words_per_second > MAX_WORDS_PER_SECOND:
            flags.append(HallucinationFlag(
                segment_idx=idx,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=seg.speaker,
                flag_type="high_word_rate",
                message=(
                    f"Word rate {seg.words_per_second:.1f} wps exceeds "
                    f"threshold {MAX_WORDS_PER_SECOND} wps — "
                    "segment may be hallucinated or mis-timed."
                ),
                severity="medium",
            ))

        # --- Very short duration ---
        if seg.duration < MIN_SEGMENT_DURATION and seg.word_count > 2:
            flags.append(HallucinationFlag(
                segment_idx=idx,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=seg.speaker,
                flag_type="short_duration",
                message=(
                    f"Segment duration {seg.duration:.3f}s is below "
                    f"minimum {MIN_SEGMENT_DURATION}s for {seg.word_count} words."
                ),
                severity="medium",
            ))

        # --- Low confidence ---
        if seg.confidence is not None and seg.confidence < LOW_CONFIDENCE_THRESHOLD:
            flags.append(HallucinationFlag(
                segment_idx=idx,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=seg.speaker,
                flag_type="low_confidence",
                message=(
                    f"Confidence {seg.confidence:.2f} is below "
                    f"threshold {LOW_CONFIDENCE_THRESHOLD:.2f}."
                ),
                severity="low",
            ))

    # --- Repeated consecutive segments ---
    for idx in range(len(segments) - 1):
        seg = segments[idx]
        nxt = segments[idx + 1]
        if (
            seg.speaker == nxt.speaker
            and seg.text.strip().lower() == nxt.text.strip().lower()
            and seg.text.strip()
        ):
            flags.append(HallucinationFlag(
                segment_idx=idx,
                start=seg.start,
                end=nxt.end,
                text=seg.text,
                speaker=seg.speaker,
                flag_type="repeated_segment",
                message=(
                    f"Identical text repeated consecutively at "
                    f"{seg.start:.1f}s and {nxt.start:.1f}s: "
                    f"{seg.text[:60]!r}"
                ),
                severity="high",
            ))

    flags.sort(key=lambda f: f.start)
    return flags


def summarise_flags(flags: List[HallucinationFlag]) -> str:
    """Return a short human-readable summary of hallucination flags."""
    if not flags:
        return "No hallucination flags detected."

    by_severity = {"high": 0, "medium": 0, "low": 0}
    for f in flags:
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

    parts = []
    if by_severity["high"]:
        parts.append(f"{by_severity['high']} HIGH")
    if by_severity["medium"]:
        parts.append(f"{by_severity['medium']} MEDIUM")
    if by_severity["low"]:
        parts.append(f"{by_severity['low']} LOW")

    return f"{len(flags)} flag(s): " + ", ".join(parts)
