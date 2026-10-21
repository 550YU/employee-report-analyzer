"""Core data models for zhrreport."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _fmt_sec(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_time(t: str) -> float:
    """Parse HH:MM:SS, MM:SS, or plain seconds string → float seconds."""
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str
    confidence: float = 1.0
    chunk_id: Optional[int] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    @property
    def words_per_second(self) -> float:
        d = self.duration
        return self.word_count / d if d > 0 else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # drop None chunk_id to keep JSON tidy
        if d["chunk_id"] is None:
            del d["chunk_id"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        valid = {k for k in d if k in cls.__dataclass_fields__}
        return cls(**{k: d[k] for k in valid})


# ---------------------------------------------------------------------------
# TimeRange
# ---------------------------------------------------------------------------

@dataclass
class TimeRange:
    start: float
    end: float

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def overlaps(self, seg_start: float, seg_end: float) -> bool:
        return seg_start < self.end and seg_end > self.start

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def __str__(self) -> str:
        return f"{_fmt_sec(self.start)}-{_fmt_sec(self.end)}"

    @classmethod
    def from_string(cls, s: str) -> "TimeRange":
        """Parse 'HH:MM:SS-HH:MM:SS', 'MM:SS-MM:SS', or 'float-float'."""
        s = s.strip()
        # Safely split: find the '-' that separates two time values.
        # A time value may itself contain ':'.  The separator '-' is the
        # one that is NOT inside a group of digits/colons.
        m = re.match(r"^([\d:]+(?:\.\d+)?)\s*-\s*([\d:]+(?:\.\d+)?)$", s)
        if not m:
            raise ValueError(
                f"Cannot parse time range {s!r}. "
                "Expected format: HH:MM:SS-HH:MM:SS, MM:SS-MM:SS, or seconds-seconds"
            )
        return cls(_parse_time(m.group(1)), _parse_time(m.group(2)))


# ---------------------------------------------------------------------------
# SpeakerDiarizationEntry
# ---------------------------------------------------------------------------

@dataclass
class SpeakerDiarizationEntry:
    speaker: str
    first_seen: float
    last_seen: float
    total_speaking_duration: float
    segment_count: int


# ---------------------------------------------------------------------------
# HallucinationFlag
# ---------------------------------------------------------------------------

@dataclass
class HallucinationFlag:
    segment_idx: int
    start: float
    end: float
    text: str
    speaker: str
    flag_type: str
    message: str
    severity: str  # 'low' | 'medium' | 'high'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HallucinationFlag":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ---------------------------------------------------------------------------
# TranscriptResult
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    segments: List[TranscriptSegment]
    source_file: str
    duration: float
    backend: str
    flags: List[HallucinationFlag] = field(default_factory=list)

    def speakers(self) -> List[str]:
        seen: List[str] = []
        for seg in self.segments:
            if seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen

    def segments_for_speaker(self, speaker: str) -> List[TranscriptSegment]:
        return [s for s in self.segments if s.speaker == speaker]

    def filter_by_range(self, tr: TimeRange) -> "TranscriptResult":
        segs = [s for s in self.segments if tr.overlaps(s.start, s.end)]
        flags = [f for f in self.flags if tr.overlaps(f.start, f.end)]
        return TranscriptResult(
            segments=segs,
            source_file=self.source_file,
            duration=tr.duration,
            backend=self.backend,
            flags=flags,
        )

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "duration": self.duration,
            "backend": self.backend,
            "segments": [s.to_dict() for s in self.segments],
            "flags": [f.to_dict() for f in self.flags],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptResult":
        segments = [TranscriptSegment.from_dict(s) for s in d.get("segments", [])]
        flags = [HallucinationFlag.from_dict(f) for f in d.get("flags", [])]
        return cls(
            segments=segments,
            source_file=d.get("source_file", ""),
            duration=float(d.get("duration", 0.0)),
            backend=d.get("backend", "unknown"),
            flags=flags,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "TranscriptResult":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)
