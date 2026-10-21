"""JSON-file transcript backend.

Reads a pre-existing JSON transcript file that follows the
:class:`~zhrreport.models.TranscriptResult` schema.  This allows
transcripts produced by external tools (e.g. Whisper, AssemblyAI, Rev)
to be fed directly into the reporting pipeline without re-transcribing.

Expected JSON layout::

    {
      "source_file": "meeting.wav",
      "duration": 3600.0,
      "backend": "whisper-large",
      "segments": [
        {
          "start": 0.0,
          "end": 5.2,
          "speaker": "SPEAKER_00",
          "text": "Hello and welcome.",
          "confidence": 0.95
        },
        ...
      ]
    }

Only ``segments`` is strictly required; all other top-level keys are
optional and default to sensible values.
"""
from __future__ import annotations

import json
from typing import List, Optional

from zhrreport.models import TimeRange, TranscriptSegment, TranscriptResult
from zhrreport.transcribers.base import TranscriberBase


class JsonTranscriber(TranscriberBase):
    """Backend that reads a JSON transcript file.

    Parameters
    ----------
    json_path:
        Absolute or relative path to the JSON transcript file.  Can also
        be passed as *source* to :meth:`transcribe_chunk`; in that case
        the *source* argument takes precedence over the constructor value.
    """

    def __init__(self, json_path: Optional[str] = None) -> None:
        self._default_path = json_path

    @property
    def backend_name(self) -> str:
        return "json"

    def describe(self) -> str:
        return "json backend – reads a pre-existing JSON transcript file"

    def transcribe_chunk(
        self,
        source: str,
        time_range: Optional[TimeRange] = None,
        chunk_id: int = 0,
    ) -> List[TranscriptSegment]:
        path = source or self._default_path
        if not path:
            raise ValueError("JsonTranscriber: no source path provided")

        try:
            result = TranscriptResult.from_json_file(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"JsonTranscriber: transcript file not found: {path!r}"
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(
                f"JsonTranscriber: invalid JSON transcript at {path!r}: {exc}"
            ) from exc

        segments = result.segments
        if time_range is not None:
            segments = [
                s for s in segments if time_range.overlaps(s.start, s.end)
            ]

        # Tag each segment with the chunk_id
        for seg in segments:
            seg.chunk_id = chunk_id

        return segments
