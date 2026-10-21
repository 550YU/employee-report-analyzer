"""Abstract base class for pluggable transcription backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from zhrreport.models import TimeRange, TranscriptSegment


class TranscriberBase(ABC):
    """Interface every transcription backend must satisfy.

    A *backend* is responsible for transcribing a single audio chunk
    (or a time-windowed slice of an audio file) and returning a list of
    :class:`~zhrreport.models.TranscriptSegment` objects.

    Implementations should be lightweight and stateless; instantiation
    must not require network access or heavy I/O.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier, e.g. ``'fixture'``, ``'json'``, ``'whisper'``."""

    @abstractmethod
    def transcribe_chunk(
        self,
        source: str,
        time_range: Optional[TimeRange] = None,
        chunk_id: int = 0,
    ) -> List[TranscriptSegment]:
        """Transcribe *source* (file path or identifier) for *time_range*.

        Parameters
        ----------
        source:
            Path to the audio file, or an identifier meaningful to the backend
            (e.g. a JSON transcript path for the ``json`` backend).
        time_range:
            The sub-range to transcribe.  ``None`` means the whole file.
        chunk_id:
            Zero-based index of this chunk within the full recording; stored
            on each returned segment for traceability.

        Returns
        -------
        List[TranscriptSegment]
            Ordered list of transcript segments in ascending time order.
            Segments whose ``start``/``end`` fall outside *time_range* (if
            given) must be excluded by the implementation.
        """

    def describe(self) -> str:
        """One-line human description shown in ``--list-backends`` output."""
        return f"{self.backend_name} backend"
