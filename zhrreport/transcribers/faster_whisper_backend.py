"""faster-whisper ASR backend.

Uses the CTranslate2-optimised Whisper implementation for low-overhead
CPU inference (int8 quantisation by default).

Availability
------------
``faster-whisper`` is an optional dependency.  Importing this module
succeeds even when the package is absent; instantiating
:class:`FasterWhisperTranscriber` raises :class:`ImportError` with a
friendly install message.

Install::

    pip install faster-whisper

Recommended cached models (no network required after first download)::

    Systran/faster-whisper-base    (default — fast, ~75 MB)
    Systran/faster-whisper-small   (better quality, ~150 MB)
    Systran/faster-whisper-medium  (high quality, ~450 MB)
    mobiuslabsgmbh/faster-whisper-large-v3-turbo  (best quality, ~800 MB)

Chunk workflow
--------------
The :class:`~zhrreport.chunker.ChunkedTranscriber` calls this backend
through the :mod:`zhrreport._worker` subprocess for each 30–60 s chunk.
For every chunk the transcriber:

1. Extracts the requested time-range slice as a 16 kHz mono WAV using
   ffmpeg (discovered via PATH or ``imageio-ffmpeg``).
2. Runs faster-whisper with ``beam_size=1``, ``vad_filter=True``,
   ``word_timestamps=False`` (fast CPU-friendly defaults).
3. Deduplicates consecutive identical segments and strips known
   hallucination phrases.
4. If the fraction of high-confidence hallucination segments exceeds
   a threshold **and** a ``fallback_model`` is configured, re-runs
   the chunk with the larger model (fail-closed escalation).
5. Cleans up the temporary WAV file.

Speaker labels
--------------
faster-whisper does not perform speaker diarization.  All segments are
assigned the anonymous label ``SPEAKER`` unless ``speaker_name`` is
supplied (used when transcribing a known-speaker range).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

from zhrreport.models import TimeRange, TranscriptSegment
from zhrreport.transcribers.base import TranscriberBase


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_availability() -> tuple[bool, str]:
    """Return ``(available, install_hint)``."""
    try:
        import faster_whisper  # noqa: F401
        return True, ""
    except ImportError:
        return False, (
            "faster-whisper is not installed.\n"
            "  Install: pip install faster-whisper\n"
            "  Requires: pip install ctranslate2 (usually bundled)\n"
            "  Docs: https://github.com/SYSTRAN/faster-whisper"
        )


# ---------------------------------------------------------------------------
# Hallucination helpers
# ---------------------------------------------------------------------------

_HALLUCINATION_PHRASES = frozenset([
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "subscribe to my channel",
    "music plays",
    "[music]",
    "(music)",
    "[applause]",
    "(applause)",
    "[inaudible]",
    "[blank_audio]",
    "[silence]",
    "subtitles by",
    "transcribed by",
    "amara.org",
    "www.",
])

_MAX_WORDS_PER_SEC = 6.5  # physically impossible rate → hallucination
_MIN_CONFIDENCE = 0.0     # no_speech_prob > 0.8 → skip
_NO_SPEECH_THRESHOLD = 0.85


def _segment_is_hallucination(text: str, duration: float, no_speech_prob: float) -> bool:
    """Return True if the segment looks like a Whisper hallucination."""
    if no_speech_prob > _NO_SPEECH_THRESHOLD:
        return True
    t = text.strip().lower()
    if not t:
        return True
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in t:
            return True
    words = len(text.split())
    if duration > 0 and words / duration > _MAX_WORDS_PER_SEC:
        return True
    # Suspiciously short with many words
    if duration < 0.3 and words > 3:
        return True
    return False


def _dedup_segments(segs: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Remove consecutive identical text and hallucination segments."""
    result: List[TranscriptSegment] = []
    prev_lower: str = ""
    for seg in segs:
        curr = seg.text.strip().lower()
        if not curr:
            continue
        if curr == prev_lower:
            continue  # deduplicate consecutive identical
        result.append(seg)
        prev_lower = curr
    return result


# ---------------------------------------------------------------------------
# ffmpeg helper (uses utils discovery)
# ---------------------------------------------------------------------------

def _extract_wav_chunk(
    source: str,
    start_sec: float,
    end_sec: Optional[float],
    out_path: str,
) -> None:
    """Extract [start_sec, end_sec] from *source* as 16 kHz mono WAV."""
    from zhrreport.utils import find_ffmpeg
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-ss", str(start_sec)]
    if end_sec is not None:
        cmd += ["-to", str(end_sec)]
    cmd += [
        "-i", source,
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg chunk extraction failed: {err}")


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------

class FasterWhisperTranscriber(TranscriberBase):
    """Production ASR backend using faster-whisper (CTranslate2).

    Parameters
    ----------
    model:
        Model identifier — anything accepted by faster-whisper, e.g.
        ``"base"``, ``"small"``, ``"Systran/faster-whisper-medium"``.
        Defaults to ``"base"`` (fast, good for Windows CPU).
    compute_type:
        CTranslate2 quantisation type.  ``"int8"`` is the Windows CPU
        default; use ``"float16"`` for GPU.
    device:
        ``"cpu"`` or ``"cuda"``.
    beam_size:
        Beam search width.  1 = greedy (fastest on CPU).
    vad_filter:
        Use Silero VAD to skip silent regions (reduces hallucinations).
    language:
        ISO 639-1 code, e.g. ``"zh"``, ``"en"``.  ``None`` = auto-detect.
    speaker_name:
        Assign this label to all returned segments (for known-speaker
        ranges).  Defaults to ``"SPEAKER"``.
    fallback_model:
        If set and the primary model is low-confidence, too sparse, or has
        all emitted segments rejected, re-transcribe with this model.
    """

    def __init__(
        self,
        model: str = "base",
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 1,
        vad_filter: bool = True,
        language: Optional[str] = None,
        speaker_name: str = "SPEAKER",
        fallback_model: Optional[str] = None,
        temp_run_id: str = "standalone",
        **_extra,
    ) -> None:
        avail, msg = check_availability()
        if not avail:
            raise ImportError(
                f"FasterWhisperTranscriber requires faster-whisper.\n{msg}"
            )
        self._model_name = model
        self._compute_type = compute_type
        self._device = device
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._language = language
        self._speaker_name = speaker_name
        self._fallback_model = fallback_model
        self._temp_run_id = temp_run_id
        # Models are loaded lazily (expensive import / init)
        self._model = None
        self._fallback = None

    @property
    def backend_name(self) -> str:
        return "faster-whisper"

    def describe(self) -> str:
        avail, msg = check_availability()
        if not avail:
            return f"faster-whisper NOT available — {msg}"
        try:
            import faster_whisper
            ver = getattr(faster_whisper, "__version__", "?")
        except Exception:
            ver = "?"
        fb = f", fallback={self._fallback_model!r}" if self._fallback_model else ""
        return (
            f"faster-whisper {ver}  model={self._model_name!r}  "
            f"device={self._device}  compute_type={self._compute_type}  "
            f"beam_size={self._beam_size}  vad={self._vad_filter}{fb}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_chunk(
        self,
        source: str,
        time_range: Optional[TimeRange] = None,
        chunk_id: int = 0,
    ) -> List[TranscriptSegment]:
        """Transcribe *source* within *time_range*.

        1. Extracts the time-range slice as a temp WAV using ffmpeg.
        2. Runs faster-whisper with VAD and dedup.
        3. Escalates to fallback model if too many hallucinations.
        4. Cleans up temp WAV.
        """
        start_sec = time_range.start if time_range else 0.0
        end_sec = time_range.end if time_range else None

        # Write temp WAV to working directory (not /tmp)
        tmp_dir = os.path.join(os.getcwd(), ".zhr_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        chunk_wav = os.path.join(
            tmp_dir,
            f"run_{self._temp_run_id}_chunk_{chunk_id}_{int(start_sec)}_{ts}.wav",
        )

        try:
            _extract_wav_chunk(source, start_sec, end_sec, chunk_wav)

            # Primary model
            if self._model is None:
                self._model = self._load_model(self._model_name)
            segs, rejected = self._run_whisper(
                self._model, chunk_wav, start_sec, chunk_id
            )

            # Fallback escalation
            if self._fallback_model and self._needs_escalation(segs, rejected):
                print(
                    f"[faster_whisper] chunk {chunk_id}: "
                    f"escalating to {self._fallback_model!r}",
                    file=sys.stderr,
                )
                if self._fallback is None:
                    self._fallback = self._load_model(self._fallback_model)
                segs, _ = self._run_whisper(
                    self._fallback, chunk_wav, start_sec, chunk_id
                )

            return segs

        finally:
            _cleanup_wav(chunk_wav)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, name: str):
        from faster_whisper import WhisperModel
        print(
            f"[faster_whisper] loading {name!r} on {self._device}/{self._compute_type} …",
            file=sys.stderr,
        )
        return WhisperModel(name, device=self._device, compute_type=self._compute_type)

    def _run_whisper(
        self,
        model,
        wav_path: str,
        start_offset: float,
        chunk_id: int,
    ) -> tuple[List[TranscriptSegment], int]:
        """Run transcription; return deduped, offset-adjusted segments."""
        segments_gen, _info = model.transcribe(
            wav_path,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            word_timestamps=False,
            language=self._language,
        )

        raw: List[TranscriptSegment] = []
        rejected = 0
        for seg in segments_gen:
            no_speech = getattr(seg, "no_speech_prob", 0.0)
            duration = max(0.001, seg.end - seg.start)
            if _segment_is_hallucination(seg.text, duration, no_speech):
                rejected += 1
                continue
            # Convert avg_logprob (log probability ≤ 0) to a [0,1] confidence.
            # exp(0) = 1.0 (perfect), exp(-0.5) ≈ 0.61, exp(-1.0) ≈ 0.37.
            # Do NOT factor in no_speech_prob here; hallucination filter already
            # applied the no_speech threshold above.
            import math as _math
            alp = max(-4.0, min(0.0, getattr(seg, "avg_logprob", -0.3)))
            confidence = round(_math.exp(alp), 3)

            raw.append(TranscriptSegment(
                start=start_offset + seg.start,
                end=start_offset + seg.end,
                speaker=self._speaker_name,
                text=seg.text.strip(),
                confidence=round(confidence, 3),
                chunk_id=chunk_id,
            ))

        return _dedup_segments(raw), rejected

    def _needs_escalation(
        self, segs: List[TranscriptSegment], rejected_count: int = 0
    ) -> bool:
        """Return whether a non-silent primary result needs fallback ASR."""
        if not segs:
            return rejected_count > 0
        # If we get very few short segments from a 30–60s chunk, escalate
        total_words = sum(s.word_count for s in segs)
        if total_words < 5:
            return True
        # Check average confidence
        avg_conf = sum(s.confidence for s in segs) / len(segs)
        return avg_conf < 0.4


def _cleanup_wav(path: str) -> None:
    """Silently remove a temp file."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
