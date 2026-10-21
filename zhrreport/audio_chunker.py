"""WAV and audio chunking helpers.

For WAV files the standard library :mod:`wave` module is used — no
third-party packages are required.

For other audio formats (MP3, M4A, FLAC, etc.) **ffmpeg** is used if
it is available on ``PATH`` or via ``imageio-ffmpeg``.  If neither is
available and the input is not a WAV file, a :class:`RuntimeError` is
raised with a clear message explaining how to install ffmpeg.
"""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from typing import List, Optional, Tuple

from zhrreport.models import TimeRange
from zhrreport.utils import ensure_dir, find_ffmpeg


# ---------------------------------------------------------------------------
# WAV splitting via stdlib
# ---------------------------------------------------------------------------

def get_wav_duration(path: str) -> float:
    """Return the duration (seconds) of a WAV file."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def split_wav(
    input_path: str,
    output_dir: str,
    chunk_duration: float = 300.0,
    time_range: Optional[TimeRange] = None,
) -> List[Tuple[str, TimeRange]]:
    """Split a WAV file into fixed-length chunks using stdlib :mod:`wave`.

    Parameters
    ----------
    input_path:
        Path to the source WAV file.
    output_dir:
        Directory where chunk files are written.
    chunk_duration:
        Target length of each chunk in seconds.
    time_range:
        If supplied, only audio within this range is processed.

    Returns
    -------
    List[Tuple[str, TimeRange]]
        Each element is ``(chunk_file_path, time_range_of_chunk)``.
    """
    ensure_dir(output_dir)
    chunks: List[Tuple[str, TimeRange]] = []

    with wave.open(input_path, "rb") as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()
        total_duration = total_frames / framerate

        start_sec = time_range.start if time_range else 0.0
        end_sec = time_range.end if time_range else total_duration

        # Clamp to actual file duration
        start_sec = max(0.0, min(start_sec, total_duration))
        end_sec = max(start_sec, min(end_sec, total_duration))

        # Seek to start
        start_frame = int(start_sec * framerate)
        wf.setpos(start_frame)

        chunk_frames = int(chunk_duration * framerate)
        current_sec = start_sec
        chunk_idx = 0

        while current_sec < end_sec:
            chunk_end_sec = min(current_sec + chunk_duration, end_sec)
            frames_to_read = int((chunk_end_sec - current_sec) * framerate)
            frames = wf.readframes(frames_to_read)
            if not frames:
                break

            chunk_path = os.path.join(
                output_dir, f"chunk_{chunk_idx:04d}.wav"
            )
            with wave.open(chunk_path, "wb") as cw:
                cw.setnchannels(n_channels)
                cw.setsampwidth(sampwidth)
                cw.setframerate(framerate)
                cw.writeframes(frames)

            chunks.append((chunk_path, TimeRange(current_sec, chunk_end_sec)))
            current_sec = chunk_end_sec
            chunk_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Generic audio splitting via ffmpeg (optional)
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    try:
        find_ffmpeg()
        return True
    except RuntimeError:
        return False


def split_audio_ffmpeg(
    input_path: str,
    output_dir: str,
    chunk_duration: float = 300.0,
    time_range: Optional[TimeRange] = None,
) -> List[Tuple[str, TimeRange]]:
    """Split any audio format supported by ffmpeg into WAV chunks.

    Raises
    ------
    RuntimeError
        If ffmpeg is not available on ``PATH`` or via ``imageio-ffmpeg``.
    """
    try:
        ffmpeg_exe = find_ffmpeg()
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    ensure_dir(output_dir)

    # Determine total duration via ffprobe / ffmpeg
    from zhrreport.utils import probe_duration
    total_duration = probe_duration(input_path)

    start_sec = time_range.start if time_range else 0.0
    end_sec = time_range.end if time_range else total_duration

    chunks: List[Tuple[str, TimeRange]] = []
    current_sec = start_sec
    chunk_idx = 0

    while current_sec < end_sec:
        chunk_end_sec = min(current_sec + chunk_duration, end_sec)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.wav")

        cmd = [
            ffmpeg_exe, "-y",
            "-ss", str(current_sec),
            "-to", str(chunk_end_sec),
            "-i", input_path,
            "-ac", "1",        # mono
            "-ar", "16000",    # 16 kHz (good for most ASR)
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            chunk_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-400:]
            raise RuntimeError(
                f"ffmpeg failed for chunk {chunk_idx}: {err}"
            )

        chunks.append((chunk_path, TimeRange(current_sec, chunk_end_sec)))
        current_sec = chunk_end_sec
        chunk_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def split_audio(
    input_path: str,
    output_dir: str,
    chunk_duration: float = 300.0,
    time_range: Optional[TimeRange] = None,
) -> List[Tuple[str, TimeRange]]:
    """Split *input_path* into WAV chunks using the best available method.

    WAV files use the stdlib implementation.  Other formats require
    ffmpeg and raise :class:`RuntimeError` if it is not installed.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return split_wav(input_path, output_dir, chunk_duration, time_range)
    return split_audio_ffmpeg(
        input_path, output_dir, chunk_duration, time_range
    )
