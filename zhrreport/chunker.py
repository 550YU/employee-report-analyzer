"""Chunked transcription with hard per-chunk timeout.

Architecture
------------
Each audio chunk is processed by launching a **subprocess** that runs
:mod:`zhrreport._worker`.  The parent process waits for the worker and
calls :func:`~zhrreport.utils.kill_process_tree` if the chunk exceeds
``timeout`` seconds.  This guarantees the child process tree is killed
even on Windows (using ``taskkill /F /T /PID``).

For the ``fixture`` and ``json`` backends the worker finishes in
milliseconds; the timeout mechanism is the same regardless of backend so
production ASR backends that spawn their own child processes are handled
correctly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import glob
import uuid
from typing import Callable, List, Optional

from zhrreport.models import TimeRange, TranscriptResult, TranscriptSegment
from zhrreport.utils import ensure_dir, kill_process_tree

# Default chunk length (seconds).  Change via --chunk-duration on CLI.
DEFAULT_CHUNK_DURATION: float = 300.0  # 5 min

# Default per-chunk worker timeout (seconds).
DEFAULT_TIMEOUT: float = 120.0


class ChunkTranscriptionError(RuntimeError):
    """Raised when any chunk fails, preventing partial reports from looking complete."""


class ChunkedTranscriber:
    """Run a transcription backend over *time_range* in fixed-size chunks.

    Parameters
    ----------
    backend_name:
        The backend identifier understood by :mod:`zhrreport.transcribers`.
    backend_kwargs:
        Extra keyword arguments forwarded to the backend constructor.
    chunk_duration:
        Length (seconds) of each chunk.
    timeout:
        Hard per-chunk timeout (seconds).  The worker subprocess is
        forcibly killed if it does not finish within this time.
    progress_cb:
        Optional callback ``(chunk_idx, total_chunks, elapsed)`` called
        after each completed chunk.
    """

    def __init__(
        self,
        backend_name: str,
        backend_kwargs: Optional[dict] = None,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
        timeout: float = DEFAULT_TIMEOUT,
        progress_cb: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        self.backend_name = backend_name
        self.backend_kwargs: dict = backend_kwargs or {}
        self.chunk_duration = chunk_duration
        self.timeout = timeout
        self.progress_cb = progress_cb
        self._temp_run_id = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source: str,
        full_duration: float,
        time_range: Optional[TimeRange] = None,
    ) -> TranscriptResult:
        """Transcribe *source* within *time_range* using chunked workers.

        Parameters
        ----------
        source:
            Path to the audio/transcript file passed to the backend.
        full_duration:
            Total recording duration in seconds; used when *time_range*
            is ``None`` to span the whole recording.
        time_range:
            The sub-range to process.  Defaults to the full recording.

        Returns
        -------
        TranscriptResult
            Merged result from all chunks, in time order.
        """
        if time_range is None:
            time_range = TimeRange(0.0, full_duration)

        chunks = self._make_chunks(time_range)
        all_segments: List[TranscriptSegment] = []
        errors: List[str] = []

        import time as _time

        for idx, chunk_range in enumerate(chunks):
            t0 = _time.monotonic()
            try:
                segs = self._run_chunk_worker(source, chunk_range, idx)
                all_segments.extend(segs)
            except TimeoutError as exc:
                errors.append(f"Chunk {idx} timed out: {exc}")
                print(
                    f"  [WARNING] Chunk {idx} ({chunk_range}) timed out after "
                    f"{self.timeout}s — skipping.",
                    file=sys.stderr,
                )
            except Exception as exc:
                errors.append(f"Chunk {idx} failed: {exc}")
                print(
                    f"  [ERROR] Chunk {idx} ({chunk_range}) failed: {exc}",
                    file=sys.stderr,
                )

            elapsed = _time.monotonic() - t0
            if self.progress_cb:
                self.progress_cb(idx, len(chunks), elapsed)

        if errors:
            details = "; ".join(errors)
            raise ChunkTranscriptionError(
                f"{len(errors)} chunk(s) failed; no reports were generated from "
                f"an incomplete transcript. {details}"
            )

        all_segments.sort(key=lambda s: s.start)
        return TranscriptResult(
            segments=all_segments,
            source_file=source,
            duration=time_range.duration,
            backend=self.backend_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_chunks(self, time_range: TimeRange) -> List[TimeRange]:
        """Divide *time_range* into sub-ranges of at most *chunk_duration*."""
        chunks: List[TimeRange] = []
        t = time_range.start
        while t < time_range.end:
            chunk_end = min(t + self.chunk_duration, time_range.end)
            chunks.append(TimeRange(t, chunk_end))
            t = chunk_end
        return chunks or [time_range]

    def _run_chunk_worker(
        self,
        source: str,
        chunk_range: TimeRange,
        chunk_id: int,
    ) -> List[TranscriptSegment]:
        """Spawn a worker subprocess for one chunk; kill on timeout."""
        cmd = [
            sys.executable,
            "-m", "zhrreport._worker",
            "--backend", self.backend_name,
            "--source", source,
            "--start", str(chunk_range.start),
            "--end", str(chunk_range.end),
            "--chunk-id", str(chunk_id),
        ]
        for k, v in self.backend_kwargs.items():
            cmd += [f"--backend-kwarg", f"{k}={v}"]
        if self.backend_name == "faster-whisper":
            cmd += ["--backend-kwarg", f"temp_run_id={self._temp_run_id}"]

        # On Windows: CREATE_NEW_PROCESS_GROUP lets us kill the whole tree.
        # On POSIX: start_new_session=True creates a new process group.
        popen_kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            try:
                proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                for pipe in (proc.stdout, proc.stderr):
                    if pipe is not None:
                        pipe.close()
            self._cleanup_killed_chunk_temp_files(self._temp_run_id, chunk_id)
            raise TimeoutError(
                f"chunk worker PID {proc.pid} exceeded {self.timeout}s"
            )

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Worker exited {proc.returncode}: {err_msg[:400]}"
            )

        try:
            raw = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Worker output is not valid JSON: {exc}"
            ) from exc

        return [TranscriptSegment.from_dict(d) for d in raw]

    @staticmethod
    def _cleanup_killed_chunk_temp_files(temp_run_id: str, chunk_id: int) -> None:
        """Remove only temporary WAVs owned by a force-killed chunk worker."""
        temp_dir = os.path.join(os.getcwd(), ".zhr_tmp")
        pattern = os.path.join(
            temp_dir, f"run_{temp_run_id}_chunk_{chunk_id}_*.wav"
        )
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
