"""Local (no-token) speaker diarization.

Three tiers of diarization backend, from lightest to heaviest:

NumpyAcousticDiarizer  (always available — stdlib + numpy)
    Uses frame-level energy / zero-crossing / spectral features and a
    pure-numpy k-means to cluster speakers.  FAST, no downloads.
    ⚠ LOW QUALITY — spectral features alone are NOT high-quality
    speaker embeddings.  Suitable only as a layout guide.

SpeechBrainDiarizer  (optional — requires speechbrain + torch + scikit-learn)
    Uses SpeechBrain's ECAPA-TDNN model for speaker embeddings and
    AgglomerativeClustering from scikit-learn.  HIGH QUALITY.
    Install:  pip install speechbrain torch scikit-learn

PyAnnoteDiarizer  (optional — requires pyannote.audio + HuggingFace token)
    Uses pyannote.audio pipeline.  HIGHEST QUALITY, needs token.
    Install:  pip install pyannote.audio

Usage
-----
>>> from zhrreport.local_diarizer import get_diarizer, AudioDiarizationEntry
>>> d = get_diarizer("numpy_acoustic")
>>> entries = d.diarize("meeting.mp3", num_speakers=2)
>>> for e in entries:
...     print(e.speaker, e.start, e.end)
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AudioDiarizationEntry:
    """Single speaker turn from audio diarization."""
    speaker: str      # anonymous e.g. SPEAKER_00
    start: float      # seconds
    end: float        # seconds

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ---------------------------------------------------------------------------
# Backend availability helpers
# ---------------------------------------------------------------------------

def _check_numpy() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
        return True, ""
    except ImportError:
        return False, "pip install numpy"


def _check_speechbrain() -> tuple[bool, str]:
    missing = []
    for pkg, inst in [("speechbrain", "speechbrain"), ("torch", "torch"),
                       ("sklearn", "scikit-learn")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(inst)
    if missing:
        return False, f"pip install {' '.join(missing)}"
    return True, ""


def _check_pyannote() -> tuple[bool, str]:
    try:
        import pyannote.audio  # noqa: F401
        return True, ""
    except ImportError:
        return False, "pip install pyannote.audio"


def _find_ffmpeg() -> str:
    """Delegate to utils discovery."""
    from zhrreport.utils import find_ffmpeg
    return find_ffmpeg()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DiarizeBase:
    """Interface for all local diarization backends."""

    def diarize(
        self,
        source: str,
        num_speakers: int = 2,
        time_range: Optional[tuple[float, float]] = None,
    ) -> List[AudioDiarizationEntry]:
        """Return anonymous speaker segments for *source*.

        Parameters
        ----------
        source:
            Path to audio file (MP3/WAV/…).
        num_speakers:
            Expected number of speakers.
        time_range:
            ``(start_sec, end_sec)`` sub-range to analyse.  ``None`` →
            process the whole file.

        Returns
        -------
        List[AudioDiarizationEntry]
            Sorted by start time.
        """
        raise NotImplementedError

    def describe(self) -> str:
        return "diarizer base"


# ---------------------------------------------------------------------------
# Audio extraction helper (shared)
# ---------------------------------------------------------------------------

def _extract_wav(
    source: str,
    out_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    sample_rate: int = 16000,
) -> None:
    """Extract *source* (optionally sub-range) to 16 kHz mono WAV."""
    ffmpeg = _find_ffmpeg()
    cmd = [ffmpeg, "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += [
        "-i", source,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg extraction failed: {err}")


def _tmp_wav(prefix: str = "diar") -> str:
    """Return path for a temp WAV in .zhr_tmp (never /tmp)."""
    import time
    tmp_dir = os.path.join(os.getcwd(), ".zhr_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"{prefix}_{int(time.time()*1000)}.wav")


# ---------------------------------------------------------------------------
# Tier 1: NumpyAcousticDiarizer  (stdlib + numpy, always-available fallback)
# ---------------------------------------------------------------------------

class NumpyAcousticDiarizer(DiarizeBase):
    """Low-quality acoustic clustering diarizer.

    ⚠ LIMITATION: Uses short-time energy + zero-crossing rate +
    1-D spectral centroid as speaker features.  These features
    distinguish *loudness/pitch profiles* but are NOT speaker embeddings.
    Results are a rough layout guide only.  Expect poor performance
    when speakers have similar vocal characteristics.

    Advantages:
    - Zero downloads, no ML models
    - Runs on any machine with numpy
    - Deterministic given same input
    """

    FRAME_MS = 500    # 500 ms analysis frame
    HOP_MS = 250      # 250 ms hop (50% overlap)
    SMOOTH_K = 5      # median smoothing kernel

    def describe(self) -> str:
        return (
            "NumpyAcousticDiarizer [LOW QUALITY FALLBACK] "
            "— energy/ZCR/spectral features + k-means, no speaker embeddings"
        )

    def diarize(
        self,
        source: str,
        num_speakers: int = 2,
        time_range: Optional[tuple[float, float]] = None,
    ) -> List[AudioDiarizationEntry]:
        avail, msg = _check_numpy()
        if not avail:
            raise ImportError(
                f"NumpyAcousticDiarizer requires numpy.  Install: {msg}"
            )
        import numpy as np

        start = time_range[0] if time_range else None
        end = time_range[1] if time_range else None

        wav_path = _tmp_wav("numpy_diar")
        try:
            _extract_wav(source, wav_path, start, end, sample_rate=16000)
            return self._process(wav_path, num_speakers, start or 0.0, np)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

    def _process(self, wav_path: str, n_speakers: int, time_offset: float, np) -> List[AudioDiarizationEntry]:
        import wave as _wave

        with _wave.open(wav_path, "rb") as wf:
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert to float32 [-1, 1]
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        frame_samples = int(self.FRAME_MS / 1000 * framerate)
        hop_samples = int(self.HOP_MS / 1000 * framerate)
        n_frames_total = max(1, (len(samples) - frame_samples) // hop_samples + 1)

        features = []
        frame_times = []

        for i in range(n_frames_total):
            s = i * hop_samples
            e = s + frame_samples
            frame = samples[s:e]
            if len(frame) < frame_samples // 2:
                break

            # Pad if needed
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))

            # Feature 1: RMS energy
            energy = float(np.sqrt(np.mean(frame ** 2)) + 1e-9)

            # Feature 2: Zero-crossing rate
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) / 2.0))

            # Feature 3: Spectral centroid (dominant frequency band)
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            freqs = np.fft.rfftfreq(len(frame), d=1.0 / framerate)
            spec_sum = spectrum.sum() + 1e-9
            centroid = float(np.dot(freqs, spectrum) / spec_sum)

            # Feature 4: Spectral rolloff (85 % energy frequency)
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * spec_sum)
            rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

            features.append([energy, zcr, centroid, rolloff])
            frame_times.append(s / framerate)

        if not features:
            return []

        X = np.array(features, dtype=np.float64)

        # Normalize features
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        X_norm = (X - mean) / std

        # Simple energy-based voice activity detection — skip near-silent frames
        energy_col = X[:, 0]
        energy_thresh = np.percentile(energy_col, 15)
        voiced = energy_col > energy_thresh

        # k-means clustering (pure numpy)
        labels = self._numpy_kmeans(X_norm, n_speakers, voiced)

        # Median smooth
        labels = self._median_smooth(labels, self.SMOOTH_K)

        # Convert frame labels → speaker turns
        return self._labels_to_entries(labels, frame_times, time_offset, n_speakers)

    @staticmethod
    def _numpy_kmeans(X: "np.ndarray", k: int, mask: "np.ndarray", max_iter: int = 50) -> "np.ndarray":
        """Simple k-means on rows of X where mask is True."""
        import numpy as np
        n = len(X)
        labels = np.zeros(n, dtype=int)

        # Initialise centroids from voiced frames with max spread
        voiced_idxs = np.where(mask)[0]
        if len(voiced_idxs) < k:
            voiced_idxs = np.arange(n)

        # k-means++ style init
        rng = np.random.default_rng(42)
        centroids = [X[voiced_idxs[rng.integers(len(voiced_idxs))]].copy()]
        for _ in range(k - 1):
            dists = np.array([
                min(np.linalg.norm(X[i] - c) for c in centroids)
                for i in voiced_idxs
            ])
            probs = dists ** 2
            probs /= probs.sum() + 1e-9
            idx = voiced_idxs[rng.choice(len(voiced_idxs), p=probs)]
            centroids.append(X[idx].copy())

        centroids = np.stack(centroids)

        for _ in range(max_iter):
            # Assign each frame to nearest centroid
            dists = np.array([
                np.linalg.norm(X - c, axis=1) for c in centroids
            ])  # (k, n)
            new_labels = np.argmin(dists, axis=0)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update centroids (only from voiced frames)
            for j in range(k):
                members = np.where((labels == j) & mask)[0]
                if len(members):
                    centroids[j] = X[members].mean(axis=0)

        return labels

    @staticmethod
    def _median_smooth(labels: "np.ndarray", k: int) -> "np.ndarray":
        """Apply median filter to label sequence."""
        import numpy as np
        n = len(labels)
        result = labels.copy()
        half = k // 2
        for i in range(half, n - half):
            window = labels[i - half: i + half + 1]
            counts = np.bincount(window, minlength=labels.max() + 1)
            result[i] = int(np.argmax(counts))
        return result

    @staticmethod
    def _labels_to_entries(
        labels: "np.ndarray",
        times: List[float],
        offset: float,
        n_speakers: int,
    ) -> List[AudioDiarizationEntry]:
        """Convert frame-level labels to speaker-turn entries."""
        if not len(labels):
            return []
        entries = []
        cur_label = labels[0]
        cur_start = times[0]
        hop = times[1] - times[0] if len(times) > 1 else 0.25

        for i in range(1, len(labels)):
            if labels[i] != cur_label:
                if times[i] - cur_start >= 0.5:  # skip micro turns
                    entries.append(AudioDiarizationEntry(
                        speaker=f"SPEAKER_{cur_label:02d}",
                        start=offset + cur_start,
                        end=offset + times[i],
                    ))
                cur_label = labels[i]
                cur_start = times[i]

        # Last segment
        last_end = times[-1] + hop
        if last_end - cur_start >= 0.5:
            entries.append(AudioDiarizationEntry(
                speaker=f"SPEAKER_{cur_label:02d}",
                start=offset + cur_start,
                end=offset + last_end,
            ))

        return entries


# ---------------------------------------------------------------------------
# Tier 2: SpeechBrainDiarizer  (optional, high quality)
# ---------------------------------------------------------------------------

class SpeechBrainDiarizer(DiarizeBase):
    """High-quality diarizer using SpeechBrain ECAPA-TDNN embeddings.

    Requires:  pip install speechbrain torch scikit-learn
    """

    def describe(self) -> str:
        return "SpeechBrainDiarizer — ECAPA-TDNN embeddings + agglomerative clustering (HIGH QUALITY)"

    def diarize(
        self,
        source: str,
        num_speakers: int = 2,
        time_range: Optional[tuple[float, float]] = None,
    ) -> List[AudioDiarizationEntry]:
        avail, msg = _check_speechbrain()
        if not avail:
            raise ImportError(
                f"SpeechBrainDiarizer requires: {msg}\n"
                f"Install: {msg}"
            )

        import numpy as np
        import torch
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.pretrained import EncoderClassifier

        start = time_range[0] if time_range else None
        end = time_range[1] if time_range else None

        wav_path = _tmp_wav("sb_diar")
        try:
            _extract_wav(source, wav_path, start, end, sample_rate=16000)
            return self._process(
                wav_path, num_speakers, start or 0.0, np, torch,
                AgglomerativeClustering, EncoderClassifier,
            )
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

    def _process(self, wav_path, n_speakers, offset, np, torch, Cluster, Encoder):
        classifier = Encoder.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.getcwd(), ".zhr_models", "ecapa"),
        )

        SEGMENT_LEN = 1.5   # seconds per embedding window
        HOP = 0.75

        with __import__("wave").open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            samples = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768.0

        seg_samples = int(SEGMENT_LEN * sr)
        hop_samples = int(HOP * sr)
        embeddings, times = [], []

        for s in range(0, max(1, len(samples) - seg_samples), hop_samples):
            chunk = samples[s: s + seg_samples]
            if len(chunk) < seg_samples // 2:
                break
            t_tensor = torch.FloatTensor(chunk).unsqueeze(0)
            with torch.no_grad():
                emb = classifier.encode_batch(t_tensor).squeeze().numpy()
            embeddings.append(emb)
            times.append(s / sr)

        if len(embeddings) < n_speakers:
            return []

        X = np.stack(embeddings)
        clust = Cluster(n_clusters=n_speakers, metric="cosine", linkage="average")
        labels = clust.fit_predict(X)

        return NumpyAcousticDiarizer._labels_to_entries(labels, times, offset, n_speakers)


# ---------------------------------------------------------------------------
# Tier 3: PyAnnoteDiarizer  (optional, token required)
# ---------------------------------------------------------------------------

class PyAnnoteDiarizer(DiarizeBase):
    """Diarizer using pyannote.audio.

    Requires:  pip install pyannote.audio
    Requires a HuggingFace access token with the pyannote/speaker-diarization
    model accepted.
    """

    def __init__(self, token: str = ""):
        self._token = token or os.environ.get("HUGGINGFACE_TOKEN", "")

    def describe(self) -> str:
        return "PyAnnoteDiarizer — pyannote.audio pipeline (HIGHEST QUALITY, needs HF token)"

    def diarize(
        self,
        source: str,
        num_speakers: int = 2,
        time_range: Optional[tuple[float, float]] = None,
    ) -> List[AudioDiarizationEntry]:
        avail, msg = _check_pyannote()
        if not avail:
            raise ImportError(
                f"PyAnnoteDiarizer requires pyannote.audio.\n"
                f"Install: {msg}\n"
                "Also accept the model at: https://hf.co/pyannote/speaker-diarization-3.1"
            )

        if not self._token:
            raise ValueError(
                "PyAnnoteDiarizer requires a HuggingFace token.\n"
                "Set token= in the constructor or HUGGINGFACE_TOKEN env var.\n"
                "Request access at: https://hf.co/pyannote/speaker-diarization-3.1"
            )

        from pyannote.audio import Pipeline
        import torch

        start = time_range[0] if time_range else None
        end = time_range[1] if time_range else None
        wav_path = _tmp_wav("pa_diar")
        try:
            _extract_wav(source, wav_path, start, end)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._token,
            )
            result = pipeline(wav_path, num_speakers=num_speakers)
            entries = []
            for turn, _, speaker in result.itertracks(yield_label=True):
                entries.append(AudioDiarizationEntry(
                    speaker=speaker,
                    start=(start or 0.0) + turn.start,
                    end=(start or 0.0) + turn.end,
                ))
            return sorted(entries, key=lambda e: e.start)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DIARIZER_REGISTRY = {
    "numpy_acoustic": NumpyAcousticDiarizer,
    "speechbrain": SpeechBrainDiarizer,
    "pyannote": PyAnnoteDiarizer,
}


def get_diarizer(name: str = "numpy_acoustic", **kwargs) -> DiarizeBase:
    """Instantiate a diarizer by name.

    Parameters
    ----------
    name:
        One of ``"numpy_acoustic"`` (default), ``"speechbrain"``,
        ``"pyannote"``.
    **kwargs:
        Forwarded to the backend constructor (e.g. ``token=`` for pyannote).

    Raises
    ------
    ValueError
        Unknown backend name.
    ImportError
        Optional dependencies missing.
    """
    if name not in _DIARIZER_REGISTRY:
        available = ", ".join(_DIARIZER_REGISTRY)
        raise ValueError(
            f"Unknown diarization backend {name!r}. "
            f"Available: {available}"
        )
    return _DIARIZER_REGISTRY[name](**kwargs)


def list_diarizers() -> dict:
    """Return dict of name → availability status."""
    status = {}
    checks = {
        "numpy_acoustic": _check_numpy,
        "speechbrain": _check_speechbrain,
        "pyannote": _check_pyannote,
    }
    for name, check_fn in checks.items():
        avail, msg = check_fn()
        status[name] = {"available": avail, "install": msg if not avail else ""}
    return status


def aggregate_speaker_ranges(
    entries: List[AudioDiarizationEntry],
) -> List[AudioDiarizationEntry]:
    """Return one first-to-last candidate range per anonymous speaker."""
    bounds: dict[str, tuple[float, float]] = {}
    for entry in entries:
        if entry.speaker in bounds:
            first, last = bounds[entry.speaker]
            bounds[entry.speaker] = (min(first, entry.start), max(last, entry.end))
        else:
            bounds[entry.speaker] = (entry.start, entry.end)
    return [
        AudioDiarizationEntry(speaker, first, last)
        for speaker, (first, last) in sorted(bounds.items(), key=lambda item: item[1][0])
    ]


def format_diarization_timeline(
    entries: List[AudioDiarizationEntry],
    total_duration: float,
    backend: str = "unknown",
    range_start: float = 0.0,
) -> str:
    """Format a list of AudioDiarizationEntry objects as a display timeline.

    Returns a multi-line box-drawing string similar to the transcript
    timeline used in the time-range confirmation workflow.
    """
    from collections import defaultdict
    from zhrreport.utils import fmt_sec, fmt_duration

    if not entries:
        return "[No diarization entries]"

    # Compute per-speaker stats
    first_seen: dict = {}
    last_seen: dict = {}
    total_dur: dict = defaultdict(float)
    count: dict = defaultdict(int)
    for e in entries:
        if e.speaker not in first_seen:
            first_seen[e.speaker] = e.start
        last_seen[e.speaker] = max(last_seen.get(e.speaker, 0.0), e.end)
        total_dur[e.speaker] += e.duration
        count[e.speaker] += 1

    speaker_ranges = aggregate_speaker_ranges(entries)
    speakers = [entry.speaker for entry in speaker_ranges]

    W = 70
    border_top = "╔" + "═" * W + "╗"
    border_mid = "╠" + "═" * W + "╣"
    border_bot = "╚" + "═" * W + "╝"

    def row(text: str) -> str:
        return "║" + f"  {text}".ljust(W) + "║"

    lines = [border_top]
    lines.append(row(f"Diarization — {backend}  ⚠ ANONYMOUS labels"))
    range_end = range_start + total_duration
    lines.append(row(f"Duration: {fmt_sec(total_duration)}   Speakers detected: {len(speakers)}"))
    lines.append(border_mid)
    lines.append(row(f"{'#':<4}{'Speaker':<16}{'First seen':<14}{'Last seen':<14}{'Speaking':<12}{'Turns'}"))
    for i, sp in enumerate(speakers, 1):
        lines.append(row(
            f"{i:<4}{sp:<16}"
            f"{fmt_sec(first_seen[sp]):<14}"
            f"{fmt_sec(last_seen[sp]):<14}"
            f"{fmt_duration(total_dur[sp]):<12}"
            f"{count[sp]}"
        ))
    lines.append(border_mid)
    lines.append(row("Suggested ranges — confirm one before ASR begins"))
    range_label = "Full recording" if range_start == 0.0 else "Analyzed range"
    lines.append(row(
        f"  A) {range_label:<16}{fmt_sec(range_start)} - {fmt_sec(range_end)}"
    ))
    for i, sp in enumerate(speakers, ord('B')):
        lines.append(row(
            f"  {chr(i)}) {sp:<17}"
            f"{fmt_sec(first_seen[sp])} - {fmt_sec(last_seen[sp])}"
        ))
    lines.append(border_bot)
    return "\n".join(lines)
