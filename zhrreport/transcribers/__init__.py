"""Transcribers sub-package."""
from zhrreport.transcribers.base import TranscriberBase
from zhrreport.transcribers.fixture import FixtureTranscriber
from zhrreport.transcribers.json_backend import JsonTranscriber

BACKENDS: dict = {
    "fixture": FixtureTranscriber,
    "json": JsonTranscriber,
}

# Optional: faster-whisper backend (requires pip install faster-whisper)
try:
    from zhrreport.transcribers.faster_whisper_backend import (
        FasterWhisperTranscriber,
        check_availability as _fw_avail,
    )
    BACKENDS["faster-whisper"] = FasterWhisperTranscriber
except ImportError:
    pass  # faster-whisper not installed


def get_backend(name: str, **kwargs) -> TranscriberBase:
    """Return an instantiated backend by *name*.

    Raises
    ------
    ValueError
        If *name* is not a registered backend.
    ImportError
        If the backend's optional dependencies are missing (e.g. faster-whisper).
    """
    if name not in BACKENDS:
        available = ", ".join(BACKENDS)
        raise ValueError(
            f"Unknown transcription backend {name!r}. "
            f"Available backends: {available}"
        )
    return BACKENDS[name](**kwargs)
