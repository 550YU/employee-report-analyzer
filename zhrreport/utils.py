"""Shared utilities for zhrreport."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys


def fmt_sec(t: float) -> str:
    """Format seconds as HH:MM:SS."""
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_duration(t: float) -> str:
    """Format seconds as human-readable duration (e.g. '1h 02m 35s')."""
    t = max(0.0, t)
    if t < 60:
        return f"{int(t)}s"
    if t < 3600:
        return f"{int(t // 60)}m {int(t % 60):02d}s"
    return f"{int(t // 3600)}h {int((t % 3600) // 60):02d}m {int(t % 60):02d}s"


def ensure_dir(path: str) -> str:
    """Create directory if missing; return path."""
    os.makedirs(path, exist_ok=True)
    return path


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and every child process.

    Uses ``taskkill /F /T`` on Windows; ``SIGKILL`` on the session's process
    group on POSIX.  Falls back to ``proc.kill()`` if anything goes wrong.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=2,
            )
        else:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def xml_escape(s: str) -> str:
    """Escape a string for safe embedding in XML text content."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )


def safe_filename(name: str) -> str:
    """Convert *name* to a filesystem-safe filename (no path separators etc.)."""
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    safe = safe.strip(". ")
    return safe or "unknown"


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe discovery (no hardcoded paths)
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    """Find the ffmpeg executable.

    Search order:
    1. ``ffmpeg`` on ``PATH``
    2. ``imageio-ffmpeg`` bundled binary (if the package is installed)

    Raises
    ------
    RuntimeError
        When ffmpeg cannot be located by either method, with a friendly
        install message.
    """
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg not found on PATH and imageio-ffmpeg is not installed.\n"
        "Fix with one of:\n"
        "  pip install imageio-ffmpeg          # lightweight Python package\n"
        "  # or install ffmpeg system-wide:\n"
        "  #   Windows: https://ffmpeg.org/download.html  (add to PATH)\n"
        "  #   macOS:   brew install ffmpeg\n"
        "  #   Linux:   apt-get install ffmpeg"
    )


def find_ffprobe() -> str | None:
    """Find ffprobe or return ``None`` if unavailable.

    Callers should fall back to parsing ``ffmpeg -i`` stderr when this
    returns ``None``.
    """
    import shutil
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    # Check next to imageio_ffmpeg's ffmpeg binary
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        probe_name = "ffprobe" + (".exe" if sys.platform == "win32" else "")
        candidate = os.path.join(os.path.dirname(ffmpeg_path), probe_name)
        if os.path.isfile(candidate):
            return candidate
    except ImportError:
        pass
    return None


def probe_duration(source: str) -> float:
    """Return duration (seconds) of *source* audio/video file.

    Uses ffprobe when available; falls back to parsing ``ffmpeg -i`` stderr.

    Raises
    ------
    RuntimeError
        When ffmpeg/ffprobe is not available or duration cannot be parsed.
    """
    ffprobe = find_ffprobe()
    if ffprobe:
        r = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                source,
            ],
            capture_output=True, text=True, timeout=30,
        )
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass

    # Fallback: parse "Duration: HH:MM:SS.ms" from ffmpeg -i stderr
    ffmpeg = find_ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-i", source],
        capture_output=True, text=True, timeout=30,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", r.stderr)
    if m:
        h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mn * 60 + s + cs / 100.0
    raise RuntimeError(
        f"Could not determine duration of {source!r}. "
        "Ensure ffmpeg is available (see find_ffmpeg)."
    )
