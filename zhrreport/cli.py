"""Main CLI for zhrreport.

Usage
-----
Generate reports from a JSON transcript::

    python -m zhrreport.cli report meeting.json --confirm-full --output-dir ./out

Generate reports with a time range::

    python -m zhrreport.cli report meeting.json --range 00:05:00-00:25:00

Transcribe from a real audio file (faster-whisper backend)::

    python -m zhrreport.cli report meeting.mp3 \\
        --backend faster-whisper --range 00:05:26-00:19:01 \\
        --speaker-name KY --output-dir ./out

    # No range → diarize first, print anonymous timeline, then confirm:
    python -m zhrreport.cli report meeting.mp3 --backend faster-whisper

Diarize an audio clip to preview speakers::

    python -m zhrreport.cli diarize meeting.mp3 --range 18:30-20:30

List available backends::

    python -m zhrreport.cli backends
    python -m zhrreport.cli diarizers
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirm_or_fail(msg: str) -> None:
    """Print error to stderr and exit 1."""
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: backends
# ---------------------------------------------------------------------------

def _cmd_backends(_args) -> int:
    from zhrreport.transcribers import BACKENDS
    print("Available transcription backends:")
    for name, cls in BACKENDS.items():
        try:
            b = cls()
            desc = b.describe()
        except Exception as exc:
            desc = f"(unavailable: {exc})"
        print(f"  {name:<16}  {desc}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: diarizers
# ---------------------------------------------------------------------------

def _cmd_diarizers(_args) -> int:
    from zhrreport.local_diarizer import list_diarizers
    print("Available diarization backends:")
    for name, info in list_diarizers().items():
        status = "✔ available" if info["available"] else f"✘ missing ({info['install']})"
        print(f"  {name:<20}  {status}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: diarize
# ---------------------------------------------------------------------------

def _cmd_diarize(args) -> int:
    """Run local diarization on an audio file and print the timeline."""
    from zhrreport.local_diarizer import (
        aggregate_speaker_ranges,
        format_diarization_timeline,
        get_diarizer,
    )
    from zhrreport.utils import probe_duration

    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        print(f"[ERROR] File not found: {source!r}", file=sys.stderr)
        return 1

    backend = getattr(args, "diarization_backend", "numpy_acoustic") or "numpy_acoustic"
    num_speakers = getattr(args, "num_speakers", 2) or 2
    range_str = getattr(args, "range", None)

    time_range = None
    if range_str:
        from zhrreport.models import TimeRange
        tr = TimeRange.from_string(range_str)
        time_range = (tr.start, tr.end)

    print(f"\n  ▶ Diarizing {os.path.basename(source)!r} …")
    print(f"  ▶ Backend  : {backend}")
    if time_range:
        print(f"  ▶ Range    : {time_range[0]:.1f}s – {time_range[1]:.1f}s")

    try:
        total_dur = probe_duration(source)
    except Exception:
        total_dur = time_range[1] if time_range else 0.0

    t0 = _time_mod.monotonic()
    diarizer = get_diarizer(backend)
    entries = diarizer.diarize(source, num_speakers=num_speakers, time_range=time_range)
    elapsed = _time_mod.monotonic() - t0

    clip_dur = (time_range[1] - time_range[0]) if time_range else total_dur
    range_start = time_range[0] if time_range else 0.0
    timeline = format_diarization_timeline(entries, clip_dur, backend, range_start)
    print(timeline)
    print(f"\n  ▶ Diarization done in {elapsed:.1f}s  ({len(entries)} turns)")

    # Persist timeline to file if --output-dir given
    if getattr(args, "output_dir", None):
        from zhrreport.utils import ensure_dir
        out = os.path.abspath(args.output_dir)
        ensure_dir(out)
        tl_path = os.path.join(out, "diarization_timeline.txt")
        with open(tl_path, "w", encoding="utf-8") as fh:
            fh.write(timeline + "\n")
            fh.write(f"\nBackend  : {backend}\n")
            fh.write(f"Duration : {clip_dur:.1f}s\n")
            fh.write(f"Elapsed  : {elapsed:.1f}s\n")
        print(f"  ✔ Timeline saved to: {tl_path}")

    return 0


# ---------------------------------------------------------------------------
# Sub-command: timeline
# ---------------------------------------------------------------------------

def _cmd_timeline(args) -> int:
    from zhrreport.diarizer import build_diarization, format_timeline
    from zhrreport.models import TranscriptResult

    source = args.source
    if not os.path.isabs(source):
        source = os.path.abspath(source)

    if not os.path.exists(source):
        print(f"[ERROR] File not found: {source!r}", file=sys.stderr)
        return 1

    ext = os.path.splitext(source)[1].lower()
    if ext != ".json":
        print(
            f"[ERROR] timeline command requires a .json transcript file, "
            f"got {ext!r}.\n"
            "Transcribe your audio first, then pass the resulting .json.",
            file=sys.stderr,
        )
        return 1

    result = TranscriptResult.from_json_file(source)
    entries = build_diarization(result.segments)
    print(format_timeline(entries, result.duration))
    return 0


def _cmd_check_hallucination(args) -> int:
    from zhrreport.chunker import ChunkTranscriptionError, ChunkedTranscriber
    from zhrreport.hallucination import detect_hallucinations, summarise_flags
    from zhrreport.models import TimeRange, TranscriptResult
    from zhrreport.utils import probe_duration

    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        print(f"[ERROR] File not found: {source!r}", file=sys.stderr)
        return 1
    if os.path.splitext(source)[1].lower() == ".json":
        try:
            result = TranscriptResult.from_json_file(source)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[ERROR] Cannot read transcript: {exc}", file=sys.stderr)
            return 1
    else:
        if not args.range:
            print(
                "[ERROR] Audio hallucination checks require --range START-END "
                "to keep ASR bounded.",
                file=sys.stderr,
            )
            return 2
        try:
            time_range = TimeRange.from_string(args.range)
            full_duration = probe_duration(source)
            backend_kwargs = {
                "model": args.model or "base",
                "speaker_name": args.speaker_name or "SPEAKER",
            }
            if args.fallback_model:
                backend_kwargs["fallback_model"] = args.fallback_model
            if args.language:
                backend_kwargs["language"] = args.language
            result = ChunkedTranscriber(
                "faster-whisper",
                backend_kwargs=backend_kwargs,
                chunk_duration=args.chunk_duration,
                timeout=args.timeout,
            ).run(source, full_duration, time_range)
        except (ValueError, OSError, ImportError, ChunkTranscriptionError) as exc:
            print(f"[ERROR] Cannot transcribe audio range: {exc}", file=sys.stderr)
            return 1

    flags = detect_hallucinations(result.segments)
    print(summarise_flags(flags))
    for flag in flags:
        print(
            f"  [{flag.severity.upper()}] {flag.start:.1f}s "
            f"{flag.speaker}: {flag.flag_type} - {flag.message}"
        )
    return 0


# ---------------------------------------------------------------------------
# Sub-command: report
# ---------------------------------------------------------------------------

def _cmd_report(args) -> int:
    import textwrap
    from zhrreport.chunker import ChunkedTranscriber, DEFAULT_CHUNK_DURATION, DEFAULT_TIMEOUT
    from zhrreport.hallucination import detect_hallucinations, summarise_flags
    from zhrreport.models import TimeRange, TranscriptResult
    from zhrreport.report_docx import write_docx_reports
    from zhrreport.report_md import write_markdown_reports
    from zhrreport.timeline import TimeRangeRequired, confirm_time_range
    from zhrreport.utils import fmt_duration

    source = args.source
    if not os.path.isabs(source):
        source = os.path.abspath(source)

    ext = os.path.splitext(source)[1].lower()
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = "json" if ext == ".json" else "faster-whisper"

    is_fixture = backend_name == "fixture"
    is_json = backend_name == "json" and ext == ".json"

    if not is_fixture and not os.path.exists(source):
        print(f"[ERROR] File not found: {source!r}", file=sys.stderr)
        return 1

    chunk_dur = getattr(args, "chunk_duration", DEFAULT_CHUNK_DURATION) or DEFAULT_CHUNK_DURATION
    timeout = getattr(args, "timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT

    # Build backend kwargs for faster-whisper
    backend_kwargs: dict = {}
    if backend_name == "json":
        backend_kwargs["json_path"] = source
    elif backend_name == "faster-whisper":
        if getattr(args, "model", None):
            backend_kwargs["model"] = args.model
        if getattr(args, "fallback_model", None):
            backend_kwargs["fallback_model"] = args.fallback_model
        if getattr(args, "language", None):
            backend_kwargs["language"] = args.language
        if getattr(args, "speaker_name", None):
            backend_kwargs["speaker_name"] = args.speaker_name

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    print(f"\n  ▶ Source  : {source}")
    print(f"  ▶ Backend : {backend_name}")
    if backend_name == "faster-whisper":
        model_str = backend_kwargs.get("model", "base")
        fb = backend_kwargs.get("fallback_model", "")
        sp = backend_kwargs.get("speaker_name", "SPEAKER")
        print(f"  ▶ Model   : {model_str}" + (f"  fallback={fb}" if fb else ""))
        print(f"  ▶ Speaker : {sp}")

    # ==================================================================
    # AUDIO PATH: diarize-first for non-JSON audio without explicit range
    # ==================================================================
    range_str = getattr(args, "range", None)
    confirm_full = getattr(args, "confirm_full", False)
    diar_backend = getattr(args, "diarization_backend", "numpy_acoustic") or "numpy_acoustic"
    num_speakers = getattr(args, "num_speakers", 2) or 2

    if not is_fixture and not is_json and ext != ".json":
        if not range_str and not confirm_full:
            # --- Diarize first, then confirm ---
            print(f"\n  ▶ No range specified — running local diarization first …")
            print(f"  ▶ Diarization backend: {diar_backend}")
            from zhrreport.local_diarizer import get_diarizer, format_diarization_timeline
            from zhrreport.utils import probe_duration
            try:
                total_dur = probe_duration(source)
            except Exception as exc:
                print(f"[ERROR] Cannot probe audio duration: {exc}", file=sys.stderr)
                return 1

            # The lightweight local backend is bounded by the recording length
            # and must inspect the full meeting so later speakers are not missed.
            diar_clip_end = total_dur
            diar_range = (0.0, diar_clip_end)
            print(f"  ▶ Diarizing full {diar_clip_end:.0f}s recording …")

            t0 = _time_mod.monotonic()
            try:
                diarizer = get_diarizer(diar_backend)
                entries = diarizer.diarize(
                    source, num_speakers=num_speakers, time_range=diar_range
                )
            except Exception as exc:
                print(f"[ERROR] Diarization failed: {exc}", file=sys.stderr)
                return 1
            elapsed = _time_mod.monotonic() - t0
            timeline = format_diarization_timeline(entries, diar_clip_end, diar_backend)
            speaker_ranges = aggregate_speaker_ranges(entries)
            print(timeline)
            print(f"\n  ▶ Diarization done in {elapsed:.1f}s")

            # Non-interactive: require explicit range
            if not sys.stdin.isatty():
                print(
                    "\n[ERROR] No --range provided and stdin is not a TTY.\n"
                    "  Provide --range START-END to transcribe a specific speaker range.\n"
                    "  Example:  --range 00:05:26-00:19:01\n"
                    "  Or use --confirm-full to process the entire recording.\n"
                    "  Diarization above shows detected speaker turns.",
                    file=sys.stderr,
                )
                return 2

            # Interactive: prompt
            print()
            print("  Enter range to transcribe (or Ctrl+C to abort):")
            print("  Examples:  00:05:26-00:19:01   or   326-1141")
            while True:
                try:
                    answer = input("  Your choice: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[ABORT]")
                    return 2
                if not answer:
                    continue
                upper = answer.upper()
                if upper in ("A", "FULL"):
                    range_str = f"0-{total_dur}"
                    break
                idx = ord(upper) - ord('B')
                if len(upper) == 1 and upper.isalpha() and 0 <= idx < len(speaker_ranges):
                    e = speaker_ranges[idx]
                    range_str = f"{e.start}-{e.end}"
                    break
                try:
                    from zhrreport.models import TimeRange as _TR
                    _TR.from_string(answer)
                    range_str = answer
                    break
                except ValueError as exc:
                    print(f"  Cannot parse {answer!r}: {exc}. Try again.")

    # ==================================================================
    # TRANSCRIPTION
    # ==================================================================

    t_transcribe_start = _time_mod.monotonic()

    if is_fixture:
        from zhrreport.transcribers.fixture import FixtureTranscriber
        _all_segs = FixtureTranscriber().transcribe_chunk("x")
        full_duration_fixture = max(s.end for s in _all_segs) if _all_segs else 1800.0
        print(f"  ▶ Fixture backend — synthetic duration {full_duration_fixture:.0f}s")

        def _progress(idx, total, elapsed):
            print(f"     Chunk {idx + 1}/{total} done  ({elapsed:.1f}s)", end="\r", flush=True)

        ct = ChunkedTranscriber(
            backend_name="fixture",
            chunk_duration=chunk_dur,
            timeout=timeout,
            progress_cb=_progress,
        )
        try:
            result = ct.run(source, full_duration_fixture)
        except RuntimeError as exc:
            print(f"[ERROR] Transcription failed: {exc}", file=sys.stderr)
            return 4
        print()
        print(f"     {len(result.segments)} segments loaded.")

    elif is_json or (ext == ".json" and backend_name == "json"):
        print("  ▶ Loading pre-existing JSON transcript …")
        result = TranscriptResult.from_json_file(source)
        print(f"     {len(result.segments)} segments, {len(result.speakers())} speaker(s), "
              f"duration {fmt_duration(result.duration)}")

    else:
        # Determine full duration for chunking
        if ext == ".wav":
            from zhrreport.audio_chunker import get_wav_duration
            full_duration = get_wav_duration(source)
        elif ext == ".json":
            tmp = TranscriptResult.from_json_file(source)
            full_duration = tmp.duration
        else:
            from zhrreport.utils import probe_duration
            try:
                full_duration = probe_duration(source)
            except Exception as exc:
                print(f"[ERROR] Cannot determine audio duration: {exc}", file=sys.stderr)
                return 1

        total_chunks = max(1, int(full_duration / chunk_dur) + 1)

        # If a range is already known, restrict transcription to that range
        # (avoids transcribing the full recording just to filter later)
        transcribe_range = None
        if range_str:
            from zhrreport.models import TimeRange as _TR2
            try:
                transcribe_range = _TR2.from_string(range_str)
                range_chunks = max(1, int(transcribe_range.duration / chunk_dur) + 1)
                print(
                    f"  ▶ Chunked transcription (range-scoped): "
                    f"~{range_chunks} chunk(s) of {fmt_duration(chunk_dur)}, "
                    f"timeout {timeout}s each"
                )
            except ValueError:
                transcribe_range = None
                print(
                    f"  ▶ Chunked transcription: "
                    f"~{total_chunks} chunk(s) of {fmt_duration(chunk_dur)}, "
                    f"timeout {timeout}s each"
                )
        elif confirm_full:
            print(
                f"  ▶ Chunked transcription: "
                f"~{total_chunks} chunk(s) of {fmt_duration(chunk_dur)}, "
                f"timeout {timeout}s each"
            )
        else:
            print(
                f"  ▶ Chunked transcription: "
                f"~{total_chunks} chunk(s) of {fmt_duration(chunk_dur)}, "
                f"timeout {timeout}s each"
            )

        def _progress(idx, total, elapsed):
            print(
                f"     Chunk {idx + 1}/{total} done  ({elapsed:.1f}s)",
                end="\r", flush=True,
            )

        ct = ChunkedTranscriber(
            backend_name=backend_name,
            backend_kwargs=backend_kwargs,
            chunk_duration=chunk_dur,
            timeout=timeout,
            progress_cb=_progress,
        )
        try:
            result = ct.run(source, full_duration, time_range=transcribe_range)
        except RuntimeError as exc:
            print(f"[ERROR] Transcription failed: {exc}", file=sys.stderr)
            return 4
        print()
        print(f"     {len(result.segments)} segments loaded.")

    t_transcribe_end = _time_mod.monotonic()

    # ------------------------------------------------------------------
    # Time-range confirmation
    # ------------------------------------------------------------------
    print("\n  ▶ Time-range confirmation …")
    try:
        time_range = confirm_time_range(
            segments=result.segments,
            recording_duration=result.duration,
            range_str=range_str,
            confirm_full=confirm_full,
        )
    except TimeRangeRequired as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"     Using range: {time_range}")

    result = result.filter_by_range(time_range)
    print(f"     {len(result.segments)} segment(s) in range, "
          f"{len(result.speakers())} speaker(s).")

    if not result.segments:
        print("[ERROR] No transcript segments found in the specified range.",
              file=sys.stderr)
        return 3

    # Apply --speaker-name rename (post-filter)
    speaker_name = getattr(args, "speaker_name", None)
    if speaker_name and result.segments:
        for seg in result.segments:
            seg.speaker = speaker_name

    # ------------------------------------------------------------------
    # Hallucination detection
    # ------------------------------------------------------------------
    if not args.skip_hallucination:
        print("\n  ▶ Hallucination detection …")
        flags = detect_hallucinations(result.segments)
        result.flags = flags
        summary = summarise_flags(flags)
        print(f"     {summary}")
        if flags:
            print("     (Details included in reports.)")
    else:
        print("\n  ▶ Hallucination detection skipped (--skip-hallucination).")

    # ------------------------------------------------------------------
    # Generate reports
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    print(f"\n  ▶ Writing reports to: {output_dir}")

    md_files = write_markdown_reports(result, time_range, output_dir)
    docx_files = write_docx_reports(result, time_range, output_dir)

    all_files = md_files + docx_files
    t_total = _time_mod.monotonic() - t_transcribe_start
    print(f"\n  ✔ Done — {len(all_files)} file(s) written  (total {t_total:.1f}s):")
    for f in all_files:
        size = os.path.getsize(f)
        print(f"      {os.path.relpath(f, output_dir):<30}  {size:>8,} bytes")

    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m zhrreport.cli",
        description=(
            "zhrreport — Generate per-speaker performance reports "
            "from meeting transcripts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m zhrreport.cli report meeting.json --confirm-full\n"
            "  python -m zhrreport.cli report meeting.json --range 00:05:00-00:25:00\n"
            "  python -m zhrreport.cli report meeting.mp3  --backend faster-whisper \\\n"
            "      --range 00:05:26-00:19:01 --speaker-name KY\n"
            "  python -m zhrreport.cli diarize meeting.mp3 --range 18:30-20:30\n"
            "  python -m zhrreport.cli timeline meeting.json\n"
            "  python -m zhrreport.cli backends\n"
            "  python -m zhrreport.cli diarizers\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- report --
    rep = sub.add_parser("report", help="Generate per-speaker reports")
    rep.add_argument("source", help="Input file (.json transcript or .wav/.mp3/…)")
    rep.add_argument(
        "--range", metavar="START-END", default=None,
        help="Time range, e.g. 00:05:00-00:25:00 or 300-1500",
    )
    rep.add_argument(
        "--confirm-full", action="store_true", default=False,
        help="Process the full recording without prompting for a range",
    )
    rep.add_argument(
        "--backend", default="auto", metavar="BACKEND",
        help="Transcription backend: auto (default) | json | fixture | faster-whisper",
    )
    rep.add_argument(
        "--model", default=None, metavar="MODEL",
        help="ASR model name for faster-whisper (default: base). "
             "E.g. Systran/faster-whisper-small",
    )
    rep.add_argument(
        "--fallback-model", dest="fallback_model", default=None, metavar="MODEL",
        help="Larger model to use when primary produces bad chunks (faster-whisper only)",
    )
    rep.add_argument(
        "--language", default=None, metavar="CODE",
        help="Optional Faster-Whisper language code, e.g. zh or en (default: auto-detect)",
    )
    rep.add_argument(
        "--diarization-backend", dest="diarization_backend",
        default="numpy_acoustic", metavar="BACKEND",
        help="Diarization backend for no-range audio workflow "
             "(numpy_acoustic | speechbrain | pyannote, default: numpy_acoustic)",
    )
    rep.add_argument(
        "--num-speakers", dest="num_speakers", type=int, default=2, metavar="N",
        help="Expected number of speakers for diarization (default: 2)",
    )
    rep.add_argument(
        "--speaker-name", dest="speaker_name", default=None, metavar="NAME",
        help="Label all segments with NAME (e.g. KY) for named per-speaker output",
    )
    rep.add_argument(
        "--output-dir", default="output", metavar="DIR",
        help="Directory to write reports into (default: ./output)",
    )
    rep.add_argument(
        "--chunk-duration", type=float, default=45.0, metavar="SECS",
        help="Chunk length in seconds for chunked transcription (default: 45)",
    )
    rep.add_argument(
        "--timeout", type=float, default=120.0, metavar="SECS",
        help="Hard per-chunk worker timeout in seconds (default: 120)",
    )
    rep.add_argument(
        "--skip-hallucination", action="store_true", default=False,
        help="Skip hallucination detection",
    )
    rep.set_defaults(func=_cmd_report)

    # -- diarize --
    di = sub.add_parser("diarize", help="Run local diarization and show anonymous speaker timeline")
    di.add_argument("source", help="Audio file (.mp3/.wav/…)")
    di.add_argument("--range", metavar="START-END", default=None,
                    help="Sub-range to diarize (default: full file)")
    di.add_argument("--diarization-backend", dest="diarization_backend",
                    default="numpy_acoustic", metavar="BACKEND",
                    help="numpy_acoustic (default) | speechbrain | pyannote")
    di.add_argument("--num-speakers", dest="num_speakers", type=int, default=2,
                    help="Expected number of speakers (default: 2)")
    di.add_argument("--output-dir", dest="output_dir", default=None,
                    help="Save diarization_timeline.txt here")
    di.set_defaults(func=_cmd_diarize)

    # -- timeline --
    tl = sub.add_parser("timeline", help="Show speaker timeline from a JSON transcript")
    tl.add_argument("source", help="JSON transcript file")
    tl.set_defaults(func=_cmd_timeline)

    hc = sub.add_parser(
        "check-hallucination",
        help="Check JSON or a bounded audio range for hallucination flags",
    )
    hc.add_argument("source", help="JSON transcript or MP3/WAV audio file")
    hc.add_argument("--range", metavar="START-END", default=None,
                    help="Required bounded range for audio, e.g. 05:26-07:26")
    hc.add_argument("--model", default=None, metavar="MODEL",
                    help="Faster-Whisper model for audio (default: base)")
    hc.add_argument("--fallback-model", dest="fallback_model", default=None,
                    metavar="MODEL", help="Fallback model for suspect audio chunks")
    hc.add_argument("--language", default=None, metavar="CODE",
                    help="Optional audio language code, e.g. zh or en")
    hc.add_argument("--speaker-name", dest="speaker_name", default=None,
                    metavar="NAME", help="Speaker label for the checked audio range")
    hc.add_argument("--chunk-duration", type=float, default=45.0, metavar="SECS",
                    help="Audio chunk length (default: 45)")
    hc.add_argument("--timeout", type=float, default=120.0, metavar="SECS",
                    help="Hard timeout per audio chunk (default: 120)")
    hc.set_defaults(func=_cmd_check_hallucination)

    # -- backends --
    bk = sub.add_parser("backends", help="List available transcription backends")
    bk.set_defaults(func=_cmd_backends)

    # -- diarizers --
    dz = sub.add_parser("diarizers", help="List available diarization backends")
    dz.set_defaults(func=_cmd_diarizers)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())