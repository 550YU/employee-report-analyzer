"""Subprocess worker entry point for chunked transcription.

Called by :class:`~zhrreport.chunker.ChunkedTranscriber` as::

    python -m zhrreport._worker \\
        --backend fixture \\
        --source path/to/file \\
        --start 0.0 \\
        --end 300.0 \\
        --chunk-id 0

Outputs a JSON array of TranscriptSegment dicts to **stdout**.
All diagnostic / error messages go to **stderr** so they don't
corrupt the JSON stream.

This separation means the parent process can unconditionally parse stdout
as JSON and inspect ``returncode`` to detect failure.
"""
from __future__ import annotations

import argparse
import json
import sys


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="zhrreport chunk transcription worker (internal use)"
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument(
        "--backend-kwarg",
        action="append",
        dest="backend_kwargs",
        default=[],
        metavar="KEY=VALUE",
        help="Extra kwargs forwarded to the backend constructor.",
    )
    args = parser.parse_args(argv)

    # Parse backend kwargs
    bkwargs: dict = {}
    for kv in args.backend_kwargs:
        if "=" not in kv:
            print(f"[worker] Invalid --backend-kwarg {kv!r} (expected KEY=VALUE)", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        bkwargs[k] = v

    # Lazy imports after argument parsing so invalid args fail fast.
    try:
        from zhrreport.models import TimeRange
        from zhrreport.transcribers import get_backend
    except ImportError as exc:
        print(f"[worker] Import error: {exc}", file=sys.stderr)
        return 2

    try:
        backend = get_backend(args.backend, **bkwargs)
    except ValueError as exc:
        print(f"[worker] Backend error: {exc}", file=sys.stderr)
        return 3

    time_range = TimeRange(args.start, args.end)

    try:
        segments = backend.transcribe_chunk(
            source=args.source,
            time_range=time_range,
            chunk_id=args.chunk_id,
        )
    except Exception as exc:
        print(f"[worker] Transcription error: {exc}", file=sys.stderr)
        return 4

    # Emit JSON array to stdout
    payload = [seg.to_dict() for seg in segments]
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
