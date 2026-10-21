"""Sample end-to-end run for zhrreport.

This script exercises the full pipeline using the bundled
``sample_transcript.json`` fixture — no ASR credentials or audio file
required.

Run from the project root::

    python sample/run_sample.py

Outputs are written to ``sample/output/``.
"""
from __future__ import annotations

import os
import sys
import zipfile

# Make sure the project root is importable when run as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from zhrreport.cli import main as cli_main

SAMPLE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT  = os.path.join(SAMPLE_DIR, "sample_transcript.json")
OUTPUT_DIR  = os.path.join(SAMPLE_DIR, "output")


def run_e2e() -> None:
    print("=" * 68)
    print("  zhrreport — Sample End-to-End Run")
    print("=" * 68)
    print(f"\n  Transcript : {TRANSCRIPT}")
    print(f"  Output dir : {OUTPUT_DIR}\n")

    # Run the CLI with --confirm-full so it doesn't try to prompt
    exit_code = cli_main([
        "report",
        TRANSCRIPT,
        "--backend", "json",
        "--confirm-full",
        "--output-dir", OUTPUT_DIR,
        "--chunk-duration", "300",
        "--timeout", "60",
    ])

    if exit_code != 0:
        print(f"\n[FAIL] CLI returned exit code {exit_code}", file=sys.stderr)
        sys.exit(exit_code)

    # Inspect generated files
    print("\n" + "=" * 68)
    print("  Output file inspection")
    print("=" * 68)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        ext = os.path.splitext(fname)[1].lower()
        notes = ""
        if ext == ".docx":
            # Verify ZIP / DOCX integrity
            try:
                with zipfile.ZipFile(fpath, "r") as zf:
                    parts = zf.namelist()
                required = {"[Content_Types].xml", "_rels/.rels",
                            "word/document.xml", "word/styles.xml"}
                missing = required - set(parts)
                if missing:
                    notes = f"  ⚠ MISSING: {missing}"
                else:
                    notes = f"  ✔ valid DOCX ZIP ({len(parts)} parts)"
            except zipfile.BadZipFile:
                notes = "  ✘ BAD ZIP"
        print(f"  {fname:<35}  {size:>8,} bytes{notes}")

    print("\n  ✔ Sample run complete.\n")


if __name__ == "__main__":
    run_e2e()
