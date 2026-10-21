"""End-to-end integration tests using the sample transcript."""
import json
import os
import sys
import unittest
import zipfile

# Ensure project root is importable
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from zhrreport.cli import main as cli_main
from zhrreport.utils import ensure_dir

_SAMPLE_JSON = os.path.join(_PROJ, "sample", "sample_transcript.json")
_OUT = os.path.join(os.path.dirname(__file__), "_e2e_output")


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_dir(_OUT)
        assert os.path.isfile(_SAMPLE_JSON), (
            f"Sample transcript not found: {_SAMPLE_JSON}"
        )

    # ------------------------------------------------------------------
    # CLI: confirm-full
    # ------------------------------------------------------------------

    def test_confirm_full_exit_zero(self):
        rc = cli_main([
            "report", _SAMPLE_JSON,
            "--backend", "json",
            "--confirm-full",
            "--output-dir", _OUT,
        ])
        self.assertEqual(rc, 0)

    def test_output_files_created(self):
        rc = cli_main([
            "report", _SAMPLE_JSON,
            "--backend", "json",
            "--confirm-full",
            "--output-dir", _OUT,
        ])
        self.assertEqual(rc, 0)
        files = os.listdir(_OUT)
        self.assertTrue(any(f.endswith(".md") for f in files))
        self.assertTrue(any(f.endswith(".docx") for f in files))

    # ------------------------------------------------------------------
    # CLI: explicit range
    # ------------------------------------------------------------------

    def test_explicit_range_exit_zero(self):
        rc = cli_main([
            "report", _SAMPLE_JSON,
            "--backend", "json",
            "--range", "00:00:00-00:06:45",
            "--output-dir", _OUT,
        ])
        self.assertEqual(rc, 0)

    # ------------------------------------------------------------------
    # CLI: no range / non-interactive → exit code 2
    # ------------------------------------------------------------------

    def test_no_range_non_interactive_exits_nonzero(self):
        """Without --confirm-full or --range in a non-interactive test, CLI must fail."""
        # The test runner has no TTY so confirm_time_range will raise TimeRangeRequired
        rc = cli_main([
            "report", _SAMPLE_JSON,
            "--backend", "json",
            "--output-dir", _OUT,
        ])
        self.assertEqual(rc, 2)

    # ------------------------------------------------------------------
    # CLI: timeline sub-command
    # ------------------------------------------------------------------

    def test_timeline_exit_zero(self):
        rc = cli_main(["timeline", _SAMPLE_JSON])
        self.assertEqual(rc, 0)

    # ------------------------------------------------------------------
    # CLI: backends sub-command
    # ------------------------------------------------------------------

    def test_backends_exit_zero(self):
        rc = cli_main(["backends"])
        self.assertEqual(rc, 0)

    def test_hallucination_command_exit_zero(self):
        rc = cli_main(["check-hallucination", _SAMPLE_JSON])
        self.assertEqual(rc, 0)

    def test_audio_hallucination_requires_bounded_range(self):
        rc = cli_main(["check-hallucination", __file__])
        self.assertEqual(rc, 2)

    def test_representative_ky_tf_outputs(self):
        rc = cli_main([
            "report", _SAMPLE_JSON, "--confirm-full", "--output-dir", _OUT,
        ])
        self.assertEqual(rc, 0)
        for speaker in ("KY", "TF"):
            self.assertTrue(os.path.isfile(os.path.join(_OUT, speaker + ".md")))
            self.assertTrue(os.path.isfile(os.path.join(_OUT, speaker + ".docx")))

    # ------------------------------------------------------------------
    # DOCX validity
    # ------------------------------------------------------------------

    def test_docx_files_are_valid_zips(self):
        ensure_dir(_OUT)
        cli_main([
            "report", _SAMPLE_JSON,
            "--backend", "json",
            "--confirm-full",
            "--output-dir", _OUT,
        ])
        docx_files = [f for f in os.listdir(_OUT) if f.endswith(".docx")]
        self.assertGreater(len(docx_files), 0, "No DOCX files produced")
        for fname in docx_files:
            fpath = os.path.join(_OUT, fname)
            self.assertTrue(zipfile.is_zipfile(fpath), f"Not a valid ZIP: {fname}")

    def test_docx_required_parts(self):
        ensure_dir(_OUT)
        cli_main([
            "report", _SAMPLE_JSON,
            "--backend", "json",
            "--confirm-full",
            "--output-dir", _OUT,
        ])
        required = {
            "[Content_Types].xml", "_rels/.rels",
            "word/document.xml", "word/_rels/document.xml.rels",
            "word/styles.xml",
        }
        docx_files = [f for f in os.listdir(_OUT) if f.endswith(".docx")]
        for fname in docx_files:
            fpath = os.path.join(_OUT, fname)
            with zipfile.ZipFile(fpath, "r") as zf:
                names = set(zf.namelist())
            missing = required - names
            self.assertEqual(
                missing, set(), f"{fname}: missing parts {missing}"
            )

    # ------------------------------------------------------------------
    # Hallucination flags present for sample data
    # ------------------------------------------------------------------

    def test_sample_has_hallucination_flags(self):
        """The sample transcript includes known-bad phrases that must be flagged."""
        from zhrreport.models import TranscriptResult
        from zhrreport.hallucination import detect_hallucinations

        result = TranscriptResult.from_json_file(_SAMPLE_JSON)
        flags = detect_hallucinations(result.segments)
        flag_types = {f.flag_type for f in flags}
        # Sample has "Thank you for watching" → known_phrase
        self.assertIn("known_phrase", flag_types)
        # Sample has repeated segment
        self.assertIn("repeated_segment", flag_types)

    # ------------------------------------------------------------------
    # Worker process
    # ------------------------------------------------------------------

    def test_worker_process_output(self):
        """Running _worker as a subprocess must return valid JSON."""
        import subprocess
        proc = subprocess.run(
            [
                sys.executable, "-m", "zhrreport._worker",
                "--backend", "fixture",
                "--source", "ignored",
                "--start", "0.0",
                "--end", "100.0",
                "--chunk-id", "0",
            ],
            capture_output=True,
            text=True,
            cwd=_PROJ,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("text", data[0])

    def test_worker_bad_backend_nonzero_exit(self):
        """Worker with unknown backend must exit non-zero."""
        import subprocess
        proc = subprocess.run(
            [
                sys.executable, "-m", "zhrreport._worker",
                "--backend", "nonexistent_xyz",
                "--source", "x",
                "--start", "0.0",
                "--end", "10.0",
            ],
            capture_output=True, text=True, cwd=_PROJ,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("nonexistent_xyz", proc.stderr)

    # ------------------------------------------------------------------
    # fixture backend produces reports
    # ------------------------------------------------------------------

    def test_fixture_backend_report(self):
        rc = cli_main([
            "report", "ignored_source",
            "--backend", "fixture",
            "--confirm-full",
            "--output-dir", _OUT,
            "--chunk-duration", "300",
        ])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
