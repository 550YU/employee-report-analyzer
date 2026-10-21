"""DOCX report generator — no pandoc required.

Builds valid ``.docx`` (Office Open XML) files using only the Python
standard library :mod:`zipfile` and :mod:`xml.etree.ElementTree`.

A ``.docx`` file is a ZIP archive containing:

* ``[Content_Types].xml`` — MIME-type registry
* ``_rels/.rels`` — package relationships
* ``word/document.xml`` — body content
* ``word/_rels/document.xml.rels`` — document relationships
* ``word/styles.xml`` — paragraph and character styles

Generated files are validated by re-opening them with :mod:`zipfile`
after creation and asserting all required parts are present.
"""
from __future__ import annotations

import datetime
import io
import os
import zipfile
from typing import List

from zhrreport.diarizer import build_diarization
from zhrreport.models import HallucinationFlag, TimeRange, TranscriptResult, TranscriptSegment
from zhrreport.utils import ensure_dir, fmt_duration, fmt_sec, safe_filename, xml_escape

# ---------------------------------------------------------------------------
# OpenXML template strings
# ---------------------------------------------------------------------------

_CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml"/>
</Relationships>"""

_STYLES = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:pStyle w:val="Heading1"/>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="36"/>
      <w:szCs w:val="36"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:pStyle w:val="Heading2"/>
      <w:spacing w:before="200" w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:pStyle w:val="Heading3"/>
      <w:spacing w:before="160" w:after="60"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:i/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Quote">
    <w:name w:val="Quote"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:pStyle w:val="Quote"/>
      <w:ind w:left="720"/>
    </w:pPr>
    <w:rPr>
      <w:i/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableCell">
    <w:name w:val="TableCell"/>
    <w:basedOn w:val="Normal"/>
  </w:style>
</w:styles>"""

# XML namespaces
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------

class _DocBuilder:
    """Accumulate OpenXML paragraph strings for a document body."""

    def __init__(self) -> None:
        self._parts: List[str] = []

    def heading(self, text: str, level: int = 1) -> "_DocBuilder":
        style = f"Heading{level}"
        self._parts.append(
            f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r></w:p>'
        )
        return self

    def para(self, text: str, bold: bool = False, italic: bool = False) -> "_DocBuilder":
        rpr = ""
        if bold:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"
        rpr_tag = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
        self._parts.append(
            f'<w:p><w:r>{rpr_tag}'
            f'<w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r></w:p>'
        )
        return self

    def quote(self, text: str) -> "_DocBuilder":
        self._parts.append(
            f'<w:p><w:pPr><w:pStyle w:val="Quote"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r></w:p>'
        )
        return self

    def table_row(self, cells: List[str]) -> "_DocBuilder":
        tds = "".join(
            f'<w:tc><w:p><w:r>'
            f'<w:t xml:space="preserve">{xml_escape(c)}</w:t>'
            f'</w:r></w:p></w:tc>'
            for c in cells
        )
        self._parts.append(f"<w:tr>{tds}</w:tr>")
        return self

    def begin_table(self) -> "_DocBuilder":
        self._parts.append(
            '<w:tbl><w:tblPr>'
            '<w:tblStyle w:val="TableGrid"/>'
            '<w:tblW w:w="5000" w:type="pct"/>'
            '</w:tblPr>'
        )
        return self

    def end_table(self) -> "_DocBuilder":
        self._parts.append("</w:tbl>")
        return self

    def empty_para(self) -> "_DocBuilder":
        self._parts.append("<w:p/>")
        return self

    def build_body(self) -> str:
        return "".join(self._parts)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def _make_document_xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"'
        ' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )


def _write_docx(doc_xml: str, output_path: str) -> None:
    """Write the DOCX ZIP archive to *output_path*."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        zf.writestr("word/styles.xml", _STYLES)
    with open(output_path, "wb") as fh:
        fh.write(buf.getvalue())


def _validate_docx(path: str) -> None:
    """Assert the DOCX at *path* is a valid ZIP with required parts."""
    required = {
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "word/_rels/document.xml.rels",
        "word/styles.xml",
    }
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"DOCX {path!r} is not a valid ZIP: {exc}") from exc

    missing = required - names
    if missing:
        raise RuntimeError(
            f"DOCX {path!r} is missing required parts: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

_SEVERITY_LABEL = {"high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}


def _build_speaker_docx(
    speaker: str,
    segments: List[TranscriptSegment],
    all_flags: List[HallucinationFlag],
    time_range: TimeRange,
    source_file: str,
    backend: str,
    generated_at: datetime.datetime,
) -> str:
    """Return document XML for a single speaker's report."""
    speaker_flags = [f for f in all_flags if f.speaker in (speaker, "*")]
    total_speaking = sum(s.duration for s in segments)
    word_count = sum(s.word_count for s in segments)

    b = _DocBuilder()

    b.heading(f"Performance Report — {speaker}", 1)
    b.empty_para()

    # Metadata table
    b.begin_table()
    b.table_row(["Field", "Value"])
    b.table_row(["Generated", generated_at.strftime("%Y-%m-%d %H:%M:%S")])
    b.table_row(["Source", os.path.basename(source_file)])
    b.table_row(["Backend", backend])
    b.table_row(["Time range", str(time_range)])
    b.table_row(["Speaking time", fmt_duration(total_speaking)])
    b.table_row(["Word count", f"{word_count:,}"])
    b.table_row(["Segments", str(len(segments))])
    b.table_row(["Quality flags", str(len(speaker_flags))])
    b.end_table()
    b.empty_para()

    if speaker_flags:
        b.para("⚠ Quality Notice", bold=True)
        high = sum(1 for f in speaker_flags if f.severity == "high")
        med  = sum(1 for f in speaker_flags if f.severity == "medium")
        low  = sum(1 for f in speaker_flags if f.severity == "low")
        parts = []
        if high: parts.append(f"{high} HIGH")
        if med:  parts.append(f"{med} MEDIUM")
        if low:  parts.append(f"{low} LOW")
        b.para(f"This transcript has {len(speaker_flags)} flag(s): " + ", ".join(parts) + ".")
        b.empty_para()

    # Transcript
    b.heading("Transcript", 2)
    start_to_flags: dict = {}
    for f in speaker_flags:
        start_to_flags.setdefault(f.start, []).append(f)

    for seg in segments:
        seg_flags = start_to_flags.get(seg.start, [])
        flag_txt = " ".join(_SEVERITY_LABEL.get(f.severity, "") for f in seg_flags)
        label = fmt_sec(seg.start)
        if flag_txt:
            b.heading(f"{label}  {flag_txt}", 3)
        else:
            b.heading(label, 3)
        b.quote(seg.text)
        if seg.confidence < 0.80:
            b.para(f"(confidence: {seg.confidence:.0%})", italic=True)

    # Flags section
    if speaker_flags:
        b.empty_para()
        b.heading("Quality Flags", 2)
        for f in speaker_flags:
            label = _SEVERITY_LABEL.get(f.severity, f.severity.upper())
            b.para(
                f"{label} {fmt_sec(f.start)} — {f.flag_type}: {f.message}"
            )

    return _make_document_xml(b.build_body())


def _build_summary_docx(
    result: TranscriptResult,
    time_range: TimeRange,
    generated_at: datetime.datetime,
) -> str:
    """Return document XML for the overall summary."""
    entries = build_diarization(result.segments)
    b = _DocBuilder()

    b.heading("Meeting Report — Summary", 1)
    b.empty_para()

    b.begin_table()
    b.table_row(["Field", "Value"])
    b.table_row(["Generated", generated_at.strftime("%Y-%m-%d %H:%M:%S")])
    b.table_row(["Source", os.path.basename(result.source_file)])
    b.table_row(["Backend", result.backend])
    b.table_row(["Time range", str(time_range)])
    b.table_row(["Duration processed", fmt_duration(time_range.duration)])
    b.table_row(["Speakers", str(len(entries))])
    b.table_row(["Total flags", str(len(result.flags))])
    b.end_table()
    b.empty_para()

    b.heading("Speaker Breakdown", 2)
    b.begin_table()
    b.table_row(["Speaker", "Speaking time", "Words", "Segments", "Flags"])
    for e in entries:
        segs = result.segments_for_speaker(e.speaker)
        words = sum(s.word_count for s in segs)
        n_flags = sum(1 for f in result.flags if f.speaker == e.speaker)
        b.table_row([
            e.speaker,
            fmt_duration(e.total_speaking_duration),
            f"{words:,}",
            str(e.segment_count),
            str(n_flags),
        ])
    b.end_table()

    if result.flags:
        b.empty_para()
        b.heading("Quality Flags", 2)
        for f in result.flags:
            label = _SEVERITY_LABEL.get(f.severity, f.severity.upper())
            b.para(
                f"{label} {fmt_sec(f.start)} {f.speaker} — "
                f"{f.flag_type}: {f.message}"
            )

    return _make_document_xml(b.build_body())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_docx_reports(
    result: TranscriptResult,
    time_range: TimeRange,
    output_dir: str,
) -> List[str]:
    """Write per-speaker + summary DOCX reports to *output_dir*.

    Each file is validated after writing.

    Returns
    -------
    List[str]
        Paths of the files written.
    """
    ensure_dir(output_dir)
    now = datetime.datetime.now()
    written: List[str] = []

    for speaker in result.speakers():
        segs = result.segments_for_speaker(speaker)
        doc_xml = _build_speaker_docx(
            speaker=speaker,
            segments=segs,
            all_flags=result.flags,
            time_range=time_range,
            source_file=result.source_file,
            backend=result.backend,
            generated_at=now,
        )
        fname = f"{safe_filename(speaker)}.docx"
        path = os.path.join(output_dir, fname)
        _write_docx(doc_xml, path)
        _validate_docx(path)
        written.append(path)

    summary_xml = _build_summary_docx(result, time_range, now)
    summary_path = os.path.join(output_dir, "_summary.docx")
    _write_docx(summary_xml, summary_path)
    _validate_docx(summary_path)
    written.append(summary_path)

    return written
