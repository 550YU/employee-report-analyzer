# Employee Report Analyzer

English | [简体中文](README.md)

An open-source GitHub Copilot CLI Skill and local Python toolkit that turns MP3, WAV, or JSON transcripts into separate, evidence-grounded Markdown and DOCX reports for each speaker. It also guides Copilot in drafting traceable supervisor assessments that require human review.

> **Important boundary:** This project produces evidence packages and AI-assisted drafts. It must not automatically decide probation, promotion, performance ratings, or employment outcomes.

## Highlights

- **Real audio input:** local Faster-Whisper ASR with Windows CPU INT8, VAD, and `base`/`small` models.
- **Candidate speaker timeline:** when no range is known, perform anonymous diarization before asking a human to confirm identity and range.
- **Fault-contained chunking:** 30–60 second chunks, isolated workers, hard timeouts, fail-closed behavior, and optional fallback-model retry for suspect chunks.
- **Hallucination checks:** detects repeated text, known hallucination phrases, abnormal speech rate, and low confidence.
- **Per-speaker deliverables:** separate Markdown and DOCX reports plus summary files.
- **Evidence-grounded assessment:** distinguishes facts, interpretation, gaps, verification items, and recommended actions with timestamps.
- **Privacy-first:** processing is local by default; JSON workflows require no cloud credentials.

## Pipeline

```text
Audio/JSON → anonymous timeline → human confirms identity/range → chunked ASR
           → hallucination checks → per-speaker evidence → Copilot draft → human approval
```

## Quick start

### 1. Install

Python 3.9+ is required.

```powershell
# Core JSON/report functionality
pip install -e .

# Recommended real-audio setup
pip install -e ".[audio]"

# Development tests
pip install -e ".[dev]"
```

Optional higher-quality diarization:

```powershell
pip install -e ".[diarize-speechbrain]"
# Or pyannote, which requires Hugging Face model access
pip install -e ".[diarize-pyannote]"
```

### 2. Preview anonymous speaker ranges

```powershell
python -m zhrreport.cli diarize meeting.mp3 `
  --range 00:00:00-00:02:00 `
  --diarization-backend numpy_acoustic `
  --num-speakers 2
```

Labels such as `SPEAKER_00` are candidate clusters, not verified identities. A human must map them to people.

### 3. Generate an individual report from a confirmed range

```powershell
python -m zhrreport.cli report meeting.mp3 `
  --backend faster-whisper `
  --range 00:05:26-00:19:01 `
  --speaker-name KY `
  --model base `
  --chunk-duration 45 `
  --timeout 120 `
  --output-dir output\KY
```

Outputs include `KY.md`, `KY.docx`, `_summary.md`, and `_summary.docx`.

### 4. Use an existing JSON transcript

```powershell
python -m zhrreport.cli report sample\sample_transcript.json `
  --confirm-full --output-dir output\sample
```

### 5. Ask Copilot for a supervisor-assessment draft

Open the repository as a Copilot CLI workspace. `.github/skills/zhrreport/SKILL.md` supplies the evidence and assessment rules. Example prompt:

```text
Read each evidence report in output and generate a separate supervisor-assessment draft following the zhrreport Skill.
```

The framework covers an overall assessment, stage expectations, four evidence dimensions, technical case reasoning, professionalism, 30–60 day actions, and an evidence-quality appendix.

## Diarization backends

| Backend | Credential | Strength | Limitation |
|---|---|---|---|
| `numpy_acoustic` | None | Lightweight and local | Acoustic clustering, not a speaker identity model |
| `speechbrain` | None | ECAPA speaker embeddings | Large dependencies/model download |
| `pyannote` | HF token/model access | Usually highest quality | Gated model and heavier environment |

## Diagnostics

```powershell
python -m zhrreport.cli backends
python -m zhrreport.cli diarizers
python -m zhrreport.cli timeline sample\sample_transcript.json
python -m zhrreport.cli check-hallucination sample\sample_transcript.json
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Real-audio integration tests explicitly skip when their dependencies or test audio are unavailable.

## Copilot integration

- Skill: `.github/skills/zhrreport/SKILL.md`
- Extension: `.github/extensions/zhrreport/extension.mjs`
- CLI: `python -m zhrreport.cli`

The extension exposes `zhrreport_generate`, `zhrreport_timeline`, `zhrreport_check_hallucination`, and `zhrreport_backends`. Extension availability depends on the Copilot CLI version and runtime; the CLI remains independently usable.

## Security, privacy, and fairness

- Never commit real recordings, transcripts, employee reports, or customer data to a public repository.
- ASR can misrecognize names, numbers, commands, products, and root causes; verify against source material.
- Anonymous cluster labels do not establish identity.
- AI output is a draft only. Authorized humans must make formal employment decisions using complete information.
- See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
