---
name: zhrreport
version: "1.2.0"
description: |
  Generate evidence-based per-speaker Markdown and DOCX performance reports
  from MP3, WAV, or JSON transcripts. Supports local Faster-Whisper ASR,
  anonymous local diarization, range confirmation, hard worker timeouts, and
  hallucination detection. 从音频或转写生成逐人证据报告与需人工审核的主管评价草稿。
tags: [transcript, audio, report, speech, performance-review, diarization]
entrypoint: "python -m zhrreport.cli"
extension: ".github/extensions/zhrreport/extension.mjs"
tools:
  - zhrreport_generate
  - zhrreport_timeline
  - zhrreport_check_hallucination
  - zhrreport_backends
---

# zhrreport

Use this Skill when a user uploads MP3/WAV meeting audio or a JSON transcript
and asks for speaker evidence reports or draft supervisor comments.

## Audio workflow

1. If the user has not supplied a range, call `zhrreport_timeline` first. It
   performs token-free local diarization and returns anonymous `SPEAKER_00`
   candidate ranges. Ask the user to confirm a range and identity before ASR.
2. Call `zhrreport_generate` with `backend: "faster-whisper"`, the confirmed
   range, and `speaker_name`. Use CPU INT8 model `base` by default; use `small`
   as an optional bad-chunk fallback. Workers use 30-60 second chunks and a
   hard per-chunk timeout.

The CLI also infers `json` for `.json` and `faster-whisper` for audio when
`--backend` is omitted.
3. Read the generated per-speaker `.md` evidence package and `_summary.md`.
   The script does not make an authoritative performance judgment.
4. Draft a management-facing supervisor assessment plus an evidence appendix.
   Ground every substantive claim in timestamped evidence, clearly label the
   assessment as a Copilot-generated draft, and require human review.

```powershell
pip install -e ".[audio]"
python -m zhrreport.cli diarize meeting.mp3 --range 00:18:30-00:20:30
python -m zhrreport.cli report meeting.mp3 --backend faster-whisper `
  --range 00:05:26-00:07:26 --speaker-name KY --model base `
  --chunk-duration 45 --timeout 120 --output-dir output\KY
```

No range on an interactive audio `report` command triggers full-recording
local diarization and waits for range confirmation before transcription. In a
non-interactive tool call, provide a range or set `confirm_full: true`.

## Diarization tiers

| Backend | Credential | Install | Limitation |
|---|---|---|---|
| `numpy_acoustic` | None | `pip install numpy imageio-ffmpeg` | Approximate acoustic clusters; not speaker embeddings |
| `speechbrain` | None | `pip install speechbrain torch scikit-learn imageio-ffmpeg` | Better ECAPA embeddings; large local dependencies/model |
| `pyannote` | HF token/model access | `pip install pyannote.audio imageio-ffmpeg` | Highest quality optional path |

Never present `numpy_acoustic` labels as verified identities. Ask a human to
map anonymous labels/ranges to names.

## Supervisor assessment synthesis

Generate two layers. Keep the management assessment concise and readable;
move ASR diagnostics and detailed verification items to the evidence appendix.

### Layer 1: management assessment

Use this order for every person:

1. **Overall assessment (总体评价)**: state the strongest demonstrated
   capability, the most important gap, and a provisional stage conclusion.
   Use wording such as "建议结论：基本达到第 1 月阶段预期（待主管确认）".
   Never present an AI conclusion as a final rating, promotion, probation, or
   employment decision.
2. **Stage expectations (阶段标准)**: adapt expectations to the person's
   stated report period. For month 1, prioritize learning-to-practice transfer,
   safe execution under guidance, escalation boundaries, and accurate records.
   For month 2, additionally examine independent closure, customer updates,
   reusable knowledge, and visible team contribution. If the role framework is
   unavailable, label these as assessment assumptions requiring supervisor
   confirmation.
3. **Work output and learning (工作成果与学习)**.
4. **Communication and presentation (表达与汇报)**.
5. **Problem solving and reflection (问题解决与自我反思)**.
6. **Growth plan and ownership (成长计划与主人翁意识)**.
7. **Technical case assessment (案例与技术能力)**: distinguish symptom,
   impact, diagnostic hypotheses, evidence, eliminated hypotheses, mitigation,
   root cause, final fix, validation, and reusable rule. Do not call mitigation
   a root-cause fix.
8. **Professionalism and potential (职业化素养与发展潜力)**: assess only
   evidenced integrity, feedback response, customer awareness, collaboration,
   documentation quality, and knowledge sharing. Do not infer personality.
9. **30-60 day actions (未来 30–60 天行动要求)**: give 2-3 actions with a
   deadline, deliverable, acceptance criteria, and verification source. Adapt
   actions to the person's stage and observed gaps; do not reuse generic goals
   mechanically.
10. **Provisional conclusion (阶段性结论)**: summarize demonstrated readiness,
    unresolved evidence gaps, and what the supervisor must verify before making
    a formal decision.

Within dimensions 3-6, use the same four-part reasoning chain:

- **事实依据**: timestamped facts or faithful paraphrases.
- **正向评价**: what the evidence reasonably demonstrates.
- **差距与风险**: what remains missing, weak, or unverified.
- **维度结论**: a concise, stage-aware conclusion.

### Layer 2: evidence and quality appendix

Include the processed range, segment and hallucination/LOW counts, excluded or
unusable ranges, all quantities and technical names requiring source-system
verification, and a human-review checklist. The appendix must not dominate the
opening management assessment.

### Four evidence dimensions

For each person, write one concise section per dimension:

1. **Work Output & Learning (工作成果与学习)**: deliverables, metrics, completed
   work, acquired skills, and application of learning.
2. **Communication & Presentation (表达与汇报)**: structure, clarity, concrete
   examples, response quality, and professional expression.
3. **Problem Solving & Reflection (问题解决与自我反思)**: root causes, solutions,
   blockers, lessons, and specific self-reflection.
4. **Growth Plan & Ownership (成长计划与主人翁意识)**: concrete next steps,
   timelines, initiative, collaboration, risk ownership, and team alignment.

Every substantive claim, including strengths and gaps, must cite a timestamped
phrase or faithful paraphrase. A recommendation may cite the gap that motivates
it instead of inventing future evidence. Treat hallucination-flagged or
very-low-confidence ASR as unreliable. If evidence is missing, write "该片段证据
不足，无法评估" rather than inferring. Scope comments only to the processed
range. Clearly separate direct facts, assessment interpretation, and items that
still require verification.

Do not use numeric scoring unless the user supplies an approved rubric. A
provisional label is allowed only when its assumptions and verification needs
are explicit. Quantities, SLA values, customer ratings, certifications, project
names, commands, and root causes from low-confidence ASR must be marked for
source verification.

Use this heading and disclosure:

```markdown
## Supervisor Comments — [Speaker]
> Draft synthesized by Copilot from transcript evidence. Human review and
> approval are required before management use.
```

## JSON and diagnostics

```powershell
python -m zhrreport.cli report meeting.json --confirm-full
python -m zhrreport.cli timeline meeting.json
python -m zhrreport.cli check-hallucination meeting.json
python -m zhrreport.cli backends
python -m zhrreport.cli diarizers
```

The JSON/fixture paths require no cloud credentials. Reports are separate MD
and DOCX files per speaker plus summary files. Chunk failures and timeouts fail
closed so incomplete transcripts are not emitted as successful reports.
