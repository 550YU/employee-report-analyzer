"""Deterministic fixture transcriber used for testing and samples.

This backend does **not** call any real ASR model.  It returns a
hard-coded set of transcript segments that simulate a realistic
30-minute internal review meeting between two speakers.

The fixture can be used:

* In automated tests (no credentials, no network, no audio file needed).
* When running the sample end-to-end pipeline (``sample/run_sample.py``).
* As a reference for what the JSON transcript format should look like.

Fixtures are clearly labelled ``backend: "fixture"`` in the output so
that callers can distinguish them from real transcriptions.
"""
from __future__ import annotations

from typing import List, Optional

from zhrreport.models import TimeRange, TranscriptSegment
from zhrreport.transcribers.base import TranscriberBase


# ---------------------------------------------------------------------------
# Fixture data – a synthetic 30-minute meeting (Alice & Bob)
# ---------------------------------------------------------------------------

_FIXTURE_SEGMENTS_RAW = [
    (0.0,    5.4,  "KY", "Good morning everyone, let's get started with today's quarterly review.", 0.97),
    (6.0,   14.2,  "TF",   "Morning KY. I've prepared the metrics dashboard — we can walk through it section by section.", 0.95),
    (15.0,  28.7,  "KY", "Perfect. Starting with Q3 revenue: we hit 94% of target, which is a solid result given the market headwinds we discussed last month.", 0.93),
    (30.0,  45.1,  "TF",   "Agreed. The slight shortfall came from two enterprise deals that slipped to Q4. Both are now closed, so we expect Q4 to make up the delta.", 0.96),
    (46.0,  59.3,  "KY", "Good. On the product side, the new onboarding flow went live in week 11. Activation rates improved by 18% in the first two weeks.", 0.94),
    (61.0,  75.8,  "TF",   "That was a team effort — the UX redesign and the backend optimisation happened in parallel, which was risky but paid off.", 0.92),
    (77.0,  92.5,  "KY", "Let's talk about the support queue. We're still running at 48-hour median resolution. The target is 24 hours. What's blocking that?", 0.95),
    (94.0, 118.0,  "TF",   "Two things: first, tier-1 agents don't have access to the new diagnostic tool yet — IT is rolling that out this week. Second, we have three open headcount positions. Until they're filled, throughput stays capped.", 0.91),
    (120.0, 138.5, "KY", "Understood. Let's flag the headcount issue to HR today. I'll also sync with IT to confirm the diagnostic tool rollout date.", 0.96),
    (140.0, 155.9, "TF",   "On engineering velocity: sprint completion rate is at 87%. Two stories were blocked by a third-party API regression that was outside our control.", 0.93),
    (157.0, 172.3, "KY", "That's acceptable. What's the remediation plan for the API dependency?", 0.97),
    (174.0, 195.6, "TF",   "We're adding a circuit-breaker and a local cache for the affected endpoint. Estimated completion by end of next sprint — so two weeks from today.", 0.94),
    (197.0, 215.0, "KY", "Good mitigation. Shifting to people topics — the engagement survey closed last Friday. Participation was 91%, which is our highest ever.", 0.95),
    (217.0, 240.5, "TF",   "The top three themes were: flexible working arrangements, clearer career progression paths, and more cross-team collaboration opportunities. We'll present the detailed results at next month's all-hands.", 0.92),
    (242.0, 260.7, "KY", "Before all-hands, I'd like each team lead to draft a one-page action plan responding to the themes relevant to their area. Can you own the engineering response?", 0.96),
    (262.0, 275.0, "TF",   "Absolutely. I'll have a draft by end of next week.", 0.97),
    (277.0, 295.8, "KY", "Great. Let me summarise the action items: one, HR to be notified about open headcount; two, IT sync on diagnostic rollout; three, circuit-breaker implementation in two weeks; four, TF to draft engineering engagement response.", 0.95),
    (297.0, 320.4, "TF",   "I'll log all of those in Jira and send a follow-up email today. Anything else before we close?", 0.96),
    (322.0, 345.0, "KY", "Just one more thing — we've been invited to present at the industry conference in November. I think it's a good visibility opportunity. TF, would you be willing to co-present on the onboarding flow case study?", 0.94),
    (347.0, 370.2, "TF",   "I'd love to. That's a great story to share externally. Happy to prepare the slide deck. Can we block two hours next week to align on messaging?", 0.95),
    (372.0, 390.5, "KY", "Done. I'll send a calendar invite right after this call. Alright, I think we're good. Thanks everyone for a productive session.", 0.97),
    (392.0, 405.8, "TF",   "Thanks KY. Talk soon.", 0.98),
    # Second half – 10-minute deeper technical discussion starting at 600s
    (600.0,  620.3, "KY", "One more topic I want to circle back on: infrastructure costs. We're over budget by 12% this quarter.", 0.93),
    (622.0,  645.7, "TF",   "Yes, the main drivers are the data pipeline scaling event in August and the increased storage for the new audit logging requirements.", 0.91),
    (647.0,  670.2, "KY", "Are there short-term optimisation opportunities without impacting reliability?", 0.96),
    (672.0,  700.5, "TF",   "Two quick wins: rightsizing the dev cluster saves roughly 15% of compute costs, and enabling S3 intelligent tiering on the archive bucket cuts storage by about 30%. Combined that erases maybe 8% of the overage.", 0.94),
    (702.0,  720.0, "KY", "Let's execute both next sprint. Add them to the backlog with a P1 label.", 0.97),
    (722.0,  740.8, "TF",   "Will do. For the remaining 4% gap, I'll model two scenarios and bring options to the next steering committee.", 0.95),
    (742.0,  760.0, "KY", "Perfect. Alright, now we're truly done. Have a great rest of the day.", 0.98),
    (762.0,  775.5, "TF",   "You too. Bye!", 0.99),
    # Simulate a segment with suspiciously high word rate (hallucination test data)
    (1800.0, 1800.5, "KY", "This sentence was generated much too quickly to be real speech and should be flagged.", 0.72),
    # Simulate a repeated segment (hallucination test data)
    (1900.0, 1910.0, "TF",   "Thank you for watching.", 0.65),
    (1910.0, 1920.0, "TF",   "Thank you for watching.", 0.64),
]


class FixtureTranscriber(TranscriberBase):
    """Deterministic backend that returns pre-canned segments.

    Suitable only for tests and demos.  Not connected to any ASR model.
    """

    @property
    def backend_name(self) -> str:
        return "fixture"

    def describe(self) -> str:
        return "fixture backend – deterministic, for testing only (NOT production ASR)"

    def transcribe_chunk(
        self,
        source: str,
        time_range: Optional[TimeRange] = None,
        chunk_id: int = 0,
    ) -> List[TranscriptSegment]:
        segments = [
            TranscriptSegment(
                start=row[0],
                end=row[1],
                speaker=row[2],
                text=row[3],
                confidence=row[4],
                chunk_id=chunk_id,
            )
            for row in _FIXTURE_SEGMENTS_RAW
        ]
        if time_range is not None:
            segments = [
                s for s in segments if time_range.overlaps(s.start, s.end)
            ]
        return segments
