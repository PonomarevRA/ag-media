---
name: ag-media-dialogue-audit
description: Find speaker changes, overlapping voices, uncertain assignments and missing speech in a Studio transcription while preserving the original ASR and source time.
---

# Audit dialogue

For repeat requests, use `agent-memory` and check the project’s local `.ag-media-memory/` first to check the source and current report revision. Reuse matching evidence or recheck only the intervals affected by changes.

1. Find the absolute original media path and its matching analysis report. Use a saved speech-ready or complete report containing `speechTurns`. When reusing an older report, supply its measured turns with `turns_path` or rerun the required speech stages.
2. Discover `dialogue_audit` with the live MCP catalog. Start with `rerun=false` and poll `dialogue_audit_job` in the same session. Re-recognize only when requested or needed for a definite missing phrase, with bounded window, audio and time options.
3. Review highest-priority candidates against the original sound and any direct visible activity. Clear video evidence can establish who was speaking at a time. Where it is ambiguous, use acoustic evidence and leave unresolved assignments explicit. A candidate alone never identifies a person.
4. Deliver the candidate interval, current speaker label, evidence, proposed correction and source reference. Changes to editorial text or speaker labels require an applicable edit request. Keep ASR proposals and original evidence available for review.

For offline auditing and supported input shapes see `docs/DIALOGUE_AUDIT.md` in the repository.
