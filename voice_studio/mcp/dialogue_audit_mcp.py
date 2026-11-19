"""Expose independently queued dialogue checks and vector-free voice profiles to agents."""

from pathlib import Path
from voice_studio.core.mcp_lanes import AsyncWorkLane

from voice_studio.app.dialogue_audit_service import audit_media, read_report
from voice_studio.mcp.media_graph_mcp import GraphBuildQueue
from voice_studio.speech.voice_profile_public import public_voice_profile, public_voice_evidence
import re


def register_dialogue_tools(mcp, studio_root):
    """Register independent audit/profile work and return resources for session cleanup."""
    queue = GraphBuildQueue(result_key='audit', label='проверки диалога')
    read_lane = AsyncWorkLane('voice-profile-read', capacity=2)

    @mcp.tool()
    def dialogue_audit(file_path: str, analysis_path: str, turns_path: str | None = None,
                             rerun: bool = False, max_windows: int = 6,
                             max_audio_ms: int = 30000, timeout_seconds: int = 180) -> dict:
        """Queue a separate source-bound audit of speaker boundaries, mixed phrases and missing text.

        analysis_path is a saved complete or speech-ready report with speechTurns. For an older
        report supply a measured turns_path JSON, or repeat only speech analysis first.
        rerun=True re-recognizes only bounded clean candidate clips, preserving raw ASR.
        Audio profiles, raw labels and evidence are not speaker-identity probabilities.
        Poll dialogue_audit_job; video and UI tools remain independently callable.
        """
        if not 0 <= max_windows <= 100 or not 0 <= max_audio_ms <= 600000 or not 1 <= timeout_seconds <= 3600:
            raise ValueError('Проверь бюджет проверки: 0–100 окон, до 600 секунд аудио и до 3600 секунд работы.')
        return queue.submit(audit_media_job, studio_root, file_path, analysis_path, turns_path,
                            rerun, max_windows, max_audio_ms, timeout_seconds)

    @mcp.tool()
    def dialogue_audit_job(job_id: str) -> dict:
        """Read the separate dialogue audit job, artifact path and review-required ASR suggestions."""
        return queue.status(job_id)

    @mcp.tool()
    @read_lane.offload
    def speaker_profiles(analysis_path: str, profile_id: str | None = None) -> dict:
        """Read bounded vector-free digital voice profiles and their source-clock cue evidence.

        Explicit speaking-video takes priority when available; weak video falls back to audio.
        Status is provisional. An abstention is not another person. No private embeddings return.
        """
        report = read_report(analysis_path)
        layer = report.get('voiceProfiles')
        layer = layer if isinstance(layer, dict) else {}
        profiles = layer.get('profiles')
        profiles = profiles if isinstance(profiles, list) else []
        profiles = [clean for row in profiles[:64] if (clean := public_voice_profile(row)) is not None
                    and (profile_id is None or clean['id'] == profile_id)]
        ids = {row['id'] for row in profiles}
        cues = []
        matched_count = 0
        for row in report['segments']:
            if not isinstance(row, dict) or not isinstance(row.get('voiceProfileId'), str) or row['voiceProfileId'] not in ids:
                continue
            start, end = row.get('startMs'), row.get('endMs')
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= 7200000:
                continue
            matched_count += 1
            if len(cues) < 500:
                raw = row.get('speaker')
                cues.append({'startMs': start, 'endMs': end,
                             'rawSpeaker': raw if type(raw) is int else None, 'voiceProfileId': row['voiceProfileId'],
                             'evidence': public_voice_evidence(row.get('voiceProfileEvidence'))})
        digest = report.get('sourceDigest')
        digest = digest if isinstance(digest, str) and re.fullmatch(r'[a-f0-9]{64}', digest) else None
        version = layer.get('version')
        version = version[:80] if isinstance(version, str) else None
        return {'version': version, 'sourceDigest': digest,
                'profiles': profiles, 'cues': cues, 'truncated': matched_count > 500,
                'meaning': 'provisional-model-evidence; not identity probability'}

    return queue, read_lane


def audit_media_job(root, file_path, analysis_path, turns_path, rerun, max_windows, max_audio_ms, timeout_seconds):
    """Translate positional queue parameters into the service's explicit bounded options."""
    return audit_media(Path(root), file_path, analysis_path, turns_path=turns_path, rerun=rerun,
                       max_windows=max_windows, max_audio_ms=max_audio_ms, timeout_seconds=timeout_seconds)
