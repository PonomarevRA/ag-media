"""Reuse the graph's bounded source pipeline for long uploads instead of truncating audio."""

from pathlib import Path

from voice_paths import ensure_temporary_root, workspace_temp
from voice_studio.core.analysis_stages import ignore_progress
from voice_studio.core.storage_cache import use_cache
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_graph_ingest import analyze_source, probe_source
from voice_studio.graph.media_stage_cache import source_digest
from voice_studio.speech.media_transcript import MAX_MEDIA_SECONDS, transcribe_media
from voice_studio.speech.recognition import SMALL_MODEL, ready, resolve_whisper_language


def transcribe_uploaded_media(root, language, source, *, allow_empty=False, emit=ignore_progress,
                               publish=None, publish_mosaic=None, **settings):
    """Keep short/live behavior; long files use the same window engine and cache as MCP."""
    source = Path(source)
    if allow_empty:
        return transcribe_media(root, language, source, allow_empty=True, emit=emit, publish=publish, **settings)
    metadata = probe_source(source)
    if metadata['durationMs'] <= MAX_MEDIA_SECONDS * 1000:
        return transcribe_media(root, language, source, emit=emit, publish=publish,
                                publish_mosaic=publish_mosaic, **settings)
    if language is None:
        small_model = SMALL_MODEL if settings.get('quality') and ready(SMALL_MODEL) else None
        language = resolve_whisper_language(None, small_model, clip_languages=settings.get('clip_languages', False))
    digest = source_digest(source)
    key = 'bounded-v2/' + digest
    ensure_temporary_root()
    group = workspace_temp(root) / 'analysis-checkpoints'
    directory = group / key
    with use_cache(group, directory):
        report = analyze_source(root, source, metadata['kind'], language,
                                visual=settings.get('visual', False), scene=settings.get('scene', False),
                                speaker_count=settings.get('speaker_count'), speakers=settings.get('speakers', False),
                                object_sampling=settings.get('object_sampling', 'small-detail'),
                                audio_classifier=settings.get('audio_classifier', 'auto'),
                                people_count=settings.get('people_count'), quality=settings.get('quality', False),
                                merge_similar=settings.get('merge_similar', False), body=settings.get('body', False),
                                gaze=settings.get('gaze', False), revolver=AnalysisRevolver(directory),
                                selected_stages=settings.get('selected_stages'), force_stages=settings.get('force_stages'),
                                object_reconciliation=settings.get('object_reconciliation', False),
                                publish=publish, emit=emit)
    report['sourceDigest'] = digest
    return report
