"""Whitelist vector-free profile/evidence facts at every public JSON boundary."""

from math import isfinite
import re

ID = re.compile(r'[a-f0-9]{32}')


def scalar_facts(row, keys):
    """Copy bounded finite scalars only; nested model/cache payloads never pass through."""
    if not isinstance(row, dict):
        return {}
    result = {}
    for key in keys:
        value = row.get(key)
        if type(value) in (int, bool) or value is None:
            result[key] = value
        elif type(value) is float and isfinite(value):
            result[key] = value
        elif isinstance(value, str):
            result[key] = value[:256]
    return result


def string_list(value, limit=12):
    """Keep short reason codes without copying arbitrary nested records."""
    return [item[:160] for item in value[:limit] if isinstance(item, str)] if isinstance(value, list) else []


def bounded_rows(value, limit):
    """Reject malformed collections instead of iterating attacker-controlled mappings/strings."""
    return value[:limit] if isinstance(value, list) else []


def public_voice_profile(row):
    """Project one bounded public profile; private embeddings and unknown fields are omitted."""
    if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not ID.fullmatch(row['id']):
        return None
    if type(row.get('number')) is not int or not 1 <= row['number'] <= 64:
        return None
    result = scalar_facts(row, ('id', 'number', 'subjectId', 'sampleCount', 'status'))
    result['visualSubjects'] = [scalar_facts(item, ('id', 'scope', 'subjectId', 'visualScope', 'identityScope'))
                                for item in bounded_rows(row.get('visualSubjects'), 32) if isinstance(item, dict)]
    result['references'] = [scalar_facts(item, ('key', 'sourceId', 'windowId', 'startMs', 'endMs', 'rawSpeaker', 'explicitVideo'))
                            for item in bounded_rows(row.get('references'), 8) if isinstance(item, dict)]
    return result


def public_voice_evidence(row):
    """Project nested quality/video evidence with explicit allowlists, never subtract vector keys."""
    result = scalar_facts(row, ('profileId', 'basis', 'state', 'score', 'margin', 'learned',
                                'key', 'sourceId', 'windowId', 'startMs', 'endMs', 'rawSpeaker'))
    if not isinstance(row, dict):
        return result
    result['reasons'] = string_list(row.get('reasons'))
    quality = row.get('quality')
    if isinstance(quality, dict):
        result['quality'] = {**scalar_facts(quality, ('accepted', 'rms', 'clippedFraction')),
                             'reasons': string_list(quality.get('reasons'))}
    video = row.get('video')
    if isinstance(video, dict):
        result['video'] = scalar_facts(video, ('subjectId', 'basis', 'strength', 'visualScope', 'identityScope'))
        result['video']['evidence'] = [scalar_facts(item, ('activeFrames', 'phraseFrames', 'faceId', 'timeMs', 'source', 'basis',
                                                          'faceEvidence', 'changeReview', 'profileFace', 'identitySimilarity'))
                                       for item in bounded_rows(video.get('evidence'), 16) if isinstance(item, dict)]
    return result
