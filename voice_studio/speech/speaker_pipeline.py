"""Connect source-scoped voice profiles and non-destructive dialogue evidence to stages."""

from collections import defaultdict
from copy import deepcopy
import logging
import subprocess

from voice_studio.core.analysis_stages import ignore_progress, stage
from voice_studio.core.media_resources import run_batch_stage
from voice_studio.mods.errors import ModError
from voice_studio.speech.speaker_profiles import collect_profiles, cue_number

LOGGER = logging.getLogger(__name__)
PROFILE_PIPELINE_VERSION = 'source-profile-integration-4'


def voice_profile_failure(error):
    """Describe the recoverable failure without exposing local paths or model output."""
    if isinstance(error, subprocess.TimeoutExpired):
        return 'Голосовая модель не ответила вовремя'
    if isinstance(error, FileNotFoundError):
        return 'Файл голосовой модели или аудио недоступен'
    if isinstance(error, ImportError):
        return 'Среда голосовой модели недоступна'
    if isinstance(error, subprocess.SubprocessError):
        return 'Воркер голосовой модели не завершил запрос'
    return 'Не удалось обработать голосовые образцы'


def _mapping(value):
    """Treat malformed optional legacy objects as absent evidence."""
    return value if isinstance(value, dict) else {}


def _rows(value):
    """Read object arrays without accidentally iterating strings or mapping keys."""
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _integer(value):
    """Accept an integer namespace/time coordinate, excluding booleans and containers."""
    return type(value) is int


def _voices(report):
    """Keep only unique valid source-profile IDs and positive source-local numbers."""
    from voice_studio.speech.voice_profile_public import public_voice_profile

    candidates = []
    ids, numbers = defaultdict(int), defaultdict(int)
    for row in _rows(_mapping(report.get('voiceProfiles')).get('profiles')):
        voice = public_voice_profile(row)
        if not voice or not _integer(voice.get('number')) or voice['number'] < 1:
            continue
        candidates.append(voice)
        ids[voice['id']] += 1
        numbers[voice['number']] += 1
    return [voice for voice in candidates if ids[voice['id']] == 1 and numbers[voice['number']] == 1]


def _source_number(cue, numbers):
    """Require a real voice ID and matching source number; raw integers prove nothing."""
    identity = cue.get('voiceProfileId')
    number = numbers.get(identity) if isinstance(identity, str) else None
    if number is not None and _integer(cue.get('profileSpeakerNumber')) and cue['profileSpeakerNumber'] == number:
        return number
    return None



def _voice_input_cues(value):
    """Exclude this stage's additive outputs from its own warm-cache dependency key."""
    if not isinstance(value, list):
        return []
    result = deepcopy(value)
    for cue in result:
        if not isinstance(cue, dict):
            continue
        if 'rawLinkedSpeakerNumber' in cue:
            cue['linkedSpeakerNumber'] = cue['rawLinkedSpeakerNumber']
        if cue.get('rawSpeaker') == cue.get('speaker'):
            cue.pop('rawSpeaker', None)
        for field in ('voiceProfileId', 'voiceProfileEvidence', 'profileSpeakerNumber', 'rawLinkedSpeakerNumber'):
            cue.pop(field, None)
    return result

def _overlaps(value):
    """Normalize independent overlap intervals before both cache lookup and profiling."""
    output = []
    for row in _rows(value):
        left, right = row.get('startMs'), row.get('endMs')
        if not (_integer(left) and _integer(right) and 0 <= left < right):
            continue
        speakers = row.get('speakers')
        numbers = [number for number in speakers if _integer(number)] if isinstance(speakers, list) else []
        output.append({'startMs': left, 'endMs': right, 'speakers': numbers})
    return output



def _invalid_overlaps(value):
    """Unknown negative intervals prevent learning instead of silently becoming no overlap."""
    if value is None:
        return False
    if not isinstance(value, list):
        return True
    for row in value:
        if not isinstance(row, dict):
            return True
        left, right = row.get('startMs'), row.get('endMs')
        if not (_integer(left) and _integer(right) and 0 <= left < right):
            return True
    return False

def attach_voice_profiles(report, wav, audio_start_ms, source_id, *, registry=None,
                          cache_path=None, window_id=None, visual_scope=None, emit=ignore_progress):
    """Attach profile fields to valid cues without replacing raw or malformed legacy records."""
    from voice_studio.speech.voice_profiles import apply_voice_profiles

    originals, clean = [], []
    for cue in _rows(report.get('segments')):
        left, right = cue.get('startMs'), cue.get('endMs')
        if not (_integer(left) and _integer(right) and 0 <= left < right):
            continue
        item = deepcopy(cue)
        window = _mapping(item.get('speakerAudioWindow'))
        first, last = window.get('startMs'), window.get('endMs')
        if not (_integer(first) and _integer(last) and left <= first < last <= right):
            item.pop('speakerAudioWindow', None)
        item['speakerVideoEvidence'] = _mapping(item.get('speakerVideoEvidence'))
        originals.append(cue)
        clean.append(item)
    invalid_overlaps = _invalid_overlaps(report.get('speechOverlaps'))
    if invalid_overlaps:
        for item in clean:
            item['overlap'] = True
    layer = {'segments': clean, 'videoEvidence': {**profile_visual_evidence(report.get('videoEvidence')),
                                                    'localVoiceAnchors': _mapping(report.get('_localVoiceAnchors'))},
             'speechOverlaps': _overlaps(report.get('speechOverlaps'))}
    with stage(emit, 'voice-profiles') as progress:
        try:
            public = run_batch_stage(apply_voice_profiles, layer, wav, audio_start_ms,
                                     source_id=source_id, registry=registry, cache_path=cache_path,
                                     window_id=window_id, visual_scope=visual_scope)
            if not isinstance(public, dict):
                raise ValueError('Invalid voice profile stage result')
        except (BrokenPipeError, ConnectionResetError):
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, ModError, subprocess.SubprocessError) as error:
            LOGGER.warning('Source voice profiling unavailable: %s', type(error).__name__)
            reason = voice_profile_failure(error)
            report['voiceProfileWarning'] = reason + '. Исходные метки сохранены.'
            progress['state'] = 'failed'
            progress['reason'] = reason
            return None
    report.pop('voiceProfileWarning', None)
    report['voiceProfiles'] = public
    numbers = {voice['id']: voice['number'] for voice in _voices(report)}
    for original, cue in zip(originals, clean):
        original.pop('voiceProfileId', None)
        original.pop('profileSpeakerNumber', None)
        original['voiceProfileEvidence'] = cue.get('voiceProfileEvidence') or {'state': 'abstained'}
        if invalid_overlaps:
            original['voiceProfileEvidence'].setdefault('reasons', []).append('invalid-overlap-evidence')
        identity = cue.get('voiceProfileId')
        number = numbers.get(identity) if isinstance(identity, str) else None
        if number is not None:
            original.setdefault('rawSpeaker', original.get('speaker'))
            original['voiceProfileId'] = identity
            original['profileSpeakerNumber'] = number
    return public


def measure_speaker_review(report, prepared_audio, offset, review, *, alternate, emit):
    """Retry on an isolated copy of editable evidence and preserve the raw visual output."""
    evidence = report["videoEvidence"]
    phrases = deepcopy(evidence.get("phrases", {}))
    working = {**report, "segments": deepcopy(report["segments"]),
               "videoEvidence": {**evidence, "phrases": phrases}}
    with stage(emit, "speaker-review") as progress:
        try:
            run_batch_stage(review, working, prepared_audio(), offset, alternate=alternate)
        except (BrokenPipeError, ConnectionResetError):
            raise
        except (OSError, ValueError, ImportError, ModError, subprocess.SubprocessError) as error:
            LOGGER.exception("Speaker review failed; visual checkpoint retained")
            reason = voice_profile_failure(error) + ". Видеоразметка сохранена."
            progress.update(state="failed", reason=reason)
            return {"speakerReview": {"status": "failed", "reason": reason}}
        details = working["speakerReview"]
        if details.get("status") in {"skipped", "partial"}:
            progress.update(state=details["status"], reason=details["reason"])
    return {"segments": working["segments"], "phrases": phrases, "speakerReview": details}


def usable_speaker_review(result):
    """Only complete checks and evidence-based skips may be replayed from cache."""
    return isinstance(result.get("speakerReview"), dict) and result["speakerReview"].get("status") not in {"failed", "partial"}


def project_voice_index(report):
    """Join appearance only through source voice IDs, never coincident raw cluster numbers."""
    if not isinstance(report.get('voiceProfiles'), dict):
        return
    voices = _voices(report)
    remap_voice_faces(report)
    appearances = {}
    try:
        for row in collect_profiles(report):
            identity = row.get('voiceProfileId')
            if isinstance(identity, str):
                appearances[identity] = row
    except (TypeError, ValueError, KeyError, AttributeError):
        # Legacy appearance data is optional; invalid face metadata cannot invent identity.
        LOGGER.warning('Legacy speaker appearance could not be projected')
    previous = {}
    for row in _rows(report.get('speakerProfileIndex')):
        nested = _mapping(row.get('voice'))
        identity = row.get('voiceProfileId') or nested.get('id')
        if not identity and row.get('scope') == 'source':
            identity = row.get('id')
        if isinstance(identity, str):
            previous[identity] = row
    current = []
    for voice in voices:
        identity, number = voice['id'], voice['number']
        appearance = appearances.get(identity) or previous.get(identity) or {}
        row = deepcopy(appearance) if appearance.get('number') == number else {}
        row.update(id=identity, voiceProfileId=identity, number=number, scope='source', voice=deepcopy(voice),
                   anchorsMs=[ref['startMs'] for ref in voice['references'] if _integer(ref.get('startMs'))][:12])
        row.setdefault('faceId', None)
        row.setdefault('faceBasis', None)
        row.setdefault('pinned', False)
        current.append(row)
    report['speakerProfileIndex'] = current
    report['speakerScope'] = 'source-profiles-with-raw-window-labels'
    displayed = set()
    for cue in _rows(report.get('segments')):
        number = cue_number(cue)
        if _integer(number) and not cue.get('overlap'):
            displayed.add(number)
    report['speakerCount'] = len(displayed)


def attach_dialogue_audit(report, source_id, *, emit=ignore_progress):
    """Locate structural dialogue problems using retained native turns, without rerunning ASR."""
    from voice_studio.speech.native_turn_bounds import audit_native_dialogue

    if not report.get('speechTurns'):
        report['dialogueAudit'] = {'status': 'unavailable', 'reason': 'native-voice-turns-unavailable'}
        return
    with stage(emit, 'dialogue-audit') as progress:
        try:
            inventory = audit_native_dialogue(report, report['speechTurns'], source={
                'sha256': source_id, 'durationMs': report['durationMs'], 'clock': 'original-media-ms'},
                visual_facts=_mapping(report.get('sceneEvidence')).get('uiActivity'))
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            LOGGER.warning('Dialogue inventory unavailable: %s', type(error).__name__)
            inventory = {'status': 'unavailable', 'reason': 'invalid-or-unsupported-evidence'}
            progress.update(state='failed', reason='Проверка диалога не смогла прочитать данные о репликах и голосовых интервалах.')
        report['dialogueAudit'] = inventory


def enrich_source_speech(report, source, prepared_audio, audio_start_ms, cache, *, formats=None, emit=ignore_progress):
    """Key voice stage and private registry by identical source-clock positive/negative evidence."""
    from voice_studio.graph.media_analysis_revolver import result_fingerprint
    from voice_studio.graph.media_stage_cache import source_digest
    from voice_studio.mod_runtime import layer_fingerprint
    from voice_studio.speech.voice_profiles import VERSION

    identity = source_digest(source)
    report['sourceDigest'] = identity
    evidence = profile_visual_evidence(report.get('videoEvidence'))
    visual = _mapping(report.get('videoEvidence'))
    overlaps = _overlaps(report.get('speechOverlaps'))
    anchors = {}
    if formats and not _invalid_overlaps(report.get('speechOverlaps')) and evidence['localAnchorsEligible']:
        from voice_studio.speech.local_voice_anchors import local_anchors, review_pairs

        candidates = local_anchors({**evidence, 'visibleSpeechIntervals': _rows(visual.get('visibleSpeechIntervals'))},
                                   report.get('speechTurns') or [])
        from voice_studio.speech.voice_profile_evidence import overlaps_audio

        anchors = {str(cue['startMs']): candidates[str(cue['startMs'])] for cue in _rows(report.get('segments'))
                   if _integer(cue.get('startMs')) and _integer(cue.get('endMs'))
                   and str(cue['startMs']) in candidates
                   and 'speakerNumber' not in cue and not overlaps_audio(cue, {'speechOverlaps': overlaps})
                   and not _mapping(cue.get('speakerVideoEvidence'))}
        pairs = review_pairs(anchors, visual) if anchors else []
    else:
        pairs = []
    upstream = {'sourceId': identity, 'audioStartMs': audio_start_ms,
                'segments': _voice_input_cues(report.get('segments')), 'turns': report.get('speechTurns', []),
                'videoProofs': evidence, 'localAnchors': anchors, 'nativeReviewPairs': pairs, 'overlaps': overlaps,
                'overlapEvidenceValid': not _invalid_overlaps(report.get('speechOverlaps'))}
    registry_key = result_fingerprint([VERSION, PROFILE_PIPELINE_VERSION, getattr(cache, 'revision', None),
                                       layer_fingerprint('voice'), upstream])

    def compute():
        """Avoid reusing references learned before changed overlap or static/frozen exclusions."""
        resolved = deepcopy(anchors)
        if pairs:
            from voice_studio.speech.local_voice_anchors import reconcile_anchors
            from voice_studio.vision.speaker_face_recheck import run_native_review

            reconcile_anchors(resolved, pairs, run_native_review(source, formats, pairs))
        layer = {'segments': deepcopy(report.get('segments', [])), 'videoEvidence': evidence,
                 '_localVoiceAnchors': resolved,
                 'speechOverlaps': deepcopy(report.get('speechOverlaps'))}
        directory = cache.revolver.directory
        path = directory / 'voice-registry' / VERSION / registry_key / 'registry.json' if directory else None
        public = attach_voice_profiles(layer, prepared_audio(), audio_start_ms, identity,
                                       cache_path=path, window_id=0, visual_scope=identity + ':window:0', emit=emit)
        result = {'segments': layer['segments']}
        if public is not None:
            result['voiceProfiles'] = public
        if layer.get('voiceProfileWarning'):
            result['voiceProfileWarning'] = layer['voiceProfileWarning']
        return result

    layer = cache.run('voice', {'version': VERSION, 'pipeline': PROFILE_PIPELINE_VERSION}, upstream, compute,
                      lambda value: isinstance(_mapping(value).get('voiceProfiles'), dict), emit=emit)
    report.update(_mapping(layer))
    project_voice_index(report)
    attach_dialogue_audit(report, identity, emit=emit)


def profile_visual_evidence(evidence, *, offset=0, id_base=0):
    """Normalize talker proof and static veto together on the same source/face clock."""
    evidence = _mapping(evidence)
    offset = offset if _integer(offset) else 0
    id_base = id_base if _integer(id_base) else 0
    phrases = {}
    for key, phrase in _mapping(evidence.get('phrases')).items():
        try:
            stamp = int(key)
        except (TypeError, ValueError, OverflowError):
            continue
        if not isinstance(phrase, dict):
            continue
        activity = deepcopy(_mapping(phrase.get('activity')))
        for field in ('face', 'track', 'profileFace'):
            if _integer(activity.get(field)):
                activity[field] += id_base
            else:
                activity.pop(field, None)
        phrases[str(stamp + offset)] = {'activity': activity}
    exclusions = []
    raw_expression = evidence.get('facialExpressions')
    expression = _mapping(raw_expression)
    raw_static = expression.get('staticFaceEvidence')
    static = _mapping(raw_static)
    static_rows = static.get('exclusions')
    invalid_static = ((raw_expression is not None and not isinstance(raw_expression, dict))
                      or (raw_static is not None and not isinstance(raw_static, dict))
                      or (static_rows is not None and not isinstance(static_rows, list)))
    if isinstance(static_rows, list) and len(_rows(static_rows)) != len(static_rows):
        invalid_static = True
    for row in _rows(static_rows):
        left, right = row.get('startMs'), row.get('endMs')
        if not (_integer(left) and _integer(right) and left < right and _integer(row.get('faceId'))):
            invalid_static = True
            continue
        item = deepcopy(row)
        item.update(startMs=left + offset, endMs=right + offset, faceId=row['faceId'] + id_base)
        if _integer(item.get('track')):
            item['track'] += id_base
        exclusions.append(item)
    if invalid_static:
        for phrase in phrases.values():
            phrase['activity']['state'] = 'unconfirmed'
    return {'phrases': phrases, 'facialExpressions': {'staticFaceEvidence': {'exclusions': exclusions}},
            'localAnchorsEligible': not invalid_static}


def remap_voice_faces(report):
    """Translate uniquely corroborated raw bindings while retaining their original namespace."""
    evidence = report.get('videoEvidence')
    if not isinstance(evidence, dict):
        return
    numbers = {voice['id']: voice['number'] for voice in _voices(report)}
    raw_matches, face_matches, editorial_matches = defaultdict(set), defaultdict(set), defaultdict(set)
    local_aliases = {}
    unresolved_raw = set()
    cues = _rows(report.get('segments'))
    for cue in cues:
        number = _source_number(cue, numbers)
        raw = cue.get('speaker')
        if number is None or cue.get('overlap') or cue.get('overlaps'):
            if _integer(raw):
                unresolved_raw.add(raw)
            continue
        if _integer(raw):
            raw_matches[raw].add(number)
        editorial = cue.get('speakerNumber')
        if _integer(editorial):
            editorial_matches[editorial].add(number)
        face = cue.get('visualFace')
        decision = _mapping(cue.get('voiceProfileEvidence'))
        if decision.get('basis') == 'explicit-video':
            proof = _mapping(decision.get('video'))
            if proof.get('basis') == 'light-asd-local':
                for row in _rows(proof.get('evidence')):
                    if _integer(row.get('faceId')):
                        face_matches[row['faceId']].add(number)
                        if _integer(row.get('profileFace')):
                            local_aliases[row['faceId']] = row['profileFace']
            elif _integer(face):
                face_matches[face].add(number)
    raw = deepcopy(_mapping(evidence.get('rawFaceSpeakers')))
    for face, binding in _mapping(evidence.get('faceSpeakers')).items():
        if isinstance(binding, dict) and not binding.get('voiceProfileMapped'):
            raw[str(face)] = deepcopy(binding)
    evidence['rawFaceSpeakers'] = raw
    mapped = {}
    for face, binding in raw.items():
        if not isinstance(binding, dict):
            continue
        try:
            face_id = int(face)
        except (TypeError, ValueError, OverflowError):
            continue
        candidates = face_matches.get(face_id)
        if candidates is None:
            label = binding.get('speaker')
            basis = binding.get('basis')
            matches = editorial_matches if basis == 'editorial' else raw_matches
            candidates = matches.get(label, set()) if _integer(label) and basis in ('editorial', 'audio-visual') else set()
            if basis != 'editorial' and _integer(label) and label in unresolved_raw:
                candidates = set()
            # A stale inferred raw binding cannot bypass the negative evidence that
            # already prevented the current cue from becoming an AV voice anchor.
            expression = _mapping(evidence.get('facialExpressions'))
            guard = _mapping(expression.get('staticFaceEvidence'))
            if any(span.get('faceId') == face_id and span.get('strongAVSpeakerAnchorEligible') is False
                   for span in _rows(guard.get('exclusions'))):
                candidates = set()
        if len(candidates) == 1:
            mapped[str(face)] = {**binding, 'speaker': next(iter(candidates)), 'voiceProfileMapped': True}
            if face_id in face_matches and binding.get('basis') == 'visible-only':
                mapped[str(face)]['basis'] = 'audio-visual'
                if face_id in local_aliases or face_id in local_aliases.values():
                    mapped[str(face)].update(confirmedIdentity=True, profileFace=local_aliases.get(face_id, face_id))
        else:
            mapped[str(face)] = {**binding, 'basis': 'visible-only', 'speaker': None, 'voiceProfileMapped': True}
    evidence['faceSpeakers'] = mapped
    for cue in cues:
        if 'linkedSpeakerNumber' not in cue:
            continue
        cue.setdefault('rawLinkedSpeakerNumber', cue['linkedSpeakerNumber'])
        cue.pop('linkedSpeakerNumber', None)
        face = cue.get('visualFace')
        binding = mapped.get(str(face)) if _integer(face) else None
        number = _source_number(cue, numbers)
        if binding and number is not None and binding.get('speaker') == number:
            cue['linkedSpeakerNumber'] = number
