"""Production-helper regressions for source identity and negative visual/audio facts."""

from array import array
from copy import deepcopy
from math import sin
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio.speech.speaker_pipeline import (attach_voice_profiles, enrich_source_speech,
    profile_visual_evidence, project_voice_index, remap_voice_faces)
from voice_studio.speech.voice_profiles import VoiceProfileRegistry

VOICE_A = 'a' * 32
VOICE_B = 'b' * 32


def voice(identity, number):
    """Supply the smallest valid source-profile public-index fixture."""
    return {'id': identity, 'number': number, 'references': [], 'sampleCount': 1, 'status': 'audio-provisional'}


def cue(start=0, raw=1, identity=VOICE_A, number=1, **extra):
    """Keep raw and source namespaces deliberately overlapping in numeric fixtures."""
    return {'startMs': start, 'endMs': start + 1800, 'speaker': raw,
            'voiceProfileId': identity, 'profileSpeakerNumber': number,
            'voiceProfileEvidence': {'basis': 'robust-audio-match'}, **extra}


class VoicePipelineHelperTests(TestCase):
    def setUp(self):
        """Create a bounded real PCM WAV and a registry with no native model dependency."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.wav = Path(self.directory.name) / 'audio.wav'
        samples = array('h', (int(4000 * sin(index * .1)) for index in range(16000 * 2)))
        with wave.open(str(self.wav), 'wb') as audio:
            audio.setparams((1, 2, 16000, len(samples), 'NONE', 'none'))
            audio.writeframes(samples.tobytes())
        self.registry = VoiceProfileRegistry('source', encoder_id='test-encoder')

    def attach(self, report):
        """Exercise the actual production helper and registry with only the encoder replaced."""
        with patch('voice_studio.speech.speaker_review.extract_embeddings',
                   side_effect=lambda _wav, spans: {index: [[1, 0]] for index in spans}):
            return attach_voice_profiles(report, self.wav, 0, 'source', registry=self.registry, visual_scope='source:window:0')

    def visual_report(self, *, frozen=False):
        """Give a clean ASD cue and optional independent static/frozen veto."""
        evidence = {'phrases': {'0': {'activity': {'state': 'visible-active', 'face': 7, 'track': 4,
                                               'activeFrames': 40, 'phraseFrames': 45}}},
                    'faceSpeakers': {'7': {'speaker': 1, 'basis': 'audio-visual'}}}
        if frozen:
            evidence['facialExpressions'] = {'staticFaceEvidence': {'exclusions': [
                {'startMs': 0, 'endMs': 1800, 'faceId': 7, 'track': 4,
                 'strongAVSpeakerAnchorEligible': False}]}}
        return {'segments': [{'startMs': 0, 'endMs': 1800, 'speaker': 1,
                              'visualFace': 7, 'linkedSpeakerNumber': 1}], 'videoEvidence': evidence}

    def test_static_veto_reaches_actual_profile_extraction_and_prevents_stale_face_rebinding(self):
        """Negative pixel evidence must survive the production helper and its appearance projection."""
        report = self.visual_report(frozen=True)
        original = deepcopy(report)
        self.attach(report)
        self.assertEqual(report['segments'][0]['voiceProfileEvidence']['basis'], 'new-audio-profile')
        self.assertIsNone(self.registry.index()[0]['subjectId'])
        self.assertFalse(self.registry.index()[0]['references'][0]['explicitVideo'])
        project_voice_index(report)
        self.assertIsNone(report['speakerProfileIndex'][0]['faceId'])
        self.assertEqual(report['videoEvidence']['faceSpeakers']['7']['basis'], 'visible-only')
        self.assertEqual(report['videoEvidence']['rawFaceSpeakers']['7'], original['videoEvidence']['faceSpeakers']['7'])
        self.assertNotIn('linkedSpeakerNumber', report['segments'][0])
        self.assertEqual(report['segments'][0]['rawLinkedSpeakerNumber'], 1)

    def test_clean_asd_proof_attaches_and_maps_the_source_profile(self):
        """A confirmed current talking face can enrich and display its source voice profile."""
        report = self.visual_report()
        self.attach(report)
        self.assertEqual(report['segments'][0]['voiceProfileEvidence']['basis'], 'explicit-video')
        project_voice_index(report)
        self.assertEqual(report['speakerProfileIndex'][0]['faceId'], 7)
        self.assertEqual(report['videoEvidence']['faceSpeakers']['7']['speaker'], 1)

    def test_editorial_speaking_confirmation_retains_priority_over_static_veto(self):
        """An independently confirmed talker remains valid even with a static source image."""
        report = self.visual_report(frozen=True)
        report['segments'][0]['speakerVideoEvidence'] = {'subjectId': 'person-a', 'confirmed': True,
            'state': 'visible-active', 'basis': 'editorial'}
        self.attach(report)
        project_voice_index(report)
        self.assertEqual(report['segments'][0]['voiceProfileEvidence']['basis'], 'explicit-video')
        self.assertEqual(report['speakerProfileIndex'][0]['faceId'], 7)

    def test_independent_overlaps_prevent_learning_and_clear_raw_linked_namespace(self):
        """An undecorated overlap cannot create a voice or leak raw label one as source one."""
        report = self.visual_report()
        report['speechOverlaps'] = [{'startMs': 1000, 'endMs': 1400, 'speakers': [1, 2]}]
        self.attach(report)
        project_voice_index(report)
        self.assertEqual(self.registry.profiles, [])
        self.assertEqual(report['segments'][0]['voiceProfileEvidence']['basis'], 'overlap')
        self.assertNotIn('linkedSpeakerNumber', report['segments'][0])
        self.assertEqual(report['speakerCount'], 0)

    def test_malformed_optional_legacy_evidence_does_not_crash_or_invent_av_proof(self):
        """Invalid arrays, static rows, activity objects and time keys are safely ignored."""
        report = {'segments': [None, {'startMs': 'bad', 'endMs': 1800},
                              {'startMs': 0, 'endMs': 1800, 'speaker': 1, 'speakerAudioWindow': 'bad'}],
                  'videoEvidence': {'phrases': {'bad': [], '0': {'activity': 'bad'}},
                                    'facialExpressions': {'staticFaceEvidence': {'exclusions': [None, {'startMs': None}]}}}}
        self.attach(report)
        self.assertEqual(len(report['segments']), 3)
        self.assertEqual(report['segments'][2]['speakerAudioWindow'], 'bad')
        self.assertEqual(report['segments'][2]['voiceProfileEvidence']['basis'], 'new-audio-profile')

    def test_malformed_negative_overlap_evidence_abstains_without_learning(self):
        """An unknown overlap interval is not silently interpreted as clean speech."""
        report = self.visual_report()
        report['speechOverlaps'] = [{'startMs': None, 'endMs': 1400}]
        self.attach(report)
        self.assertEqual(self.registry.profiles, [])
        self.assertIn('invalid-overlap-evidence', report['segments'][0]['voiceProfileEvidence']['reasons'])
        self.assertNotIn('overlap', report['segments'][0])

    def test_malformed_static_ledger_disables_inferred_av_but_keeps_audio_fallback(self):
        """A corrupt veto ledger cannot accidentally authorize model-only speaking anchors."""
        report = self.visual_report()
        report['videoEvidence']['facialExpressions'] = {'staticFaceEvidence': {'exclusions': ['invalid']}}
        self.attach(report)
        self.assertEqual(report['segments'][0]['voiceProfileEvidence']['basis'], 'new-audio-profile')
        self.assertIsNone(self.registry.index()[0]['subjectId'])

    def test_invalid_legacy_face_and_index_rows_do_not_crash_projection(self):
        """Malformed optional metadata cannot supply appearance to a valid source profile."""
        report = {'segments': [cue()], 'voiceProfiles': {'profiles': [None, voice(VOICE_A, 1)]},
                  'speakerProfileIndex': [None, {'number': 1, 'faceId': 99}],
                  'videoEvidence': {'faceSpeakers': {'bad': None, 'also-bad': {'speaker': []}},
                                    'rawFaceSpeakers': {'oops': [], 'bad': {'speaker': []}}, 'editorTimeline': None}}
        project_voice_index(report)
        self.assertIsNone(report['speakerProfileIndex'][0]['faceId'])
        self.assertEqual(report['speakerProfileIndex'][0]['id'], VOICE_A)


class VoiceNamespaceTests(TestCase):
    def test_raw_number_collision_does_not_copy_previous_identity_or_appearance(self):
        """Source profile one never inherits unrelated raw speaker one's face/clothes."""
        report = {'segments': [cue(raw=5)], 'voiceProfiles': {'profiles': [voice(VOICE_A, 1)]},
                  'speakerProfileIndex': [{'id': 'raw-person', 'number': 1, 'faceId': 99, 'pinned': True,
                                           'appearance': {'hairSwatch': '#ff0000'}}],
                  'videoEvidence': {'faceSpeakers': {'99': {'speaker': 1, 'basis': 'audio-visual'}}}}
        project_voice_index(report)
        row = report['speakerProfileIndex'][0]
        self.assertIsNone(row['faceId'])
        self.assertFalse(row['pinned'])
        self.assertNotIn('appearance', row)
        self.assertEqual(report['segments'][0]['speaker'], 5)

    def test_previous_same_source_voice_id_preserves_appearance(self):
        """Stable source identity, unlike integer coincidence, permits reviewed appearance reuse."""
        report = {'segments': [], 'voiceProfiles': {'profiles': [voice(VOICE_A, 1)]},
                  'speakerProfileIndex': [{'id': VOICE_A, 'scope': 'source', 'number': 1, 'faceId': 7, 'pinned': True}]}
        project_voice_index(report)
        self.assertEqual(report['speakerProfileIndex'][0]['faceId'], 7)

    def test_unique_raw_to_source_binding_is_remapped_and_original_retained(self):
        """A confirmed raw face binding follows a unique source profile, preserving raw labels."""
        report = {'segments': [cue(raw=4, number=2, visualFace=7, linkedSpeakerNumber=4)],
                  'voiceProfiles': {'profiles': [voice(VOICE_A, 2)]},
                  'videoEvidence': {'faceSpeakers': {'7': {'speaker': 4, 'basis': 'audio-visual'}}}}
        remap_voice_faces(report)
        self.assertEqual(report['videoEvidence']['faceSpeakers']['7']['speaker'], 2)
        self.assertEqual(report['videoEvidence']['rawFaceSpeakers']['7']['speaker'], 4)
        self.assertEqual(report['segments'][0]['speaker'], 4)
        self.assertEqual(report['segments'][0]['linkedSpeakerNumber'], 2)
        remap_voice_faces(report)
        self.assertEqual(report['videoEvidence']['rawFaceSpeakers']['7']['speaker'], 4)

    def test_ambiguous_raw_binding_abstains_and_cannot_restore_a_raw_linked_label(self):
        """One raw cluster split across profiles cannot attach a face to either profile."""
        report = {'segments': [cue(raw=4, visualFace=7, linkedSpeakerNumber=4),
                               cue(start=3000, raw=4, identity=VOICE_B, number=2)],
                  'voiceProfiles': {'profiles': [voice(VOICE_A, 1), voice(VOICE_B, 2)]},
                  'videoEvidence': {'faceSpeakers': {'7': {'speaker': 4, 'basis': 'audio-visual'}}}}
        remap_voice_faces(report)
        self.assertEqual(report['videoEvidence']['faceSpeakers']['7']['basis'], 'visible-only')
        self.assertNotIn('linkedSpeakerNumber', report['segments'][0])

    def test_partial_raw_cluster_mapping_cannot_bind_a_face_on_abstained_speech(self):
        """A unique match among only known cues does not resolve the rest of a raw cluster."""
        report = {'segments': [cue(raw=4), {'startMs': 3000, 'endMs': 3400, 'speaker': 4,
                    'visualFace': 7, 'linkedSpeakerNumber': 4, 'voiceProfileEvidence': {'state': 'abstained'}}],
                  'voiceProfiles': {'profiles': [voice(VOICE_A, 1)]},
                  'videoEvidence': {'faceSpeakers': {'7': {'speaker': 4, 'basis': 'audio-visual'}}}}
        remap_voice_faces(report)
        self.assertIsNone(report['videoEvidence']['faceSpeakers']['7']['speaker'])
        self.assertNotIn('linkedSpeakerNumber', report['segments'][1])

    def test_visible_face_presence_without_confirmed_raw_binding_does_not_attach(self):
        """A visible-only face cannot inherit whichever audio profile happens to play."""
        report = {'segments': [cue(raw=4, visualFace=7)], 'voiceProfiles': {'profiles': [voice(VOICE_A, 1)]},
                  'videoEvidence': {'faceSpeakers': {'7': {'speaker': 4, 'basis': 'visible-only'}}}}
        remap_voice_faces(report)
        self.assertIsNone(report['videoEvidence']['faceSpeakers']['7']['speaker'])


class VoiceEvidenceCacheTests(TestCase):
    def test_offset_projection_preserves_veto_and_rejects_malformed_coordinates(self):
        """A valid veto and its ASD face use exactly the same offset source namespace."""
        evidence = {'phrases': {'100': {'activity': {'face': 7, 'track': 2}}, 'bad': [], '200': None},
                    'facialExpressions': {'staticFaceEvidence': {'exclusions': [
                        {'startMs': 50, 'endMs': 300, 'faceId': 7, 'track': 2,
                         'strongAVSpeakerAnchorEligible': False}, {'startMs': None}, None]}}}
        projected = profile_visual_evidence(evidence, offset=1000, id_base=40)
        self.assertEqual(projected['phrases']['1100']['activity']['face'], 47)
        span = projected['facialExpressions']['staticFaceEvidence']['exclusions'][0]
        self.assertEqual((span['startMs'], span['endMs'], span['faceId'], span['track']), (1050, 1300, 47, 42))
        self.assertFalse(span['strongAVSpeakerAnchorEligible'])
        self.assertEqual(len(projected['phrases']), 1)

    def test_warm_result_keeps_the_same_identity_after_additive_projection(self):
        """Projected voice fields must not invalidate the very cache that produced them."""
        from voice_studio.graph.media_analysis_revolver import result_fingerprint
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.wav'
            source.write_bytes(b'fixed source')
            report = {'segments': [{'startMs': 0, 'endMs': 1800, 'speaker': 5,
                                    'visualFace': 7, 'linkedSpeakerNumber': 5}], 'durationMs': 2000,
                      'videoEvidence': {'faceSpeakers': {'7': {'speaker': 5, 'basis': 'audio-visual'}}}}
            class Cache:
                """Use real fingerprints with an in-memory checkpoint fixture."""
                revision = 'same-model-code'
                revolver = SimpleNamespace(directory=Path(directory))

                def __init__(self):
                    """Count only genuine cache misses."""
                    self.saved = {}
                    self.computes = 0

                def run(self, name, settings, upstream, operation, valid, **kwargs):
                    """Replay cached results exactly as the production cache does."""
                    key = result_fingerprint([settings, upstream])
                    if key not in self.saved:
                        self.computes += 1
                        self.saved[key] = operation()
                    return deepcopy(self.saved[key])

            def attach(layer, wav, offset, identity, **kwargs):
                """Supply one stable source identity to the surrounding real projection logic."""
                layer['segments'][0].update(voiceProfileId=VOICE_A, profileSpeakerNumber=1, rawSpeaker=5,
                                          voiceProfileEvidence={'basis': 'robust-audio-match'})
                return {'profiles': [voice(VOICE_A, 1)]}

            cache = Cache()
            with patch('voice_studio.speech.speaker_pipeline.attach_voice_profiles', side_effect=attach):
                enrich_source_speech(report, source, lambda: source, 0, cache)
                enrich_source_speech(report, source, lambda: source, 0, cache)
            self.assertEqual(cache.computes, 1)
            self.assertEqual(report['speakerProfileIndex'][0]['id'], VOICE_A)
            self.assertEqual(report['segments'][0]['rawLinkedSpeakerNumber'], 5)
            self.assertEqual(report['segments'][0]['linkedSpeakerNumber'], 1)

    def test_stage_and_private_registry_keys_change_with_source_veto_and_overlap(self):
        """Changed negative evidence must invalidate both cached output and learned references."""
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.wav'
            source.write_bytes(b'first source')
            captured = []
            class Cache:
                """Capture the exact production cache dependencies while invoking its operation."""
                revision = 'model-code-revision'
                revolver = SimpleNamespace(directory=Path(directory))

                def run(self, name, settings, upstream, operation, valid, **kwargs):
                    """Keep source and negative facts alongside the computed registry path."""
                    captured.append({'upstream': deepcopy(upstream), 'settings': settings})
                    return operation()

            paths = []
            def attach(layer, wav, offset, identity, **kwargs):
                """Record the real helper's private registry location without doing inference."""
                paths.append(kwargs['cache_path'])
                return {'profiles': []}

            base = {'segments': [{'startMs': 0, 'endMs': 1800, 'speaker': 1}], 'durationMs': 2000}
            frozen = deepcopy(base)
            frozen['videoEvidence'] = {'facialExpressions': {'staticFaceEvidence': {'exclusions': [
                {'startMs': 0, 'endMs': 1800, 'faceId': 7, 'strongAVSpeakerAnchorEligible': False}]}}}
            mixed = deepcopy(base)
            mixed['speechOverlaps'] = [{'startMs': 1000, 'endMs': 1200, 'speakers': [1, 2]}]
            with patch('voice_studio.speech.speaker_pipeline.attach_voice_profiles', side_effect=attach):
                for report in [base, frozen, mixed]:
                    enrich_source_speech(deepcopy(report), source, lambda: source, 0, Cache())
                source.write_bytes(b'different immutable source')
                enrich_source_speech(deepcopy(base), source, lambda: source, 0, Cache())
            self.assertEqual(len(set(paths)), 4)
            self.assertNotEqual(captured[0]['upstream'], captured[1]['upstream'])
            self.assertNotEqual(captured[0]['upstream'], captured[2]['upstream'])
            self.assertNotEqual(captured[0]['upstream']['sourceId'], captured[3]['upstream']['sourceId'])
            self.assertFalse(captured[1]['upstream']['videoProofs']['facialExpressions']['staticFaceEvidence']['exclusions'][0]['strongAVSpeakerAnchorEligible'])
            self.assertEqual(len(captured[2]['upstream']['overlaps']), 1)
