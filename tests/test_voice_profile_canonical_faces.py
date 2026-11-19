"""Verify canonical face propagation without weakening acoustic or raw-label policy."""

from array import array
from copy import deepcopy
from math import sin
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import wave

from voice_studio.speech.voice_profiles import VoiceProfileRegistry, apply_voice_profiles


class OrderedExtractor:
    """Supply conflicting voices so canonical speaking proof must decide identity."""

    def __init__(self, vectors):
        """Consume the caller's fixed features once per selected clean audio interval."""
        self.vectors = iter(vectors)

    def create_stream(self):
        """Reuse this minimal stream protocol for one serial test extraction."""
        return self

    def accept_waveform(self, rate, samples):
        """Leave actual PCM quality gating to the production profile helper."""
        pass

    def input_finished(self):
        """Complete the current bounded fixture audio interval."""
        pass

    def is_ready(self, stream):
        """Accept each finished fixture span."""
        return True

    def compute(self, stream):
        """Return the next controlled embedding instead of invoking a native model."""
        return next(self.vectors)


class CanonicalFaceVoiceTests(TestCase):
    def test_returning_canonical_face_reuses_identity_while_distinct_face_keeps_own_profile(self):
        """Same-person return beats changed audio; same raw label never merges two faces."""
        segments = [
            {'startMs': 0, 'endMs': 1800, 'speaker': 1, 'text': 'First person.'},
            {'startMs': 2000, 'endMs': 3800, 'speaker': 1, 'text': 'Different visible person.'},
            {'startMs': 4000, 'endMs': 5800, 'speaker': 9, 'text': 'First person returns.'},
        ]
        report = {'segments': deepcopy(segments), 'videoEvidence': {'phrases': {}}}
        for cue, face in zip(report['segments'], [6, 20, 6]):
            report['videoEvidence']['phrases'][str(cue['startMs'])] = {'activity': {
                'state': 'visible-active', 'face': face, 'activeFrames': 40, 'phraseFrames': 45}}
        registry = VoiceProfileRegistry('fixture-source', encoder_id='test-encoder')
        with TemporaryDirectory() as directory:
            wav = Path(directory) / 'audio.wav'
            pcm = array('h', (int(4000 * sin(index * .1)) for index in range(16000 * 6)))
            with wave.open(str(wav), 'wb') as audio:
                audio.setparams((1, 2, 16000, len(pcm), 'NONE', 'none'))
                audio.writeframes(pcm.tobytes())
            apply_voice_profiles(report, wav, 0, source_id='fixture-source', registry=registry,
                                 visual_scope='fixture-source:window:0',
                                 extractor=OrderedExtractor([[1, 0], [1, 0], [0, 1]]))

        first, other, returning = report['segments']
        self.assertEqual(first['voiceProfileId'], returning['voiceProfileId'])
        self.assertNotEqual(first['voiceProfileId'], other['voiceProfileId'])
        self.assertEqual(len(report['voiceProfiles']['profiles']), 2)
        for original, actual in zip(segments, report['segments']):
            self.assertEqual({key: actual[key] for key in original}, original)
            self.assertEqual(actual['voiceProfileEvidence']['basis'], 'explicit-video')
        self.assertIn('explicit-video-overrides-audio', other['voiceProfileEvidence']['reasons'])
