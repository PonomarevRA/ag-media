"""Keep native EOF repair bounded while the public dialogue validator stays strict."""

from copy import deepcopy
from unittest import TestCase

from voice_studio.app.dialogue_audit_service import validate_inputs
from voice_studio.speech.dialogue_audit import audit_dialogue, stable_hash
from voice_studio.speech.native_turn_bounds import audit_native_dialogue
from voice_studio.speech.speaker_pipeline import attach_dialogue_audit


class NativeTurnBoundsTests(TestCase):
    """Verify provenance, raw preservation, caller trust and hard negative intervals."""

    def fixture(self, end=1024):
        """Supply a native turn with the same 24 ms overshoot as the corpus failure."""
        report = {'durationMs': 1000, 'sourceDigest': 'a' * 64, 'language': 'ps',
                  'segments': [{'startMs': 0, 'endMs': 1000, 'text': 'test'}],
                  'speechTurns': [{'startMs': 100, 'endMs': end, 'speaker': 1, 'raw': {'keep': True}}]}
        return report, {'sha256': 'a' * 64, 'durationMs': 1000}

    def test_native_copy_repairs_eof_and_preserves_raw_and_honest_hashes(self):
        report, source = self.fixture()
        before = deepcopy(report)
        turns = {'turns': deepcopy(report['speechTurns']), 'clockOffsetMs': 0}
        result = audit_native_dialogue(report, turns, source=source)
        self.assertEqual(report, before)
        self.assertEqual(turns['turns'][0]['endMs'], 1024)
        provenance = result['nativeTurnBounds']
        self.assertEqual(provenance['rawTurnsHash'], stable_hash(turns))
        self.assertEqual(provenance['auditTurnsHash'], result['inputHashes']['turns'])
        self.assertNotEqual(provenance['rawTurnsHash'], provenance['auditTurnsHash'])
        self.assertEqual(result['inputHashes']['report'], stable_hash(report))
        self.assertEqual(provenance['adjustments'], [{'turnIndex': 0, 'originalEndMs': 1024,
                         'usedEndMs': 1000, 'basis': 'bounded-native-eof-overrun'}])
        self.assertEqual(result, audit_native_dialogue(report, turns, source=source))

    def test_public_validator_still_rejects_the_same_native_overshoot(self):
        report, source = self.fixture()
        with self.assertRaisesRegex(ValueError, 'invalid source-clock interval'):
            audit_dialogue(report, report['speechTurns'], source=source)

    def test_exact_eof_does_not_change_strict_cache_or_result(self):
        report, source = self.fixture(1000)
        self.assertEqual(audit_native_dialogue(report, report['speechTurns'], source=source),
                         audit_dialogue(report, report['speechTurns'], source=source))

    def test_policy_accepts_40ms_but_rejects_41ms(self):
        report, source = self.fixture(1040)
        self.assertEqual(audit_native_dialogue(report, report['speechTurns'], source=source)
                         ['nativeTurnBounds']['maxEofOvershootMs'], 40)
        report['speechTurns'][0]['endMs'] = 1041
        with self.assertRaises(ValueError):
            audit_native_dialogue(report, report['speechTurns'], source=source)

    def test_corrupt_negative_and_wholly_outside_intervals_still_fail(self):
        for start, end in [(-1, 1024), (1000, 1024), (1001, 1024), (True, 1024),
                           (100, '1024'), (100, True), (100, 90)]:
            with self.subTest(start=start, end=end):
                report, source = self.fixture()
                report['speechTurns'][0].update(startMs=start, endMs=end)
                with self.assertRaises(ValueError):
                    audit_native_dialogue(report, report['speechTurns'], source=source)

    def test_different_external_turns_are_not_native_repair_candidates(self):
        report, source = self.fixture()
        external = deepcopy(report['speechTurns'])
        external[0]['speaker'] = 2
        with self.assertRaises(ValueError):
            audit_native_dialogue(report, external, source=source)

    def test_raw_overrun_change_invalidates_cache_even_when_normalized_turns_match(self):
        report, source = self.fixture(1024)
        first = audit_native_dialogue(report, report['speechTurns'], source=source)
        report['speechTurns'][0]['endMs'] = 1025
        second = audit_native_dialogue(report, report['speechTurns'], source=source)
        self.assertEqual(first['inputHashes']['turns'], second['inputHashes']['turns'])
        self.assertNotEqual(first['cacheKey'], second['cacheKey'])

    def test_service_validation_accepts_matching_native_copy_without_mutating(self):
        report, _ = self.fixture()
        before = deepcopy(report)
        validate_inputs(report, report['speechTurns'], False, 12, 100000, 60)
        self.assertEqual(report, before)

    def test_batch_pipeline_publishes_completed_audit_and_keeps_raw_eof(self):
        report, _ = self.fixture()
        attach_dialogue_audit(report, 'a' * 64)
        self.assertIn('summary', report['dialogueAudit'])
        self.assertEqual(report['speechTurns'][0]['endMs'], 1024)
        self.assertEqual(report['dialogueAudit']['nativeTurnBounds']['adjustmentCount'], 1)
