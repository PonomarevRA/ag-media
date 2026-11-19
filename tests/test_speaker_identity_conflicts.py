"""Protect saved editorial groups from contradictory current face measurements."""

from copy import deepcopy
from unittest import TestCase

from tests.test_speaker_identity import report
from voice_studio.speech.speaker_identity import (
    IDENTITY_REVIEW_WARNING, apply_confirmations, make_confirmation,
)
from voice_studio.speech.speaker_profiles import cue_number


class SpeakerIdentityConflictTests(TestCase):
    def confirmation(self, *, additional=()):
        """Create a normal stored decision; bypass authority is never persisted into it."""
        return make_confirmation(report(), 1, [5], list(additional))

    def test_current_strong_pair_vetoes_replay_after_geometric_renumbering(self):
        """A measured non-match survives changed detector IDs without rewriting the saved decision."""
        confirmation = self.confirmation()
        saved = deepcopy(confirmation)
        later = report(100)
        original = deepcopy(later['segments'])
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[101, 108]]}
        self.assertEqual(apply_confirmations(later, [confirmation]), [])
        self.assertEqual(later['segments'], original)
        self.assertEqual(confirmation, saved)
        conflict, = later['speakerIdentityConflicts']
        self.assertEqual((conflict['id'], conflict['state'], conflict['basis']),
                         ('stable', 'needs-review', 'strong-face-nonmatch'))
        self.assertEqual(conflict['pairs'], [[101, 108]])

    def test_old_numeric_nonmatch_ids_do_not_veto_unrelated_current_faces(self):
        """Only current rematched canonical IDs participate in the safety comparison."""
        later = report(100)
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
        group, = apply_confirmations(later, [self.confirmation()])
        self.assertEqual(group['faceIds'], [101, 108])
        self.assertNotIn('speakerIdentityConflicts', later)

    def test_absent_failed_and_malformed_nonmatch_facts_do_not_invent_a_conflict(self):
        """Weak or unavailable measurement remains unknown rather than contradicting a user decision."""
        for tracking in ({}, {'status': 'failed', 'distinctFaces': [[1, 8]]},
                         {'status': 'ok', 'distinctFaces': 7},
                         {'status': 'ok', 'distinctFaces': [[1], [1, True], ['1', 8]]}):
            with self.subTest(tracking=tracking):
                later = report()
                later['videoEvidence']['faceTracking'] = tracking
                self.assertEqual(len(apply_confirmations(later, [self.confirmation()])), 1)
                self.assertNotIn('speakerIdentityConflicts', later)

    def test_ephemeral_reaffirmation_is_exact_and_never_changes_saved_confirmation(self):
        """A fresh caller can override this model veto only for the exact requested group."""
        confirmation = self.confirmation()
        saved = deepcopy(confirmation)
        for requested, expected in ((['stable-other'], 0), ('stable', 0), (['stable'], 1)):
            with self.subTest(requested=requested):
                later = report()
                later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
                self.assertEqual(len(apply_confirmations(later, [confirmation], reaffirmed_ids=requested)), expected)
        self.assertEqual(confirmation, saved)
        self.assertNotIn('reaffirmed_ids', confirmation)

    def test_reaffirmation_does_not_bypass_existing_simultaneous_face_guard(self):
        """The explicit model override does not relax the existing physical co-visibility constraint."""
        later = report()
        confirmation = self.confirmation()
        for hits in later['videoEvidence']['timeline']['frames'][1:4]:
            hits.append([8, .65, .2, .3, .3, 0, 8])
        self.assertEqual(apply_confirmations(later, [confirmation], reaffirmed_ids=['stable']), [])
        self.assertEqual(later['speakerIdentityConflicts'][0]['basis'], 'co-visible-faces')

    def test_unrelated_valid_group_survives_a_vetoed_group(self):
        """Reject one historical group without changing another disjoint confirmed identity."""
        later = report()
        later['videoEvidence']['faceSpeakers']['32'] = {'speaker': 2, 'basis': 'audio-visual'}
        other = make_confirmation(later, 2, [], [])
        other['id'] = 'other'
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
        group, = apply_confirmations(later, [self.confirmation(), other])
        self.assertEqual(group['id'], 'other')
        self.assertEqual(later['segments'][2]['speakerIdentityId'], 'other')
        self.assertEqual(later['speakerIdentityConflicts'][0]['id'], 'stable')

    def test_repeat_recovery_with_veto_removes_only_derived_overlay_and_restores_bindings(self):
        """An earlier applied overlay cannot survive a later veto or erase current explicit edits."""
        later = report()
        confirmation = self.confirmation(additional=[32])
        original_bindings = deepcopy(later['videoEvidence']['faceSpeakers'])
        apply_confirmations(later, [confirmation])
        later['segments'][0]['speakerNumber'] = 9
        later['speakerProfileWarning'] = 'Предыдущая заметка.'
        later['speakerProfileIndex'].append({'id': 'stable', 'confirmedIdentityId': 'stable', 'number': 1})
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[8, 32]]}
        for _ in range(2):
            self.assertEqual(apply_confirmations(later, [confirmation]), [])
        self.assertEqual(later['segments'][0]['speakerNumber'], 9)
        self.assertEqual([cue_number(cue) for cue in later['segments']], [9, 5, 2])
        self.assertTrue(all('speakerIdentityId' not in cue for cue in later['segments']))
        self.assertTrue(all('linkedSpeakerNumber' not in cue for cue in later['segments']))
        self.assertNotIn('speakerIdentityGroups', later)
        self.assertTrue(all(row.get('confirmedIdentityId') != 'stable' for row in later['speakerProfileIndex']))
        bindings = later['videoEvidence']['faceSpeakers']
        self.assertEqual(bindings[8], original_bindings['8'])
        self.assertEqual(bindings[1], {'speaker': 9, 'basis': 'editorial'})
        self.assertFalse(any(binding.get('confirmedIdentity') for binding in bindings.values()))
        self.assertEqual(later['speakerProfileWarning'].count(IDENTITY_REVIEW_WARNING), 1)
        self.assertIn('Предыдущая заметка.', later['speakerProfileWarning'])
        self.assertEqual(len(later['speakerIdentityConflicts']), 1)

    def test_legacy_overlay_without_binding_backup_restores_only_measured_source_voice(self):
        """Legacy raw integer labels do not substitute for an available source-scoped AV proof."""
        later = report()
        confirmation = self.confirmation()
        apply_confirmations(later, [confirmation])
        later['speakerIdentityGroups'][0].pop('replayFaceBindings')
        later['segments'][1].update(voiceProfileId='a' * 32, profileSpeakerNumber=5,
                                    voiceProfileEvidence={'basis': 'explicit-video'})
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
        self.assertEqual(apply_confirmations(later, [confirmation]), [])
        self.assertEqual(later['videoEvidence']['faceSpeakers'][8],
                         {'speaker': 5, 'basis': 'audio-visual', 'voiceProfileMapped': True})
        self.assertEqual(later['videoEvidence']['faceSpeakers'][1]['basis'], 'visible-only')

    def test_repeated_veto_restores_preexisting_av_link_even_without_voice_profile_fallback(self):
        """An automatic AV label must survive withdrawal instead of becoming unassigned."""
        later = report()
        confirmation = self.confirmation()
        later['segments'][1].update(linkedSpeakerNumber=5,
                                    voiceProfileEvidence={'basis': 'audio-abstain', 'state': 'abstained'})
        later['segments'][0]['speakerProfileId'] = confirmation['id']
        original = deepcopy(later['segments'])
        apply_confirmations(later, [confirmation])
        self.assertEqual(cue_number(later['segments'][1]), 1)
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
        for _ in range(2):
            self.assertEqual(apply_confirmations(later, [confirmation]), [])
        self.assertEqual(later['segments'], original)
        self.assertEqual(cue_number(later['segments'][1]), 5)

    def test_withdrawal_does_not_restore_over_a_later_explicit_linked_label(self):
        """Only the replay's own current assignment can be replaced by its backup."""
        later = report()
        confirmation = self.confirmation()
        later['segments'][1]['linkedSpeakerNumber'] = 5
        apply_confirmations(later, [confirmation])
        later['segments'][1]['linkedSpeakerNumber'] = 9
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
        apply_confirmations(later, [confirmation])
        self.assertEqual(later['segments'][1]['linkedSpeakerNumber'], 9)

    def test_overlay_backup_is_not_restored_onto_a_retimed_cue(self):
        """A previous automatic link is source-clock evidence, not authority for a different interval."""
        later = report()
        confirmation = self.confirmation()
        later['segments'][1]['linkedSpeakerNumber'] = 5
        apply_confirmations(later, [confirmation])
        later['segments'][1].update(startMs=10420, endMs=10800, speakerNumber=9)
        later['videoEvidence']['faceTracking'] = {'status': 'ok', 'distinctFaces': [[1, 8]]}
        # Keep another nonmatch participant so the explicit cue's contested face
        # cannot turn the conflict into a partial one-face recovery.
        confirmation = self.confirmation(additional=[32])
        later['videoEvidence']['faceTracking']['distinctFaces'] = [[1, 32]]
        apply_confirmations(later, [confirmation])
        self.assertNotIn('linkedSpeakerNumber', later['segments'][1])
        self.assertEqual(later['segments'][1]['speakerNumber'], 9)
