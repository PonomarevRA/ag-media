"""Exercise same-source identity confirmations independently of detector numbering."""

from copy import deepcopy
from unittest import TestCase

from voice_studio.speech.speaker_identity import apply_confirmations, make_confirmation
from voice_studio.speech.speaker_profiles import attach_ids, collect_profiles, cue_number


def report(shift=0):
    """Produce two returning face tracks and one face present without a speech cue."""
    frames = [([], []) for _ in range(30)]
    for index, face in enumerate((1, 8, 32)):
        for frame in range(index * 10, index * 10 + 10):
            frames[frame] = [[face, .2, .2, .3, .3, 0, face + shift]]
    return {
        "segments": [
            {"startMs": 0, "endMs": 360, "speaker": 1, "text": "First", "visualFace": 1 + shift},
            {"startMs": 420, "endMs": 800, "speaker": 5, "text": "Second", "visualFace": 8 + shift},
            {"startMs": 840, "endMs": 1160, "speaker": 2, "text": "Another"},
        ],
        "speakerProfileIndex": [{"number": 1, "id": "stable"}],
        "videoEvidence": {"timeline": {"stepMs": 40, "frames": frames},
                          "faceSpeakers": {str(1 + shift): {"speaker": 1, "basis": "audio-visual"},
                                           str(8 + shift): {"speaker": 5, "basis": "audio-visual"}}},
    }


class SpeakerIdentityTests(TestCase):
    def test_reuses_known_identity_after_renumbering(self):
        """Require matching source frames and keep unconfirmed speakers separate."""
        confirmation = make_confirmation(report(), 1, [5], [32])
        later = report(100)
        group, = apply_confirmations(later, [confirmation])
        self.assertEqual(group["faceIds"], [101, 108, 132])
        self.assertEqual([cue_number(cue) for cue in later["segments"]], [1, 1, 2])
        self.assertEqual([cue["speaker"] for cue in later["segments"]], [1, 5, 2])
        profile = next(p for p in attach_ids(collect_profiles(later), []) if p["number"] == 1)
        self.assertEqual((profile["id"], profile["faceId"], profile["pinned"]), ("stable", 101, True))

    def test_reads_missing_appearance_only_from_confirmed_tracks(self):
        confirmation = make_confirmation(report(), 1, [5], [32])
        later = report()
        later["videoEvidence"]["editorTimeline"] = {"faceAttributes": {
            "1": {"age": 26},
            "8": {"age": 40, "appearanceClass": "male", "hairSwatch": "#2d221d", "hairSamples": 2},
            "32": {"appearanceClass": "female"},
            "99": {"age": 70},
        }}
        apply_confirmations(later, [confirmation])
        profile = next(row for row in collect_profiles(later) if row["number"] == 1)
        self.assertEqual(profile["faceId"], 1)
        self.assertEqual(profile["appearance"], {"age": 26, "hairSwatch": "#2d221d", "hairSamples": 2})

    def test_new_explicit_conflict_blocks_visual_assignment(self):
        """Do not propagate a confirmed group over a contradictory human cue edit."""
        confirmation = make_confirmation(report(), 1, [5], [32])
        later = report()
        later["segments"][1]["speakerNumber"] = 7
        group, = apply_confirmations(later, [confirmation])
        self.assertEqual(group["faceIds"], [1, 32])
        self.assertEqual(cue_number(later["segments"][1]), 7)

    def test_missing_geometry_does_not_replay_a_label_by_number(self):
        """Identical numeric labels in a changed track cannot establish identity."""
        source = report()
        confirmation = make_confirmation(source, 1, [5], [32])
        unrelated = deepcopy(source)
        unrelated["videoEvidence"]["timeline"]["frames"] = [[] for _ in range(30)]
        self.assertEqual(apply_confirmations(unrelated, [confirmation]), [])
        self.assertEqual([cue_number(cue) for cue in unrelated["segments"]], [1, 5, 2])

    def test_saved_replay_cannot_join_people_visible_together(self):
        """A previous manual group cannot override simultaneous disjoint face evidence."""
        source = report()
        confirmation = make_confirmation(source, 1, [5], [])
        for hits in source["videoEvidence"]["timeline"]["frames"][1:4]:
            hits.append([8, .65, .2, .3, .3, 0, 8])
        self.assertEqual(apply_confirmations(source, [confirmation]), [])
        self.assertEqual([cue_number(cue) for cue in source["segments"]], [1, 5, 2])
