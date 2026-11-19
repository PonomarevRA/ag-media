"""Exercise guardrails for voice and face agreement and stable profile references."""

from unittest import TestCase
from unittest.mock import patch

from voice_studio.vision.face_speakers import bind_faces
from voice_studio.speech.speaker_profiles import attach_ids
from voice_studio.speech.voiced_speaker_review import review_voiced_windows
from voice_studio.speech.speaker_review import VoiceProfile, review_speakers, review_visible_cues, validate_pairs


class SpeakerReviewTests(TestCase):
    @staticmethod
    def short_reply_fixture():
        """Provide two independent references and one partially confirmed reply."""
        cues = [
            {"startMs": 0, "endMs": 2000, "speaker": 1, "visualFace": 3},
            {"startMs": 2000, "endMs": 3000, "speaker": 1},
            {"startMs": 3000, "endMs": 5000, "speaker": 1, "visualFace": 3},
        ]
        candidate = {"track": 11, "face": 7, "activeFrames": 18, "phraseFrames": 25, "sample": {"time_ms": 2500}}
        phrases = {"2000": {"activity": {"state": "partial-or-ambiguous", "referenceCandidate": candidate}}}
        evidence = {"timeline": {"stepMs": 40, "frames": []}, "phrases": phrases,
                    "faceTracking": {"distinctFaces": [[3, 7]]},
                    "editorTimeline": {"shots": [{"startMs": 0}, {"startMs": 3000}]}}
        return {"segments": cues, "videoEvidence": evidence, "speakerCount": 1}

    def test_short_reply_keeps_original_cluster_and_gets_a_distinct_speaker(self):
        result = self.short_reply_fixture()
        with patch('voice_studio.speech.speaker_review.extract_embeddings', return_value={0: [(0, 1)], 1: [(1, 0), (1, 0)]}):
            report = review_speakers(result, None, 0)
        bind_faces(result)
        cue = result["segments"][1]
        self.assertEqual((cue["speaker"], cue["linkedSpeakerNumber"], cue["visualFace"]), (1, 2, 7))
        self.assertEqual(result["speakerCount"], 2)
        self.assertEqual(report["shortReplyDecisions"][0]["voiceCosine"], 0)

    def test_short_reply_abstains_on_overlap_or_unhelpful_voice(self):
        for case in ("overlap", "matching voice", "incoherent anchor voices", "same face", "inside cut"):
            with self.subTest(case=case):
                result = self.short_reply_fixture()
                cue = result["segments"][1]
                evidence = result["videoEvidence"]
                anchors = [(1, 0), (1, 0)]
                sample = [(0, 1)]
                if case == "overlap":
                    cue["overlaps"] = [{"startMs": 2100, "endMs": 2800}]
                if case == "matching voice":
                    sample = [(1, 0)]
                if case == "incoherent anchor voices":
                    anchors = [(1, 0), (.1, .995)]
                if case == "same face":
                    evidence["faceTracking"]["distinctFaces"] = []
                if case == "inside cut":
                    evidence["editorTimeline"]["shots"].append({"startMs": 2400})
                with patch('voice_studio.speech.speaker_review.extract_embeddings', return_value={0: sample, 1: anchors}):
                    self.assertIsNone(review_speakers(result, None, 0))
                self.assertNotIn("profileSpeakerNumber", cue)

    def test_distinct_faces_and_simultaneous_voices_block_cluster_fusion(self):
        voice = [(1.0, 0.0), (1.0, 0.0)]
        first = VoiceProfile(1, 2, 0, voice, False)
        second = VoiceProfile(2, 3, 1300, voice, False)
        self.assertEqual(validate_pairs({1: first, 2: second}, []), [])
        second.face = 2
        self.assertEqual(validate_pairs({1: first, 2: second}, [{"speakers": [1, 2]}]), [])
        self.assertEqual(validate_pairs({1: first, 2: second}, [{"speakers": [1, 3]}]), [])
        self.assertEqual(validate_pairs({1: first, 2: second}, [])[0]["to"], 1)

    def test_local_review_changes_only_the_exact_visible_fragment(self):
        profiles = {1: VoiceProfile(1, 6, 0, [(1, 0)], False),
                    2: VoiceProfile(2, 6, 1300, [(.60, .80)], False)}
        cues = [
            {"startMs": 0, "endMs": 1300, "speaker": 1, "visualFace": 6},
            {"startMs": 1300, "endMs": 2600, "speaker": 2, "visualFace": 6},
            {"startMs": 2600, "endMs": 4100, "speaker": 2},
        ]
        turns = [{"startMs": 0, "endMs": 2600, "speaker": 4}, {"startMs": 2600, "endMs": 4100, "speaker": 5}]
        decisions = review_visible_cues(cues, profiles, turns, 0, {})
        self.assertEqual([row["startMs"] for row in decisions], [1300])
        self.assertEqual(cues[1]["profileSpeakerNumber"], 1)
        self.assertNotIn("profileSpeakerNumber", cues[2])
        self.assertEqual(cues[1]["speaker"], 2)

    def test_pinned_profile_reuse_requires_an_unambiguous_matching_face(self):
        old = {"id": "fixed", "faceId": 4, "anchorsMs": [10, 20], "pinned": True}
        matching = {"faceId": 4, "anchorsMs": [20, 30], "pinned": True}
        wrong_face = {"faceId": 5, "anchorsMs": [10, 20], "pinned": True}
        rows = attach_ids([matching, wrong_face], [old])
        self.assertEqual(rows[0]["id"], "fixed")
        self.assertTrue(rows[0]["reused"])
        self.assertFalse(rows[1]["reused"])


    def test_voiced_window_repairs_display_label_without_rewriting_overlap_or_raw_cluster(self):
        anchor = {"startMs": 0, "endMs": 1900, "speaker": 1, "visualFace": 3}
        reply = {"startMs": 2100, "endMs": 4500, "speaker": 4,
                 "speakerAudioWindow": {"startMs": 2900, "endMs": 4380}}
        unrelated = {"startMs": 4600, "endMs": 6000, "speaker": 4}
        frames = [[] for _ in range(150)]
        for index in range(74, 103):
            frames[index] = [[5, .2, .2, .1, .1, 1, 3]]
        timeline = {"stepMs": 40, "frames": frames}
        extract = lambda *_: {0: [(1., 0.)], 1: [(.56, .8285)]}
        decisions = review_voiced_windows([anchor, reply, unrelated], timeline, [], None, 0, extract)
        self.assertEqual([(row["from"], row["to"]) for row in decisions], [(4, 1)])
        self.assertEqual(reply["speaker"], 4)
        self.assertEqual(reply.pop("profileSpeakerNumber"), 1)
        self.assertNotIn("profileSpeakerNumber", unrelated)
        self.assertEqual(review_voiced_windows([anchor, reply], timeline, [{"speakers": [1, 4]}],
                                               None, 0, extract), [])
        frames[80].append([6, .5, .2, .1, .1, 1, 7])
        self.assertEqual(review_voiced_windows([anchor, reply], timeline, [], None, 0, extract), [])
