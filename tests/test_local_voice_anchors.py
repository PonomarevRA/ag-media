"""Guard voice sampling and identity promotion on a cue with multiple shots."""

from unittest import TestCase

from voice_studio.speech.local_voice_anchors import local_anchors, reconcile_anchors, review_pairs
from voice_studio.speech.voice_profile_evidence import speaking_evidence


def evidence():
    """Create two exclusive speaking faces and the original source clock."""
    frames = [[] for _ in range(160)]
    intervals = []
    for face, start, end, cue in ((3, 400, 1800, 0), (6, 5080, 5800, 2500)):
        intervals.append({"startMs": start, "endMs": end, "cueStartMs": cue, "track": face,
                          "faceId": face, "basis": "light-asd-local"})
        for index in range(start // 40, end // 40):
            frames[index] = [[face, .1 * face, .2, .15, .3, 1, face]]
    return {"visibleSpeechIntervals": intervals, "timeline": {"frames": frames, "stepMs": 40}}


class LocalVoiceAnchorTests(TestCase):
    def test_original_pair_preserves_short_bound_and_shares_subject(self):
        visual = evidence()
        turns = [{"startMs": 380, "endMs": 1600, "speaker": 1},
                 {"startMs": 5050, "endMs": 5820, "speaker": 1}]
        anchors = local_anchors(visual, turns)
        pair, = review_pairs(anchors, visual)
        reconcile_anchors(anchors, [pair], [{"cueMs": 2500, "face": 3, "similarity": .775}])
        report = {"videoEvidence": {"localVoiceAnchors": anchors}}
        first = speaking_evidence({"startMs": 0, "endMs": 2500}, report, "source")
        last = speaking_evidence({"startMs": 2500, "endMs": 6000}, report, "source")
        self.assertEqual(first["subjectId"], last["subjectId"])
        self.assertEqual(last["evidence"], [{"timeMs": 5080, "endMs": 5800, "faceId": 6,
                                             "profileFace": 3, "identitySimilarity": .775}])

    def test_concurrent_faces_and_freeze_do_not_establish_identity(self):
        visual = evidence()
        turns = [{"startMs": 380, "endMs": 1600, "speaker": 1},
                 {"startMs": 5050, "endMs": 5820, "speaker": 1}]
        anchors = local_anchors(visual, turns)
        for hits in visual["timeline"]["frames"][20:22]:
            hits.append([6, .6, .2, .15, .3, 0, 6])
        self.assertEqual(review_pairs(anchors, visual), [])
        visual["facialExpressions"] = {"staticFaceEvidence": {"exclusions": [
            {"startMs": 1200, "endMs": 1300, "faceId": 3, "strongAVSpeakerAnchorEligible": False}]}}
        self.assertEqual(set(local_anchors(visual, turns)), {"2500"})
