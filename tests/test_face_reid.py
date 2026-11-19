"""Check stable visible numbers and conflicting audiovisual evidence."""

from unittest import TestCase

from voice_studio.vision.face_reid import distinct_faces, group_tracks, unit_vector
from voice_studio.vision.face_speakers import bind_faces


def signatures(*vectors):
    return {i: [unit_vector(vector)] * 2 for i, vector in enumerate(vectors, 1)}


class FaceReidentificationTests(TestCase):
    def test_returning_face_keeps_number_but_concurrent_track_cannot_merge(self):
        tracks = [[(0, None), (8, None)], [(2, None), (5, None)], [(20, None), (30, None)]]
        aliases, matches = group_tracks(tracks, signatures((1, 0), (.8, .6), (1, 0)))
        self.assertEqual(aliases, {1: 1, 2: 2, 3: 1})
        self.assertEqual(matches[0]["track"], 3)

    def test_ambiguous_face_does_not_join_either_existing_identity(self):
        tracks = [[(0, None), (8, None)], [(2, None), (5, None)], [(20, None), (30, None)]]
        aliases, matches = group_tracks(tracks, signatures((1, 0), (0, 1), (1, 1)))
        self.assertEqual(aliases, {1: 1, 2: 2, 3: 3})
        self.assertEqual(matches, [])

    def test_confirmed_editorial_mapping_and_silent_face_numbers_remain_separate(self):
        hit_active = [11, 0, 0, .5, .5, 1, 2]
        hit_silent = [3, .5, 0, .5, .5, 0, 3]
        result = {
            "segments": [
                {"startMs": 0, "endMs": 800, "speaker": 1, "speakerNumber": 2, "visualFace": 2},
                {"startMs": 800, "endMs": 1600, "speaker": 1, "visualFace": 2},
                {"startMs": 1600, "endMs": 2400, "speaker": 1},
            ],
            "videoEvidence": {"timeline": {"stepMs": 40, "frames": [[hit_active, hit_silent]] * 40 + [[hit_silent]] * 20}},
        }
        bind_faces(result)
        visible = result["videoEvidence"]["faceSpeakers"]
        self.assertEqual(visible[2]["speaker"], 2)
        self.assertEqual(visible[3], {"speaker": 3, "basis": "visible-only"})
        self.assertEqual(result["segments"][1]["linkedSpeakerNumber"], 2)
        self.assertNotIn("linkedSpeakerNumber", result["segments"][2])
        self.assertEqual(result["segments"][1]["speaker"], 1)

    def test_one_raw_cluster_can_keep_two_distinct_confirmed_faces(self):
        aliases = {1: 3, 2: 5, 3: 3}
        conflicts = distinct_faces(aliases, signatures((1, 0), (0, 1), (1, 0)))
        self.assertEqual(conflicts, [[3, 5]])
        result = {
            "segments": [
                {"startMs": 0, "endMs": 900, "speaker": 1, "visualFace": 3},
                {"startMs": 900, "endMs": 1600, "speaker": 1, "visualFace": 5},
                {"startMs": 1600, "endMs": 2100, "speaker": 1},
            ],
            "videoEvidence": {"timeline": {"stepMs": 40, "frames": []},
                              "faceTracking": {"distinctFaces": conflicts}},
        }
        bind_faces(result)
        self.assertEqual([row.get("linkedSpeakerNumber") for row in result["segments"]], [1, 2, None])
        self.assertTrue(all(row["speaker"] == 1 for row in result["segments"]))

    def test_two_visible_people_split_even_when_one_confirmed_reply_is_short(self):
        """Keep separate speakers when disjoint face boxes coexist and ASR shares a label."""
        result = {
            "segments": [
                {"startMs": 0, "endMs": 800, "speaker": 1, "visualFace": 3},
                {"startMs": 800, "endMs": 1120, "speaker": 1, "visualFace": 5},
            ],
            "videoEvidence": {"timeline": {"stepMs": 40, "frames": [
                [[11, 0, 0, .3, .3, 1, 3], [12, .5, 0, .3, .3, 0, 5]]
            ] * 28}},
        }
        bind_faces(result)
        self.assertEqual([cue["linkedSpeakerNumber"] for cue in result["segments"]], [1, 2])

    def test_frame_vote_owner_keeps_unique_number_when_direct_faces_conflict(self):
        """The distinct-face split must account for owners without an explicit cue face."""
        result = {
            "segments": [
                {"startMs": 0, "endMs": 2000, "speaker": 1},
                {"startMs": 2000, "endMs": 2400, "speaker": 1, "visualFace": 2},
                {"startMs": 2400, "endMs": 2720, "speaker": 1, "visualFace": 3},
            ],
            "videoEvidence": {"timeline": {"stepMs": 40, "frames":
                [[[1, 0, 0, .3, .3, 1, 1]]] * 50 +
                [[[2, 0, 0, .3, .3, 1, 2], [3, .5, 0, .3, .3, 0, 3]]] * 18}},
        }
        bind_faces(result)
        bindings = result["videoEvidence"]["faceSpeakers"]
        self.assertEqual(bindings[1]["speaker"], 1)
        self.assertEqual(len({row["speaker"] for row in bindings.values()}), 3)
        self.assertNotEqual(result["segments"][2]["linkedSpeakerNumber"], 1)
        result["segments"][0]["visualFace"] = 1
        bind_faces(result)
        labels = [row["linkedSpeakerNumber"] for row in result["segments"]]
        self.assertEqual(labels, [1, 2, 3])

    def test_distinct_faces_with_competing_cluster_keep_established_labels(self):
        """A confirmed face can retain its strong speaker association on a misclustered cue."""
        result = {
            "segments": [
                {"startMs": 0, "endMs": 600, "speaker": 1, "visualFace": 2},
                {"startMs": 600, "endMs": 1100, "speaker": 1, "visualFace": 3},
                {"startMs": 1100, "endMs": 3100, "speaker": 2, "visualFace": 2},
            ],
            "videoEvidence": {"timeline": {"stepMs": 40, "frames":
                [[[2, 0, 0, .3, .3, 1, 2], [3, .5, 0, .3, .3, 0, 3]]] * 78}},
        }
        bind_faces(result)
        self.assertEqual([cue["linkedSpeakerNumber"] for cue in result["segments"]], [2, 1, 2])
