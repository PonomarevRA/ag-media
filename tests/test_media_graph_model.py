"""Verify speaker, shot and observation links against the shared source clock."""

from unittest import TestCase

from voice_studio.graph.media_graph_model import build_graph


class MediaGraphModelTests(TestCase):
    def test_unbound_visible_face_keeps_its_own_profile_attributes_without_speech_identity(self):
        report = {"videoEvidence": {"editorTimeline": {
            "faceAttributes": {"17": {"age": 31, "ageSource": "measured-model", "ageSamples": 2}},
            "clothingByFace": {"17": {"swatch": "#123456", "source": "mask"}},
            "facePhotos": {"18": "not-stored-in-graph"}}}}
        graph = build_graph(report, kind="video", duration_ms=1000)
        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(nodes["face:17"]["data"]["appearance"]["age"], 31)
        self.assertEqual(nodes["face:17"]["data"]["clothing"]["swatch"], "#123456")
        self.assertNotIn("appearance", nodes["face:18"]["data"])
        self.assertFalse(any(node["kind"] == "speaker" for node in graph["nodes"]))
        self.assertNotIn("not-stored-in-graph", str(graph))

    def test_optional_people_counts_keep_valid_estimates_when_import_fields_are_malformed(self):
        for malformed in ([1], {"samples": 1}, {"samples": [None]}):
            with self.subTest(malformed=malformed):
                report = {"videoEvidence": {"editorTimeline": {"personPilot": malformed}}}
                graph = build_graph(report, kind="video", duration_ms=800)
                self.assertFalse(any(row["kind"] == "person_count" for row in graph["nodes"]))
        samples = [{"timeMs": 300, "count": 0, "people": 1}]
        report = {"videoEvidence": {"editorTimeline": {"personPilot": {"samples": samples}}}}
        graph = build_graph(report, kind="video", duration_ms=800)
        node, = (row for row in graph["nodes"] if row["kind"] == "person_count")
        self.assertEqual(node["data"], {"count": 0, "source": "nanodet", "people": []})

    def test_video_links_exact_speaker_profile_and_visual_evidence(self):
        rows = [[[1, .1, .2, .3, .4, 1, 8], [2, .7, .2, .2, .3, 0, 9]]] * 20
        report = {
            "language": "en", "durationMs": 800,
            "speakerProfileIndex": [{"number": 2, "id": "real-profile", "faceId": 8, "faceBasis": "editorial"}],
            "segments": [{"startMs": 100, "endMs": 600, "text": "This is true.", "speaker": 1, "speakerNumber": 2}],
            "videoEvidence": {
                "timeline": {"stepMs": 40, "frames": rows},
                "editorTimeline": {"durationMs": 800, "shots": [{"startMs": 0}, {"startMs": 400}]},
                "faceSpeakers": {"8": {"speaker": 2, "basis": "editorial"}, "9": {"speaker": 1, "basis": "visible-only"}},
                "phrases": {"100": {"activity": {"state": "visible-active", "track": 1}}},
            },
            "sceneEvidence": {"status": "ok", "frames": [
                {"timeMs": 300, "objects": [{"label": "cup", "score": .9, "box": [.1, .2, .3, .4]}],
                 "text": [{"text": "DemoSign", "box": [.5, .6, .1, .1]}]},
                {"timeMs": 500, "objects": [{"label": "cup", "score": .8, "box": [.1, .2, .3, .4]}]},
            ]},
        }
        result = build_graph(report, kind="video", duration_ms=800)
        nodes = {row["id"]: row for row in result["nodes"]}
        edges = result["edges"]
        self.assertEqual(nodes["speaker:2"]["data"]["profileId"], "real-profile")
        self.assertIn(("utterance:100:0", "spoken_by", "speaker:2"), edges)
        self.assertIn(("speaker:2", "profile_face", "face:8"), edges)
        self.assertNotIn(("speaker:1", "profile_face", "face:9"), edges)
        self.assertIn(("utterance:100:0", "in_shot", "shot:0"), edges)
        self.assertIn(("utterance:100:0", "in_shot", "shot:400"), edges)
        self.assertIn("DemoSign", nodes["shot:0"]["data"]["description"])
        self.assertEqual(len([row for row in nodes.values() if row["kind"] == "object"]), 1)
        self.assertEqual(len([row for row in nodes.values() if row["kind"] == "object_observation"]), 2)
        self.assertEqual(result["faceSampleMs"], 40)
        sample = dict(result["faceFrames"])[120]
        self.assertTrue(sample[0]["active"])
        self.assertIsNone(sample[1]["speakerId"])
        self.assertFalse(dict(result["faceFrames"])[80][0]["active"])
        self.assertTrue(all(left in nodes and right in nodes for left, _, right in edges))

    def test_mixed_audio_cue_links_only_explicit_timed_participants(self):
        report = {"durationMs": 4000, "segments": [
            {"startMs": 300, "endMs": 2100, "text": "Two voices.", "speaker": 1, "overlap": True,
             "overlaps": [{"startMs": 300, "endMs": 2100, "speakers": [1, 2]}]},
            {"startMs": 2300, "endMs": 3300, "text": "Unresolved.", "speaker": 2, "overlap": True},
            {"startMs": 3400, "endMs": 4000, "text": "Reviewed.", "speaker": 1, "speakerNumber": 2, "overlap": True},
        ]}
        result = build_graph(report, kind="audio", duration_ms=4000)
        edges = result["edges"]
        self.assertIn(("overlap:300:0:0", "participant", "speaker:2"), edges)
        self.assertIn(("overlap:300:0:0", "overlap_of", "utterance:300:0"), edges)
        self.assertNotIn(("utterance:300:0", "spoken_by", "speaker:1"), edges)
        self.assertNotIn(("utterance:2300:0", "spoken_by", "speaker:2"), edges)
        self.assertIn(("utterance:3400:0", "spoken_by", "speaker:2"), edges)
        self.assertEqual(result["faceFrames"], [])

    def test_long_source_has_no_shot_bridge_across_unanalysed_gaps(self):
        hour = 3_600_000
        report = {
            "coverage": {"shots": [
                {"startMs": 0, "endMs": 1000, "status": "sampled"},
                {"startMs": hour, "endMs": hour + 1000, "status": "sampled"},
            ]},
            "videoEvidence": {"editorTimeline": {"shots": [
                {"startMs": 0}, {"startMs": 500}, {"startMs": hour}, {"startMs": hour + 500},
            ]}},
            "segments": [{"startMs": hour - 10, "endMs": hour + 200, "text": "At one hour."}],
        }
        result = build_graph(report, kind="video", duration_ms=2 * hour)
        shots = [(row["startMs"], row["endMs"]) for row in result["nodes"] if row["kind"] == "shot"]
        self.assertEqual(shots, [(0, 500), (500, 1000), (hour, hour + 500), (hour + 500, hour + 1000)])
        self.assertNotIn(("shot:500", "next", f"shot:{hour}"), result["edges"])
        key = f"utterance:{hour - 10}:0"
        self.assertIn((key, "in_shot", f"shot:{hour}"), result["edges"])
        self.assertNotIn((key, "in_shot", "shot:500"), result["edges"])

    def test_audio_catalog_preserves_secondary_streams_and_local_window_labels(self):
        report = {
            "language": "en", "durationMs": 701000, "speakerScope": "window",
            "streams": [{"kind": "audio", "index": 1, "channels": 2}, {"kind": "audio", "index": 2}],
            "segments": [{"startMs": 700000, "endMs": 701000, "text": "Another chapter.", "speaker": 1000001,
                          "speakerWindowStartMs": 600000, "speakerLocalNumber": 1}],
        }
        result = build_graph(report, kind="audio", duration_ms=701000)
        nodes = {row["id"]: row for row in result["nodes"]}
        self.assertEqual(nodes["audio_track:2"]["data"], {"index": 2})
        self.assertIn(("utterance:700000:0", "on_track", "audio_track:1"), result["edges"])
        self.assertEqual(nodes["speaker:1000001"]["label"], "Speaker 1 · 00:10:00")
        self.assertEqual(result["speakerScope"], "window")
