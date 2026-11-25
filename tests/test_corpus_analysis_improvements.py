"""Regressions from the source-backed dialogue, screen and object audit."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from voice_studio.graph.media_graph_model import build_graph, object_layers
from voice_studio.speech.dialogue_audit_runner import read_asr_json, retry_inventory


class WindowRecoveryTests(TestCase):
    def test_token_encoding_damage_preserves_segments_without_another_decode(self):
        """Only corruption confined to token strings can take the in-place fallback."""
        with TemporaryDirectory() as temp:
            output = Path(temp) / "asr.json"
            output.write_bytes(b'{"transcription":[{"text":"3.5 cm","tokens":[{"text":"\xff"}]}]}')
            result = read_asr_json(output, "-ojf")
            self.assertEqual(result, {"transcription": [{"text": "3.5 cm"}], "outputFormat": "segments-only"})
            output.write_bytes(b'{"transcription":[{"text":"\xff","tokens":[{"text":"\xff"}]}]}')
            with self.assertRaises(UnicodeDecodeError):
                read_asr_json(output, "-ojf")

    def test_one_failed_window_does_not_hide_later_result_or_elevate_loop(self):
        """Retry Turbo on a bad window and publish the later legitimate Small draft."""
        first = {"startMs": 0, "endMs": 6500, "speaker": 1}
        second = {"startMs": 6500, "endMs": 13000, "speaker": 1}
        inventory = {"candidates": [{
            "id": "candidate", "priority": "high", "reasons": [],
            "startMs": 0, "reviewWindows": [first, second], "evidence": {"cueIndices": [0]},
        }]}
        loop = {"transcription": [{"text": "again now " * 12, "offsets": {"from": 0, "to": 30000}}]}
        success = {"transcription": [{"text": "Set the width to 3.5 cm", "offsets": {"from": 30, "to": 1100}}]}
        runner = "voice_studio.speech.dialogue_audit_runner."
        with TemporaryDirectory() as temp, patch(runner + "crop_pcm", return_value=800), \
                patch(runner + "decode_local", side_effect=[loop, ValueError("decode failed"), success]) as decode:
            result = retry_inventory(inventory, None, Path(temp), [("small", "small"), ("turbo", "turbo")],
                                     "en", 2, 13000, lambda: None, lambda event: None)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["budget"]["turboWindows"], 1)
        self.assertEqual([row["status"] for row in result["windows"]], ["failed", "decoded"])
        self.assertEqual([row["model"] for row in result["suggestions"]], ["small"])
        self.assertEqual(result["suggestions"][0]["startMs"], 6530)
        self.assertEqual([call.args[3] for call in decode.call_args_list], ["small", "turbo", "small"])


class EvidenceProjectionTests(TestCase):
    def test_original_cue_and_audit_draft_keep_their_own_screen_witnesses(self):
        """Return two source clocks and bind the crop node without confirming an actual click."""
        report = {"language": "en", "segments": [{"startMs": 100, "endMs": 1000, "text": "Click Done"}],
                  "dialogueAudit": {"candidates": [{"id": "gap", "startMs": 2000, "endMs": 3000}],
                                    "repairs": {"suggestions": [{
                                        "candidateId": "gap", "startMs": 2200, "endMs": 2700,
                                        "text": "Set 3.5 cm", "acceptance": "review-required", "model": "turbo"}]}},
                  "sceneEvidence": {"frames": [
                      {"timeMs": 800, "interfaces": [{"basis": "frame-layout", "panelType": "ui",
                         "box": [.5, 0, .5, 1], "previousStateMs": 500, "detailBasis": "original-resolution",
                         "elements": [{"text": "Done", "role": "button_candidate", "basis": "ocr",
                                       "box": [.6, .2, .2, .1]}]}]},
                      {"timeMs": 2300, "text": [{"text": "Width 3,5 cm", "box": [.6, .3, .2, .1]}]},
                  ]}}
        graph = build_graph(report, kind="video", duration_ms=5000)
        nodes = {node["kind"]: node for node in graph["nodes"]}
        original, draft, screen = (nodes[kind] for kind in ("utterance", "dialogue_draft", "screen_region"))
        self.assertEqual(original["data"]["screenEvidence"][0]["actionCandidate"],
                         "spoken-click-with-screen-state-change")
        self.assertEqual(draft["data"]["screenEvidence"][0]["match"], "3.5 cm")
        self.assertEqual(draft["data"]["screenEvidence"][0]["timeMs"], 2300)
        self.assertEqual(draft["startMs"], 2200)
        self.assertTrue(screen["data"]["cropAvailable"])
        self.assertIn((draft["id"], "screen_evidence", nodes["text_observation"]["id"]), graph["edges"])

    def test_transport_label_conflict_keeps_one_identity_and_a_separate_simultaneous_car(self):
        """Repeated containment supports a profile, while another car remains independent."""
        frames = []
        for stamp in (100, 250, 400, 550):
            frames.append({"timeMs": stamp, "objects": [
                {"label": "airplane", "score": .59, "box": [.08, .16, .49, .58]},
                {"label": "car", "score": .50, "box": [.17, .4, .24, .22]},
                {"label": "car", "score": .54, "box": [.72, .32, .19, .24]},
            ]})
        report = {"durationMs": 2000, "sceneEvidence": {"frames": frames},
                  "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}]}}}
        profiles = object_layers(report)["objectProfiles"]
        self.assertEqual(len(profiles), 2)
        mixed, = [profile for profile in profiles if profile["hasLabelConflict"]]
        self.assertEqual(mixed["identityId"], mixed["id"])
        self.assertEqual({row["label"] for row in mixed["labels"]}, {"airplane", "car"})


    def test_local_parts_link_cross_cut_and_cross_label_without_changing_observations(self):
        """A measured local-parts witness can resolve a changed raw class across a cut."""
        first = {"label": "car", "score": .71, "box": [.1, .1, .6, .5]}
        second = {"label": "truck", "score": .7, "box": [.2, .2, .5, .4],
                  "objectPartMatches": [{"timeMs": 100, "label": "car", "box": first["box"],
                    "basis": "orb-local-parts", "inliers": 24, "inlierFraction": .9,
                    "sourceCoverage": .3, "targetCoverage": .3, "sourceCells": 5, "targetCells": 5}]}
        report = {"durationMs": 5000, "objectReconciliationEnabled": True,
                  "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}, {"startMs": 2000}]}},
                  "sceneEvidence": {"frames": [{"timeMs": 100, "objects": [first]},
                                               {"timeMs": 3100, "objects": [second]}]}}
        graph = build_graph(report, kind="video", duration_ms=5000)
        profile, = [row for row in graph["nodes"] if row["kind"] == "object_profile"]
        observations = [row for row in graph["nodes"] if row["kind"] == "object_observation"]
        self.assertEqual({row["data"]["identityId"] for row in observations}, {profile["id"]})
        self.assertEqual({row["data"]["rawLabel"] for row in observations}, {"car", "truck"})


@skipUnless(find_spec("cv2"), "optional vision runtime is required")
class RegionStateTests(TestCase):
    def test_stable_panel_schedules_once_until_content_changes(self):
        """Require repeated geometry and reuse the last reading on unchanged pixels."""
        import numpy as np
        from voice_studio.vision.screen_regions import ScreenRegionStates

        states = ScreenRegionStates()
        panel = {"box": [.5, 0, .5, 1], "candidateType": "mobile-ui"}
        image = np.full((200, 300, 3), 245, dtype=np.uint8)
        changed = image.copy()
        changed[48:86, 175:275] = 0
        for index, frame in enumerate([image, image, image, changed, changed, changed]):
            states.sample(frame, {"shot": 0, "timeMs": index * 500, "sourceFrame": index}, panel)
        first, second = states.plan()
        self.assertEqual(first["timeMs"], 500)
        self.assertEqual(first["lastObservedMs"], 1000)
        self.assertEqual(second["previousStateMs"], 500)
        self.assertEqual(second["lastObservedMs"], 2500)
