"""Verify UI inference without changing the original detector evidence."""

from unittest import TestCase

from voice_studio.graph.media_graph_model import box_coordinates, build_graph, object_layers
from voice_studio.graph.object_screens import screen_evidence


class ObjectScreenTests(TestCase):
    def test_monitor_profile_keeps_detector_class_and_frame_score(self):
        box = [.10, .10, .80, .70]
        report = {"durationMs": 2000, "language": "en",
                  "shotColors": [{"startMs": 0, "average": "#ffffff", "sampleCount": 2}], "sceneEvidence": {"frames": [{
            "timeMs": 900, "shot": 0, "objects": [{"label": "refrigerator", "score": .411, "box": box}],
            "text": [{"text": line, "box": [.20, .21 + index * .09, .20, .05]}
                     for index, line in enumerate(("Picture Effects", "Behind Text", "Selection Pane"))],
        }]}}
        layers = object_layers(report)
        profile, = layers["objectProfiles"]
        clip, = layers["objectTimeline"]
        self.assertEqual((profile["label"], profile["interfaceEvidence"]["type"]), ("monitor", "ui"))
        self.assertEqual(profile["labels"][0]["label"], "refrigerator")
        self.assertEqual((clip["label"], clip["representative"]["score"]), ("refrigerator", .411))

    def test_phone_with_measured_fields_gets_mobile_ui(self):
        box = [.64, .05, .34, .88]
        frame = {"interfaces": [{"box": box, "deviceLabel": "cell phone", "elements": [
            {"text": line, "role": "button_candidate", "box": [.67, .20 + index * .12, .25, .05]}
            for index, line in enumerate(("Search", "Images", "Open results"))]}]}
        result = screen_evidence("cell phone", box, frame, box_coordinates)
        self.assertEqual(result["type"], "mobile-ui")
        self.assertNotIn("displayLabel", result)

    def test_book_or_part_of_ui_outside_box_does_not_become_monitor(self):
        frame = {"text": [{"text": line, "box": [.65, .20 + index * .12, .30, .05]}
                          for index, line in enumerate(("Picture Effects", "Behind Text", "Selection Pane"))]}
        self.assertIsNone(screen_evidence("book", [.02, .02, .45, .9], frame, box_coordinates))
        self.assertIsNone(screen_evidence("cup", [.55, .02, .45, .9], frame, box_coordinates))

    def test_a_broader_layout_cannot_lend_outside_controls_or_device_type(self):
        box = [.10, .10, .40, .80]
        frame = {"text": [{"text": line, "box": [.2, .2 + i * .12, .19, .05]}
                          for i, line in enumerate(("Table of contents", "Introduction", "Acknowledgements"))],
                 "interfaces": [{"box": [.10, .10, .80, .80], "deviceLabel": "cell phone", "elements": [
                     {"text": line, "role": "button_candidate", "box": [.65, .2 + i * .12, .14, .06]}
                     for i, line in enumerate(("Search", "Next page", "More"))]}]}
        self.assertIsNone(screen_evidence("book", box, frame, box_coordinates))
        self.assertIsNone(screen_evidence("tv", box, frame, box_coordinates))


    def test_scroll_is_linked_to_the_panel_instead_of_a_phone_or_another_shot(self):
        panel = {"box": [.63, 0, .37, 1], "basis": "frame-layout", "panelType": "mobile-ui",
                 "elements": [], "connections": []}
        url = "/api/scene-scroll-preview/" + "a" * 32 + "/0.png"
        phone = {"label": "cell phone", "score": .7, "box": [.21, .61, .1, .2]}
        scene = {"status": "ok", "shotStartsMs": [2000],
                 "frames": [{"timeMs": stamp, "objects": [phone], "interfaces": [panel]}
                            for stamp in (750, 1500, 2200, 2700)],
                 "interfaceScrolls": [{"previewUrl": url, "startMs": 1750, "endMs": 1900,
                                       "viewports": [{"timeMs": 1860, "box": [.65, .13, .33, .74]}]}]}
        layers = object_layers({"durationMs": 3000, "sceneEvidence": scene, "language": "en"})
        profiles = layers["objectProfiles"]
        annotated, = [profile for profile in profiles if profile.get("scrollPreviews")]
        self.assertEqual(annotated["label"], "mobile-ui")
        self.assertEqual(annotated["scrollPreviews"][0]["previewUrl"], url)
        self.assertIsNone(annotated["labels"][0]["score"])
        self.assertEqual(annotated["appearances"][0]["startMs"], 750)


    def test_verified_panel_hides_tv_aliases_on_the_same_shot_and_keeps_the_handheld_phone(self):
        panel = {"box": [.63, 0, .37, 1], "basis": "frame-layout", "panelType": "mobile-ui",
                 "elements": [], "connections": []}
        television = {"label": "tv", "box": [.65, .03, .31, .92], "score": .6}
        handheld = {"label": "cell phone", "box": [.28, .55, .11, .25], "score": .8}
        scene = {"status": "ok", "shotStartsMs": [1000], "frames": [
            {"timeMs": 900, "objects": [television]},
            {"timeMs": 1080, "objects": [television, handheld]},
            {"timeMs": 1200, "panelChecked": True, "objects": [television, handheld], "interfaces": [panel]},
            {"timeMs": 1600, "panelChecked": True, "objects": [television]},
        ]}
        layers = object_layers({"durationMs": 2000, "sceneEvidence": scene})
        clips = layers["objectTimeline"]
        self.assertEqual([row["startMs"] for row in clips if row["label"] == "tv"], [900, 1600])
        self.assertEqual([row["sampleCount"] for row in clips if row["label"] == "cell phone"], [2])
        self.assertEqual([row["label"] for row in clips if row["startMs"] == 1200], ["mobile-ui"])


    def test_tv_tracks_cannot_bridge_a_positive_panel_even_across_detection_passes(self):
        panel = {"box": [.63, 0, .37, 1], "basis": "frame-layout", "panelType": "mobile-ui",
                 "elements": [], "connections": []}
        television = {"label": "tv", "box": [.65, .03, .31, .92], "score": .6}
        other = {"label": "tv", "box": [.08, .01, .30, .52], "score": .7}
        shifted = {**television, "box": [.55, .03, .31, .92]}
        for before, later in ((television, television), (shifted, television), (television, shifted)):
            for after in (later, {**later, "detailBasis": "small-object-region"}):
                with self.subTest(before=before, after=after):
                    scene = {"status": "ok", "shotStartsMs": [], "frames": [
                        {"timeMs": 100, "panelChecked": True, "objects": [before, other]},
                        {"timeMs": 800, "panelChecked": True, "objects": [television, other], "interfaces": [panel]},
                        {"timeMs": 1500, "panelChecked": True, "objects": [after, other]},
                    ]}
                    rows = object_layers({"durationMs": 2000, "sceneEvidence": scene})["objectTimeline"]
                    right = [(row["startMs"], row["endMs"]) for row in rows if row["label"] == "tv"
                             and row["representative"]["box"][0] > .5]
                    self.assertEqual(right, [(100, 101), (1500, 1501)])
                    self.assertEqual(sum(row["label"] == "tv" and row["sampleCount"] == 3 for row in rows), 1)


    def test_frame_tree_keeps_measured_content_and_records_tv_as_an_alias(self):
        panel = {"box": [.63, 0, .37, 1], "basis": "frame-layout", "panelType": "mobile-ui", "connections": [],
                 "elements": [{"role": "button_candidate", "text": "Save", "basis": "ocr", "box": [.69, .70, .25, .1]},
                              {"role": "text_line", "text": "Save", "basis": "ocr", "box": [.74, .72, .09, .04]}]}
        report = {"durationMs": 1500, "language": "en", "sceneEvidence": {
            "status": "ok", "shotStartsMs": [], "frames": [{"timeMs": 500, "interfaces": [panel], "objects": [
                {"label": "tv", "score": .7, "box": [.65, .03, .31, .92]},
                {"label": "book", "score": .7, "box": [.71, .25, .14, .23]},
                {"label": "cell phone", "score": .8, "box": [.28, .55, .11, .25]},
            ]}]}}
        graph = build_graph(report, kind="video", duration_ms=1500)
        aliases = [node for node in graph["nodes"] if node["kind"] == "object_alias"]
        alias, = aliases
        nodes = {node["label"]: node for node in graph["nodes"] if node["kind"] == "object_observation"}
        renderer = nodes["mobile-ui"]["id"]
        parents = {child: parent for child, relation, parent in graph["edges"] if relation == "contained_in"}
        self.assertEqual(parents[alias["id"]], renderer)
        self.assertIn((alias["id"], "alias_of", renderer), graph["edges"])
        self.assertEqual(parents["screen_region:500:0"], renderer)
        self.assertEqual(parents["ui_element:500:0:0"], "screen_region:500:0")
        self.assertEqual(parents["ui_element:500:0:1"], "ui_element:500:0:0")
        self.assertEqual(parents[nodes["book"]["id"]], "screen_region:500:0")
        self.assertNotIn(nodes["cell phone"]["id"], parents)
        self.assertNotIn("tv", {clip["label"] for clip in object_layers(report)["objectTimeline"]})


    def test_tv_alias_reuses_nearby_evidence_without_crossing_frames_in_the_tree(self):
        panel = {"box": [.63, 0, .37, 1], "basis": "frame-layout", "panelType": "mobile-ui",
                 "elements": [], "connections": []}
        television = {"label": "tv", "score": .7, "box": [.65, .03, .31, .92]}
        scene = {"status": "ok", "shotStartsMs": [], "frames": [
            {"timeMs": 420, "objects": [television]},
            {"timeMs": 500, "interfaces": [panel]},
        ]}
        graph = build_graph({"sceneEvidence": scene}, kind="video", duration_ms=1500)
        alias, = [node for node in graph["nodes"] if node["kind"] == "object_alias"]
        panel_node, = [node for node in graph["nodes"] if node["kind"] == "object_observation"]
        self.assertIn((alias["id"], "alias_of", panel_node["id"]), graph["edges"])
        self.assertFalse(any(source == alias["id"] and relation == "contained_in"
                             for source, relation, _ in graph["edges"]))
