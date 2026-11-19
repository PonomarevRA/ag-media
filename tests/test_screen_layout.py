"""Verify visual interface evidence and its persistence on the source clock."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, identify

HAS_VISION = find_spec("cv2") is not None and find_spec("numpy") is not None
FULL_FRAME = {"box": [0, 0, 1, 1], "basis": "frame-layout"}


def diagram_fixture():
    """Draw a simple two-block flowchart and return deterministic OCR boxes."""
    import cv2
    import numpy as np

    image = np.full((400, 640, 3), 245, dtype=np.uint8)
    ink = (25, 25, 25)
    cv2.rectangle(image, (64, 155), (238, 232), ink, 3)
    cv2.rectangle(image, (386, 155), (560, 232), ink, 3)
    cv2.line(image, (238, 194), (386, 194), ink, 3)
    lines = [
        {"text": "Input", "box": [121/640, 183/400, 52/640, 18/400]},
        {"text": "Output", "box": [439/640, 183/400, 67/640, 18/400]},
    ]
    return image, lines


class ScreenGraphTests(TestCase):
    def test_interface_graph_retains_layout_and_exact_target_across_reuse(self):
        region = {
            "box": [.1, .1, .8, .8],
            "basis": "device-crop",
            "deviceLabel": "tv",
            "elements": [
                {"role": "diagram_block", "text": "Input", "box": [.2, .2, .2, .2], "basis": "outline"},
                {"role": "diagram_block", "text": "Output", "box": [.6, .2, .2, .2], "basis": "outline"},
            ],
            "connections": [{"from": 0, "to": 1, "segment": [[.4, .3], [.6, .3]]}],
        }
        coverage = {"scene": [{"startMs": 0, "endMs": 900, "status": "sampled"}],
                    "shots": [{"startMs": 0, "endMs": 900, "status": "sampled"}]}
        report = {"language": "en", "durationMs": 900, "coverage": coverage,
                  "sceneEvidence": {"status": "ok", "frames": [{"timeMs": 240, "interfaces": [region]}]}}
        graph = build_graph(report, kind="video", duration_ms=900)
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = graph["edges"]
        screen_id = "screen_region:240:0"
        input_id, output_id = "ui_element:240:0:0", "ui_element:240:0:1"
        connector_id = "ui_connection:240:0:0"
        self.assertEqual(nodes[input_id]["label"], "Input")
        self.assertEqual(nodes[output_id]["data"]["box"], [.6, .2, .2, .2])
        self.assertEqual(nodes[connector_id]["data"]["segment"], [[.4, .3], [.6, .3]])
        self.assertIn((output_id, "in_screen", screen_id), edges)
        self.assertIn((connector_id, "endpoint_a", input_id), edges)
        self.assertIn((connector_id, "endpoint_b", output_id), edges)
        self.assertIn((input_id, "in_shot", "shot:0"), edges)
        self.assertTrue(any(source == input_id and relation == "at_frame" for source, relation, target in edges))

        with TemporaryDirectory() as folder:
            source = Path(folder) / "fixture.mp4"
            source.write_bytes(b"stable source identity")
            original = identify(source)
            store = MediaGraphStore(Path(folder) / "graphs")
            snapshot = store.publish(original, {"kind": "video", "durationMs": 900}, graph, "en", {})["snapshotId"]
            _, stamp, context = store.frame_context(snapshot, node_id=input_id)
            self.assertEqual(stamp, 240)
            self.assertEqual(context["targetObservationId"], input_id)
            self.assertTrue(any(mark["kind"] == "target" and mark["box"] == [.2, .2, .2, .2]
                                for mark in context["boxes"]))
            frozen, _ = store.scene_evidence(snapshot, original["sha256"], "en")
            (frame,) = frozen["frames"]
            self.assertEqual(frame["interfaces"], [region])

    def test_invalid_boxes_and_dangling_connectors_do_not_enter_graph(self):
        region = {
            **FULL_FRAME,
            "elements": [
                {"role": "diagram_block", "text": "Good", "box": [.1, .1, .2, .2], "basis": "outline"},
                {"role": "diagram_block", "text": "Outside", "box": [.8, .8, .5, .5], "basis": "outline"},
            ],
            "connections": [
                {"from": 0, "to": 1, "segment": [[.3, .2], [.8, .2]]},
                {"from": 0, "to": 7, "segment": [[.3, .2], [.8, .2]]},
            ],
        }
        graph = build_graph({"sceneEvidence": {"frames": [{"timeMs": 20, "interfaces": [region]}]}},
                            kind="video", duration_ms=100)
        self.assertFalse(any(node["kind"] == "ui_connection" for node in graph["nodes"]))
        nodes = [node for node in graph["nodes"] if node["kind"] == "ui_element"]
        self.assertEqual([node["label"] for node in nodes], ["Good"])


@skipUnless(HAS_VISION, "OpenCV is available in the media vision runtime")
class ScreenVisionTests(TestCase):
    def test_outlined_diagram_blocks_connect_without_inferred_direction(self):
        from voice_studio.vision.screen_layout import analyze_layout

        image, lines = diagram_fixture()
        result = analyze_layout(image, FULL_FRAME, lines)
        self.assertIsNotNone(result)
        self.assertTrue(result["connections"])
        endpoints = [(result["elements"][row["from"]], result["elements"][row["to"]])
                     for row in result["connections"]]
        self.assertTrue(any({left.get("text"), right.get("text")} == {"Input", "Output"}
                            for left, right in endpoints))

    def test_device_crop_distinguishes_control_candidates_and_keeps_frame_coordinates(self):
        import cv2
        import numpy as np

        from voice_studio.vision.screen_layout import analyze_layout

        image = np.full((440, 800, 3), 55, dtype=np.uint8)
        cv2.rectangle(image, (280, 44), (760, 396), (245, 245, 245), -1)
        ink = (25, 25, 25)
        cv2.rectangle(image, (320, 104), (710, 154), ink, 3)
        cv2.rectangle(image, (441, 214), (589, 266), ink, 3)
        region = {"box": [.35, .1, .6, .8], "basis": "device-crop", "deviceLabel": "cell phone"}
        lines = [
            {"text": "Search...", "box": [337/800, 119/440, 75/800, 19/440]},
            {"text": "Upload", "box": [481/800, 230/440, 69/800, 19/440]},
            {"text": "Subtitle elsewhere", "box": [.02, .86, .24, .06]},
        ]
        result = analyze_layout(image, region, lines)
        self.assertIsNotNone(result)
        elements = {item.get("text"): item for item in result["elements"]}
        self.assertEqual(elements["Search..."]["role"], "field_candidate")
        self.assertEqual(elements["Upload"]["role"], "button_candidate")
        self.assertNotIn("Subtitle elsewhere", elements)
        self.assertAlmostEqual(elements["Upload"]["box"][0], 441/800, delta=.01)
        self.assertEqual(result["connections"], [])

    def test_unlabelled_connected_thumbnail_grid_is_not_a_diagram(self):
        import cv2
        import numpy as np

        from voice_studio.vision.screen_layout import analyze_layout

        image = np.full((400, 640, 3), 245, dtype=np.uint8)
        ink = (25, 25, 25)
        for x in (64, 386):
            cv2.rectangle(image, (x, 155), (x + 174, 232), ink, 3)
        cv2.line(image, (238, 194), (386, 194), ink, 3)
        self.assertIsNone(analyze_layout(image, FULL_FRAME, []))

    def test_scene_summary_admits_diagram_with_two_labels(self):
        import cv2

        from voice_studio.vision.screen_analysis import ScreenAnalysis

        image, lines = diagram_fixture()
        with TemporaryDirectory() as folder:
            source = Path(folder) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(source), image))
            with patch("voice_studio.vision.screen_analysis.recognize", return_value=({}, {})):
                frames, crop_count, _ = ScreenAnalysis().summarize({240: source}, {240: lines})
        self.assertEqual(crop_count, 0)
        (layout,) = frames[240]
        endpoints = [(layout["elements"][row["from"]], layout["elements"][row["to"]])
                     for row in layout["connections"]]
        self.assertTrue(any({a.get("text"), b.get("text")} == {"Input", "Output"} for a, b in endpoints))

    def test_tiled_screenshot_retains_whole_frame_layout(self):
        import cv2
        import numpy as np

        from voice_studio.vision.screen_analysis import screenshot_layout

        image = np.full((480, 960, 3), 245, dtype=np.uint8)
        line = {"text": "Search documents", "box": [.2, .25, .14, .04]}
        with TemporaryDirectory() as folder:
            source = Path(folder) / "screenshot.png"
            self.assertTrue(cv2.imwrite(str(source), image))
            def recognize(jobs, **options):
                if options.get("page_mode") == 7:
                    return {}, {}
                self.assertEqual(len(jobs), 5)
                _, staged, box, width, height = jobs[0]
                self.assertEqual(staged.name, "oriented.png")
                self.assertEqual((box, width, height), ((0, 0, 1, 1), 960, 480))
                np.testing.assert_array_equal(cv2.imread(str(staged)), image)
                return {0: [line]}, {"tesseract": "ok"}

            with patch("voice_studio.vision.screen_analysis.recognize", side_effect=recognize) as ocr:
                result = screenshot_layout(source)
        self.assertEqual(result["imageSize"], [960, 480])
        self.assertEqual(result["interface"]["box"], [0, 0, 1, 1])
        self.assertEqual(result["interface"]["basis"], "screenshot")
        self.assertEqual(result["interface"]["elements"][0]["text"], "Search documents")
        self.assertEqual(ocr.call_args_list[1].kwargs["page_mode"], 7)

    def test_plain_subtitles_do_not_make_the_frame_an_interface(self):
        import cv2
        import numpy as np

        from voice_studio.vision.screen_layout import analyze_layout

        image = np.full((400, 640, 3), 72, dtype=np.uint8)
        cv2.rectangle(image, (90, 311), (550, 382), (0, 0, 0), -1)
        cv2.putText(image, "Just a spoken caption", (111, 359), cv2.FONT_HERSHEY_SIMPLEX, .9,
                    (255, 255, 255), 2)
        lines = [{"text": "Just a spoken caption", "box": [.16, .83, .59, .07]}]
        self.assertIsNone(analyze_layout(image, FULL_FRAME, lines))
