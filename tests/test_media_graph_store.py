"""Check focused graph navigation and original-file isolation."""

from pathlib import Path
from hashlib import file_digest
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_graph_store import MediaGraphStore, checked_source, identify, verified_digest, describe_source


def node(key, kind, start, end, label, **data):
    return {"id": key, "kind": kind, "startMs": start, "endMs": end, "label": label, "data": data}


class MediaGraphStoreTests(TestCase):
    def test_remounted_original_is_content_verified_once_and_replacement_stays_rejected(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mov"
            source.write_bytes(b"original")
            record = identify(source)
            record["signature"][0] += 1  # Same source exposed by a different mount/device namespace.
            verified_digest.cache_clear()
            with patch("voice_studio.graph.media_graph_store.file_digest", wraps=file_digest) as digest:
                self.assertEqual(checked_source(record), source.resolve())
                self.assertEqual(checked_source(record), source.resolve())
                self.assertEqual(digest.call_count, 1)
                self.assertEqual(describe_source(record, [])["verification"], "sha256")
            source.write_bytes(b"replaced")  # Same length must still invalidate the cache.
            with self.assertRaises(ValueError):
                checked_source(record)
            self.assertEqual(describe_source(record, [])["status"], "signature-changed")

    def test_timeline_is_chronological_with_repeatable_keyset_pages(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mov"
            source.write_bytes(b"sample")
            store = MediaGraphStore(Path(folder) / "graphs")
            nodes = [node("late", "frame", 1800, 1801, "Late"),
                     node("early", "shot", 0, 1000, "План", description="Текст DemoSign"),
                     node("mid", "utterance", 200, 1000, "Ну, это правда " + "д" * 300),
                     node("same", "utterance", 1800, 1900, "Повтор")]
            key = store.publish(identify(source), {"kind": "video", "durationMs": 2000},
                                {"nodes": nodes, "edges": [("mid", "in_shot", "early")]}, "ru", {})["snapshotId"]
            first = store.timeline(key, 0, 2000, limit=2)
            self.assertEqual(first["snapshotId"], key)
            second = store.timeline(key, 0, 2000, limit=2, after=first["nextAfter"])
            self.assertEqual([n["id"] for n in first["nodes"] + second["nodes"]], ["early", "mid", "late", "same"])
            self.assertIsNone(second["nextAfter"])
            self.assertEqual(store.search(key, "это правда")["nodes"][0]["id"], "mid")
            self.assertEqual(store.search(key, "это правда")["snapshotId"], key)
            self.assertEqual(store.node(key, "mid")["snapshotId"], key)
            self.assertGreater(len(store.node(key, "mid")["label"]), 240)
            self.assertEqual(store.search(key, "demosign", "shot")["nodes"][0]["id"], "early")
            self.assertEqual(store.node(key, "early")["neighbors"][0]["direction"], "in")

    def test_ui_category_pages_across_types_and_leaves_other_searches_available(self):
        from voice_studio.mcp.web_graph import read_graph

        with TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mov"
            source.write_bytes(b"sample")
            store = MediaGraphStore(Path(folder) / "graphs")
            nodes = [
                node("panel", "object_profile", 400, 800, "Search mobile",
                     interfaceEvidence={"type": "mobile-ui"}),
                node("desktop", "object_profile", 400, 800, "Desktop interface",
                     interfaceEvidence={"type": "ui"}),
                node("sign", "text_observation", 400, 401, "Search sign"),
                node("book", "object_profile", 400, 800, "Search book"),
                node("button", "ui_element", 400, 401, "Search button"),
                node("screen", "screen_region", 500, 501, "Search screen"),
            ]
            meta = store.publish(identify(source), {"kind": "video", "durationMs": 2000},
                                 {"nodes": nodes, "edges": []}, "en", {})
            key = meta["snapshotId"]
            self.assertEqual(meta["nodesByCategory"]["ui"], 4)
            self.assertEqual(meta["nodesByKind"]["object_profile"], 3)
            first = store.timeline(key, 0, 1000, "ui", limit=2)
            second = store.timeline(key, 0, 1000, "ui", limit=2, after=first["nextAfter"])
            self.assertEqual([row["id"] for row in first["nodes"] + second["nodes"]],
                             ["panel", "desktop", "button", "screen"])
            self.assertIsNone(second["nextAfter"])
            query = {"snapshot_id": key, "query": "Search", "kind": "ui", "limit": 1}
            page = read_graph(store.root, {"action": "search", "args": query})
            tail = store.search(key, "Search", "ui", after=page["nextAfter"])
            self.assertEqual([row["id"] for row in page["nodes"] + tail["nodes"]], ["panel", "button", "screen"])
            self.assertEqual([row["id"] for row in store.search(key, "Search", "text_observation")["nodes"]], ["sign"])
            self.assertEqual([row["id"] for row in store.search(key, "Search", "object_profile")["nodes"]],
                             ["panel", "book"])

    def test_scene_reuse_preserves_verified_layout_without_duplicate_mobile_object(self):
        from voice_studio.graph.media_graph_model import build_graph

        with TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mp4"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            panel = {"box": [.65, 0, .35, 1], "basis": "frame-layout", "panelType": "mobile-ui",
                     "elements": [], "connections": []}
            alias = {"label": "tv", "box": [.66, .02, .33, .95], "score": .7}
            report = {"language": "en", "sceneEvidence": {"status": "ok", "frames": [
                {"timeMs": 300, "panelChecked": True, "objects": [], "text": []},
                {"timeMs": 400, "panelChecked": True, "objects": [alias], "text": []},
                {"timeMs": 500, "panelChecked": True, "objects": [alias], "text": [], "interfaces": [panel]},
            ]}, "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}], "durationMs": 1000}}}
            original = identify(source)
            graph = build_graph(report, kind="video", duration_ms=1000)
            key = store.publish(original, {"kind": "video", "durationMs": 1000}, graph, "en", {})["snapshotId"]
            scene, _ = store.scene_evidence(key, original["sha256"], "en")
            empty, observed, frame = scene["frames"]
            self.assertEqual(empty, {"timeMs": 300, "panelChecked": True, "objects": [], "text": []})
            self.assertTrue(observed["panelChecked"])
            self.assertEqual(frame["interfaces"], [panel])
            self.assertEqual(frame["objects"], [alias])
            rebuilt = build_graph({**report, "sceneEvidence": scene}, kind="video", duration_ms=1000)
            self.assertEqual([(row["label"], row["startMs"]) for row in rebuilt["nodes"]
                              if row["kind"] == "object_observation"], [("tv", 400), ("mobile-ui", 500)])
            self.assertEqual([edge for edge in rebuilt["edges"] if edge[1] == "alias_of"],
                             [edge for edge in graph["edges"] if edge[1] == "alias_of"])

    def test_original_change_and_absent_face_geometry_are_distinguished(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mov"
            source.write_bytes(b"sample")
            store = MediaGraphStore(Path(folder) / "graphs")
            item = node("sign", "text_observation", 250, 251, "DemoSign", box=[.1, .2, .3, .4])
            original = identify(source)
            key = store.publish(original, {"kind": "video", "durationMs": 2000},
                                {"nodes": [item], "edges": [], "faceSampleMs": None}, "ru", {})["snapshotId"]
            _, stamp, overlay = store.frame_context(key, node_id="sign")
            self.assertEqual(stamp, 250)
            self.assertEqual([box["label"] for box in overlay["boxes"]], ["DemoSign"])
            self.assertEqual(store.frame_context(key, 1300)[2]["boxes"], [])
            source.write_bytes(b"replaced!")
            with self.assertRaises(ValueError):
                checked_source(original)


    def test_touching_scan_windows_do_not_invent_a_short_shot(self):
        from voice_studio.graph.media_graph_model import build_graph

        report = {"coverage": {"shots": [{"startMs": 0, "endMs": 1000, "status": "ok"},
                                          {"startMs": 1000, "endMs": 4000, "status": "ok"}]},
                  "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}, {"startMs": 2800}]}}}
        graph = build_graph(report, kind="video", duration_ms=4000)
        shots = [n for n in graph["nodes"] if n["kind"] == "shot"]
        self.assertEqual([(n["startMs"], n["endMs"]) for n in shots], [(0, 2800), (2800, 4000)])

    def test_technical_caption_uses_only_its_own_overlap_and_final_speaker(self):
        from voice_studio.graph.media_graph_model import build_graph

        with TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mov"
            source.write_bytes(b"sample")
            store = MediaGraphStore(Path(folder) / "graphs")
            report = {"language": "en", "speakerProfileIndex": [{"number": 2, "id": "right"}],
                      "segments": [
                          {"startMs": 0, "endMs": 900, "text": "Mixed reply", "speaker": 1, "overlap": True,
                           "overlaps": [{"startMs": 300, "endMs": 800, "speakers": [1, 2]}]},
                          {"startMs": 400, "endMs": 700, "text": "Assigned reply", "speakerNumber": 2,
                           "overlap": True, "speakerProfileId": "stale"},
                      ]}
            graph = build_graph(report, kind="video", duration_ms=1000)
            key = store.publish(identify(source), {"kind": "video", "durationMs": 1000}, graph, "en", {})["snapshotId"]
            self.assertEqual(store.frame_context(key, 350)[2]["caption"], "Overlapping speech: Mixed reply")
            self.assertEqual(store.frame_context(key, 500)[2]["caption"], "Speaker 2: Assigned reply")
            self.assertEqual(store.node(key, "speaker:2")["data"]["profileId"], "right")
