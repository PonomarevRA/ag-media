"""Verify interval evidence and review counts across both media-graph index shapes."""

from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.graph.media_graph_interval import summarize_interval
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


def fact(identity, kind, **data):
    """Make an in-window synthetic fact sharing the same geometry and source time."""
    return {"id": identity, "kind": kind, "startMs": 1200, "endMs": 1600,
            "label": "Synthetic fact", "data": data}


class MediaIntervalIndexTests(TestCase):
    def test_review_counts_and_tie_order_also_match_legacy_index(self):
        """Compare exact counts, bounded examples and tie order on the same graph."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "synthetic-source.bin"
            source.write_bytes(b"Synthetic source for interval tests")
            store = MediaGraphStore(root / "graphs")
            rows = [fact(f"issue-{number:02}", "dialogue_issue") for number in range(12)]
            rows += [fact("overlap", "overlap"),
                     fact("unresolved", "utterance", overlapUnresolved=True),
                     fact("unrelated", "extension", extensionType="test.signal", overlapUnresolved=True)]
            snapshot = store.publish(identify(source), {"kind": "video", "durationMs": 2000},
                                     {"nodes": rows, "edges": []}, "en", {})["snapshotId"]
            current = summarize_interval(store, snapshot, 1000, 2000)
            with closing(sqlite3.connect(store.root / f"{snapshot}.sqlite")) as db, db:
                db.execute("DROP INDEX nodes_by_kind")
                db.execute("CREATE INDEX nodes_by_kind ON nodes(kind, start_ms)")
            legacy = summarize_interval(store, snapshot, 1000, 2000)

        del current["timingMs"], legacy["timingMs"]
        self.assertEqual(legacy, current)
        self.assertEqual(current["sections"]["events"]["omitted"], 5)
        doubts = {item["code"]: item for item in current["doubts"]["items"]}
        issues = doubts["dialogue-review-candidate"]
        self.assertEqual((issues["count"], issues["omittedExamples"]), (12, 9))
        self.assertEqual([row["id"] for row in issues["examples"]], ["issue-00", "issue-01", "issue-02"])
        self.assertEqual([row["id"] for row in doubts["speech-overlap"]["examples"]],
                         ["overlap", "unresolved"])

    def test_shared_start_preserves_legacy_frame_and_scene_order(self):
        """Choose the same shot, depth and scene order for facts sharing a start time."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "synthetic-source.bin"
            source.write_bytes(b"Synthetic source for frame ordering")
            source_identity = identify(source)
            store = MediaGraphStore(root / "graphs")
            rows = []
            for kind in ("shot", "camera_motion", "person_motion"):
                for label, end, depth in (("long", 200, "background"), ("short", 100, "foreground")):
                    row = fact(f"{kind}-{label}", kind, color={"name": label}, track=1,
                               faceId=1, depth=depth, moodbook={"name": label})
                    rows.append({**row, "startMs": 50, "endMs": end, "label": label})
            graph = {
                "nodes": rows, "edges": [], "faceSampleMs": 40,
                "faceFrames": [(40, [{"track": 1, "faceId": 1, "active": False, "box": [.1, .1, .2, .2]}])],
                "coverage": {"scene": [{"startMs": 0, "endMs": 2000, "status": "sampled"}]},
            }
            snapshot = store.publish(source_identity, {"kind": "video", "durationMs": 2000}, graph, "en", {})["snapshotId"]
            _, _, frame = store.frame_context(snapshot, 75)
            scene, _ = store.scene_evidence(snapshot, source_identity["sha256"], "en")

            with closing(sqlite3.connect(store.root / f"{snapshot}.sqlite")) as db, db:
                db.execute("DROP INDEX nodes_by_kind")
                db.execute("CREATE INDEX nodes_by_kind ON nodes(kind, start_ms)")

            self.assertEqual(frame, store.frame_context(snapshot, 75)[2])
            self.assertEqual(scene, store.scene_evidence(snapshot, source_identity["sha256"], "en")[0])
            self.assertEqual(frame["shot"], "short")
            self.assertEqual(frame["people"][0]["depth"], "foreground")
            self.assertEqual([color["name"] for color in scene["shotColors"]], ["long", "short"])
            self.assertEqual([motion["endMs"] for motion in scene["cameraMotion"]], [200, 100])
