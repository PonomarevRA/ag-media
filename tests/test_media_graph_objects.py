"""Keep tracked objects, context, and reused evidence anchored to the source clock."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.graph.media_graph_model import build_graph, object_layers
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


class MediaGraphObjectTests(TestCase):
    def test_same_class_nested_proposal_keeps_evidence_without_splitting_track(self):
        phone = {"label": "cell phone", "box": [.4869, .734, .1297, .2237], "score": .524}
        partial = {"label": "cell phone", "box": [.5075, .7636, .0624, .1233], "score": .387}
        other = {"label": "cell phone", "box": [.2828, .5913, .0966, .2127], "score": .517}
        report = {"durationMs": 3000, "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}]}},
                  "sceneEvidence": {"frames": [
                      {"timeMs": 100, "objects": [phone, other]},
                      {"timeMs": 1100, "objects": [phone, other, partial]},
                      {"timeMs": 2100, "objects": [phone, other]},
                  ]}}
        layers = object_layers(report)
        self.assertEqual(len(layers["objectProfiles"]), 2)
        self.assertEqual([clip["sampleCount"] for clip in layers["objectTimeline"]], [3, 3])
        geometry = [[box for _, box, _ in clip["boxSamples"]] for clip in layers["objectTimeline"]]
        self.assertCountEqual(geometry, [[phone["box"]] * 3, [other["box"]] * 3])
        for clip in layers["objectTimeline"]:
            self.assertEqual([stamp for stamp, _, _ in clip["boxSamples"]], [100, 1100, 2100])
        graph = build_graph(report, kind="video", duration_ms=3000)
        source_rows = {tuple(node["data"]["box"]): node["id"] for node in graph["nodes"]
                       if node["kind"] == "object_observation" and node["startMs"] == 1100}
        self.assertIn((source_rows[tuple(partial["box"])], "part_of_observation",
                       source_rows[tuple(phone["box"])]), graph["edges"])

    def test_nested_tv_estimates_form_one_track_and_keep_disjoint_displays(self):
        large = {"label": "tv", "box": [.50, .10, .42, .76], "score": .52}
        middle = {**large, "box": [.55, .20, .32, .56], "score": .54}
        smaller = {**large, "box": [.60, .31, .22, .34], "score": .53}
        tiny = {**large, "box": [.63, .36, .16, .23], "score": .59}
        separate = {**large, "box": [.05, .20, .31, .52], "score": .67}
        report = {"durationMs": 1500, "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}]}},
                  "sceneEvidence": {"frames": [
                      {"timeMs": 100, "objects": [tiny, middle, large, separate, smaller]},
                      {"timeMs": 500, "objects": [tiny, separate]},
                      {"timeMs": 900, "objects": [large, separate]},
                  ]}}
        self.assertEqual([row["sampleCount"] for row in object_layers(report)["objectTimeline"]], [3, 3])
        graph = build_graph(report, kind="video", duration_ms=1500)
        aliases = [(source, target) for source, relation, target in graph["edges"] if relation == "part_of_observation"]
        self.assertEqual(len(aliases), 3)
        self.assertEqual(len({target for _, target in aliases}), 1)

    def test_detail_extends_baseline_without_stealing_its_same_frame_match(self):
        first = {"label": "cup", "box": [.30, .5, .1, .1], "score": .70}
        moved = {**first, "box": [.33, .5, .1, .1]}
        competing = {**first, "box": [.265, .5, .1, .1], "detailBasis": "small-object-region"}
        earlier = {**first, "detailBasis": "small-object-region"}
        frames = [{"timeMs": 300, "objects": [first]}, {"timeMs": 600, "objects": [moved]}]
        report = {"durationMs": 2000, "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}]}},
                  "sceneEvidence": {"frames": frames}}
        baseline, = object_layers(report)["objectTimeline"]

        report["sceneEvidence"]["frames"] = [
            {"timeMs": 100, "objects": [earlier]},
            frames[0], {"timeMs": 600, "objects": [moved, competing]},
        ]
        clips = object_layers(report)["objectTimeline"]
        original = next(item for item in clips if item["id"] == baseline["id"])
        self.assertEqual(original["sampleCount"], 2)
        self.assertEqual((original["startMs"], original["endMs"]), (300, 601))

        report["sceneEvidence"]["frames"] = [{"timeMs": 100, "objects": [earlier]}, *frames]
        extended, = object_layers(report)["objectTimeline"]
        self.assertEqual(extended["id"], baseline["id"])
        self.assertEqual((extended["startMs"], extended["sampleCount"]), (100, 3))

    def test_conflicting_crop_label_cannot_rename_the_baseline_profile(self):
        phone = {"label": "cell phone", "box": [.3, .5, .1, .1], "score": .74}
        book = {**phone, "label": "book", "score": .99, "detailBasis": "small-object-region"}
        report = {"durationMs": 1300, "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}]}},
                  "sceneEvidence": {"frames": [{"timeMs": 100, "objects": [phone]},
                                               {"timeMs": 300, "objects": [phone]}]}}
        original, = object_layers(report)["objectProfiles"]
        report["sceneEvidence"]["frames"][1]["objects"].append(book)
        profiles = object_layers(report)["objectProfiles"]
        self.assertEqual({row["label"] for row in profiles}, {"cell phone", "book"})
        retained = next(row for row in profiles if row["id"] == original["id"])
        self.assertEqual(retained["label"], "cell phone")

    def test_reappearing_object_preserves_another_visible_object_and_source_frames(self):
        phone = {"label": "cell phone", "box": [.46, .72, .13, .22], "score": .65,
                 "objectPalette": [{"hex": "#355576", "share": .7}]}
        other = {**phone, "box": [.25, .56, .13, .22]}
        report = {"durationMs": 6000, "videoEvidence": {"editorTimeline": {
            "shots": [{"startMs": 0}, {"startMs": 3500}]}},
            "sceneEvidence": {"frames": [{"timeMs": time, "objects": [phone, other]}
                                          for time in (100, 500, 3800, 4200)]}}
        layers = object_layers(report)
        self.assertEqual(len(layers["objectProfiles"]), 2)
        item = next(row for row in layers["objectProfiles"]
                    if row["appearances"][0]["representative"]["box"] == phone["box"])
        self.assertEqual(item["shotIds"], ["shot:0", "shot:3500"])
        self.assertEqual([(a["startMs"], a["endMs"], a["sampleCount"]) for a in item["appearances"]],
                         [(100, 501, 2), (3800, 4201, 2)])
        self.assertEqual({clip["profileId"] for clip in layers["objectTimeline"]
                          if clip["representative"]["box"] == phone["box"]}, {item["id"]})
        graph = build_graph(report, kind="video", duration_ms=6000)
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.mp4"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            snapshot = store.publish(identify(source), {"kind": "video", "durationMs": 6000},
                                     graph, "ru", {})["snapshotId"]
            _, _, context = store.frame_context(snapshot, at_ms=3800, node_id=item["id"])
            self.assertEqual(context["targetObservationId"], item["appearances"][1]["representative"]["observationId"])
            _, _, gap = store.frame_context(snapshot, at_ms=2000, node_id=item["id"])
            self.assertNotIn("targetObservationId", gap)

    def test_profile_combines_competing_labels_and_keeps_distinct_items_and_cuts(self):
        table = {"label": "dining table", "box": [.0583, .4968, .8902, .1312], "score": .52}
        boat = {"label": "boat", "box": [.0141, .4762, .9844, .3032], "score": .321}
        laptop = {"label": "laptop", "box": [.35, .5, .15, .12], "score": .65}
        report = {"durationMs": 2000, "videoEvidence": {"editorTimeline": {
            "durationMs": 2000, "shots": [{"startMs": 0}, {"startMs": 1000}]}},
            "sceneEvidence": {"frames": [
                {"timeMs": 100, "objects": [table, boat, laptop]},
                {"timeMs": 580, "objects": [{**boat, "score": .324}]},
                {"timeMs": 1050, "objects": [boat]},
            ]}}
        layers = object_layers(report)
        profiles = layers["objectProfiles"]
        self.assertEqual(len(profiles), 3)
        table_profile = next(item for item in profiles if item["label"] == "dining table")
        self.assertEqual([(item["label"], item["score"]) for item in table_profile["labels"]],
                         [("dining table", .52), ("boat", .324)])
        boat_clips = [item for item in layers["objectTimeline"] if item["label"] == "boat"]
        self.assertEqual(boat_clips[0]["profileId"], table_profile["id"])
        self.assertNotEqual(boat_clips[1]["profileId"], table_profile["id"])
        graph = build_graph(report, kind="video", duration_ms=2000)
        node = next(node for node in graph["nodes"] if node["id"] == table_profile["id"])
        self.assertIn((boat_clips[0]["id"], "part_of_profile", node["id"]), graph["edges"])
        self.assertEqual(node["data"]["labels"], table_profile["labels"])
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.mov"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            snapshot = store.publish(identify(source), {"kind": "video", "durationMs": 2000},
                                     graph, "ru", {})["snapshotId"]
            _, time, context = store.frame_context(snapshot, node_id=node["id"])
            self.assertEqual((time, context["targetObservationId"]), (100, table_profile["labels"][0]["observationId"]))

    def test_normalized_visual_evidence_recovers_only_the_supported_same_shot_match(self):
        palette = [{"hex": "#942223", "share": .58}]
        first = {"label": "car", "box": [.2, .2, .3, .4], "score": .65, "objectPalette": palette}
        match = {"timeMs": 100, "label": "car", "box": first["box"], "objectSimilarity": .97,
                 "edgeSimilarity": .87, "backgroundSimilarity": .29, "scaleRatio": .9}
        second = {"label": "truck", "box": [.26, .2, .25, .36], "score": .61,
                  "objectPalette": palette, "objectMatches": [match]}
        report = {"durationMs": 1100, "sceneEvidence": {"status": "ok", "frames": [
            {"timeMs": 100, "objects": [first]}, {"timeMs": 400, "objects": [second]},
        ]}, "videoEvidence": {"editorTimeline": {"shots": [{"startMs": 0}]}}}
        profile, = object_layers(report)["objectProfiles"]
        appearance = profile["appearance"]
        self.assertEqual([item["label"] for item in appearance["views"]], ["car", "truck"])
        evidence, = appearance["matches"]
        self.assertEqual((evidence["objectSimilarity"], evidence["backgroundSimilarity"],
                          evidence["scaleRatio"]), (.97, .29, .9))
        self.assertEqual(evidence["toObservationId"], appearance["views"][1]["observationId"])

        graph = build_graph(report, kind="video", duration_ms=1100)
        node = next(node for node in graph["nodes"] if node["id"] == profile["id"])
        self.assertEqual(node["data"]["appearance"], appearance)
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.mov"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            snapshot = store.publish(identify(source), {"kind": "video", "durationMs": 1100},
                                     graph, "ru", {})["snapshotId"]
            reused, _ = store.scene_evidence(snapshot, identify(source)["sha256"], "ru")
            self.assertEqual(object_layers({**report, "sceneEvidence": reused})["objectProfiles"], [profile])

        match.update(objectSimilarity=.42, backgroundSimilarity=.99)
        self.assertEqual(len(object_layers(report)["objectProfiles"]), 2)
        match["objectSimilarity"] = .97
        report["videoEvidence"]["editorTimeline"]["shots"].append({"startMs": 300})
        self.assertEqual(len(object_layers(report)["objectProfiles"]), 2)

    def test_object_text_is_deduplicated_and_bound_to_unambiguous_same_frame_evidence(self):
        table = {"label": "table", "box": [.08, .38, .82, .47]}
        cup = {"label": "cup", "box": [.40, .44, .20, .22]}
        bottle = {"label": "bottle", "box": [.57, .43, .22, .24]}
        card = {"text": "Синий чай", "box": [.43, .48, .08, .04]}
        report = {"durationMs": 1500, "videoEvidence": {"editorTimeline": {
            "durationMs": 1500, "shots": [{"startMs": 0}, {"startMs": 1000}]}},
            "sceneEvidence": {"frames": [
                {"timeMs": 0, "objects": [table, cup, bottle], "text": [
                    card, {"text": "Перекрытие", "box": [.57, .5, .025, .03]},
                    {"text": "На столе", "box": [.19, .54, .10, .04]},
                    {"text": "Субтитр", "box": [.34, .90, .30, .044]},
                ]},
                {"timeMs": 400, "objects": [table, cup, bottle], "text": [
                    {**card, "text": "синий   чай"}, {"text": "2 литра", "box": [.64, .49, .08, .04]},
                ]},
                {"timeMs": 550, "text": [{**card, "text": "Без рамки предмета"}]},
                {"timeMs": 1100, "objects": [cup]},
            ]}}
        profiles = object_layers(report)["objectProfiles"]
        first_shot = {item["label"]: item for item in profiles if item["shotId"] == "shot:0"}
        self.assertEqual([entry["text"] for entry in first_shot["table"]["visibleText"]], ["На столе"])
        self.assertEqual([entry["text"] for entry in first_shot["bottle"]["visibleText"]], ["2 литра"])
        cup_text, = first_shot["cup"]["visibleText"]
        self.assertEqual((cup_text["text"], cup_text["timeMs"]), ("Синий чай", 0))
        later_cup = next(item for item in profiles if item["shotId"] == "shot:1000")
        self.assertNotIn("visibleText", later_cup)

        graph = build_graph(report, kind="video", duration_ms=1500)
        snapshot_cup = next(node for node in graph["nodes"] if node["id"] == first_shot["cup"]["id"])
        self.assertEqual(snapshot_cup["data"]["visibleText"], [cup_text])
        self.assertIn((snapshot_cup["id"], "has_visible_text", cup_text["observationId"]), graph["edges"])

    def test_stable_profile_connects_three_exclusive_label_fragments(self):
        frames = [{"timeMs": stamp, "objects": [{"label": label, "box": [.3, .4, .16, .2], "score": score}]}
                  for stamp, label, score in [(0, "book", .42), (350, "cup", .75), (710, "bottle", .48)]]
        report = {"durationMs": 1000, "sceneEvidence": {"frames": frames},
                  "videoEvidence": {"editorTimeline": {"durationMs": 1000, "shots": [{"startMs": 0}]}}}
        model = object_layers(report)
        profile, = model["objectProfiles"]
        self.assertEqual(profile["label"], "cup")
        self.assertEqual(len(profile["labels"]), 3)
        self.assertEqual({row["profileId"] for row in model["objectTimeline"]}, {profile["id"]})

    def test_profile_abstains_when_two_tracks_can_explain_a_label_switch(self):
        report = {"durationMs": 1000, "videoEvidence": {"editorTimeline": {
            "durationMs": 1000, "shots": [{"startMs": 0}]}},
            "sceneEvidence": {"frames": [
                {"timeMs": 0, "objects": [{"label": "table", "box": [.15, .2, .4, .4], "score": .72},
                                          {"label": "table", "box": [.25, .2, .4, .4], "score": .72}]},
                {"timeMs": 400, "objects": [{"label": "boat", "box": [.2, .2, .4, .4], "score": .34}]},
            ]}}
        self.assertEqual(len(object_layers(report)["objectProfiles"]), 3)
        report["sceneEvidence"]["frames"][0]["objects"].pop()
        self.assertEqual(len(object_layers(report)["objectProfiles"]), 1)

    def test_track_preview_and_graph_frame_share_the_strongest_observation(self):
        report = {"durationMs": 1000,
                  "videoEvidence": {"editorTimeline": {"durationMs": 1000, "shots": [{"startMs": 0}]}},
                  "sceneEvidence": {"frames": [
            {"timeMs": 0, "objects": [{"label": "boat", "box": [.1, .2, .5, .2], "score": .4},
                                      {"label": "cup", "box": [.8, .4, .1, .1]}]},
            {"timeMs": 400, "objects": [{"label": "boat", "box": [.15, .2, .5, .2], "score": .87}]},
            {"timeMs": 800, "objects": [{"label": "boat", "box": [.16, .22, .5, .2], "score": .87}]},
        ]}}
        graph = build_graph(report, kind="video", duration_ms=1000)
        clips = {clip["label"]: clip for clip in object_layers(report)["objectTimeline"]}
        boat = clips["boat"]
        self.assertEqual(boat["startMs"], 0)
        sample = boat["representative"]
        self.assertEqual((sample["timeMs"], sample["score"], sample["box"]), (400, .87, [.15, .2, .5, .2]))
        self.assertIsNone(clips["cup"]["representative"]["score"])
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.mov"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            media = {"kind": "video", "durationMs": 1000}
            snapshot = store.publish(identify(source), media, graph, "ru", {})["snapshotId"]
            self.assertEqual(store.node(snapshot, boat["id"])["data"]["representative"], sample)
            self.assertEqual(store.node(snapshot, sample["observationId"])["data"]["score"], sample["score"])
            _, time, context = store.frame_context(snapshot, node_id=boat["id"])
            self.assertEqual((time, context["targetObservationId"]), (400, sample["observationId"]))
            self.assertEqual(store.frame_context(snapshot, at_ms=0, node_id=boat["id"])[1], 0)
            for node in graph["nodes"]:
                node["data"].pop("representative", None)
            old = store.publish(identify(source), media, graph, "ru", {})["snapshotId"]
            legacy = store.node(old, boat["id"])["data"]["representative"]
            self.assertEqual((legacy["timeMs"], legacy["score"]), (0, .4))
            self.assertEqual(store.frame_context(old, node_id=boat["id"])[2]["targetObservationId"],
                             legacy["observationId"])

    def test_separate_cups_and_context_stop_at_cut(self):
        scenes = []
        for stamp in (0, 400, 800):
            scenes.append({"timeMs": stamp, "objects": [
                {"label": "cup", "box": [.20, .40, .10, .12], "score": .82},
                {"label": "cup", "box": [.75, .65, .10, .12], "score": .91},
            ]})
        face = [[1, .16, .24, .18, .14, 0, 1], [2, .68, .18, .18, .14, 0, 2]]
        report = {"language": "ru", "durationMs": 1200, "segments": [],
                  "videoEvidence": {"editorTimeline": {"durationMs": 1200, "shots": [{"startMs": 0}, {"startMs": 800}]},
                                    "timeline": {"stepMs": 400, "frames": [face, face, face]},
                                    "faceSpeakers": {"1": {"speaker": 1, "basis": "audio-visual"}}},
                  "sceneEvidence": {"status": "ok", "frames": scenes}}
        graph = build_graph(report, kind="video", duration_ms=1200)
        tracks = [node for node in graph["nodes"] if node["kind"] == "object_track"]
        self.assertEqual(sorted((node["startMs"], node["data"]["sampleCount"]) for node in tracks),
                         [(0, 2), (0, 2), (800, 1), (800, 1)])
        clips = object_layers(report)["objectTimeline"]
        self.assertEqual({clip["id"] for clip in clips}, {node["id"] for node in tracks})
        self.assertEqual([(clip["startMs"], clip["endMs"], clip["shotId"]) for clip in clips],
                         [(0, 401, "shot:0"), (0, 401, "shot:0"), (800, 801, "shot:800"), (800, 801, "shot:800")])
        speaker = next(node for node in graph["nodes"] if node["kind"] == "speaker")
        self.assertEqual([entry["weight"] for entry in speaker["data"]["shotContext"]], [.08, .02])
        self.assertEqual({entry["shotId"] for entry in speaker["data"]["shotContext"]}, {"shot:0"})
        self.assertEqual([node["id"] for node in graph["nodes"] if node["kind"] == "speaker"], ["speaker:1"])

    def test_reuse_requires_exact_hash_and_precise_geometry(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.mov"
            source.write_bytes(b"fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            media = {"kind": "video", "durationMs": 1000, "streams": [{"kind": "video", "width": 1080, "height": 1920}]}
            box = [.2, .4, .1, .1]
            nodes = [{"id": "sample", "kind": "object_observation", "startMs": 40, "endMs": 41,
                      "label": "cup", "data": {"box": box, "score": .8}}]
            coverage = {"scene": [{"startMs": 0, "endMs": 1000, "status": "sampled"}]}
            file = identify(source)
            key = store.publish(file, media, {"nodes": nodes, "edges": [], "coverage": coverage}, "ru", {})["snapshotId"]
            reused, windows = store.scene_evidence(key, file["sha256"], "ru")
            self.assertEqual(reused["frames"][0]["objects"], [{"label": "cup", "box": box, "score": .8}])
            self.assertEqual(windows, coverage["scene"])
            self.assertEqual(store.node(key, "sample")["data"]["boxPx"], [216, 768, 108, 192])
            context = store.frame_context(key, node_id="sample")[2]
            self.assertEqual(context["targetObservationId"], "sample")
            self.assertEqual([mark["kind"] for mark in context["boxes"]], ["target"])
            with self.assertRaises(ValueError):
                store.scene_evidence(key, "other", "ru")
            coverage["scene"] = [
                {"startMs": 0, "endMs": 500, "status": "sampled"},
                {"startMs": 500, "endMs": 1000, "status": "failed"},
            ]
            partial = store.publish(file, media, {"nodes": nodes, "edges": [], "coverage": coverage},
                                    "ru", {})["snapshotId"]
            with self.assertRaisesRegex(ValueError, "незавершённые"):
                store.scene_evidence(partial, file["sha256"], "ru")
