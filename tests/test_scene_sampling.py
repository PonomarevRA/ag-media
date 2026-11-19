"""Check native frame coverage independently of installed model weights."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.synthesis.engines import ffmpeg
from voice_studio.speech.media_transcript import FORMATS
from voice_studio.vision.scene_sampling import evenly_spaced, label_visible_faces, sample_plan
from voice_studio.vision.video_scene import review_scene


def frames_at(fps, count):
    """Build source ordinals and millisecond timestamps for a native frame rate."""
    return [(index, round(index * 1000 / fps)) for index in range(count)]


class SceneSamplingTests(TestCase):
    def test_basic_checks_ten_frames_per_shot_or_every_available_frame(self):
        rows = sample_plan(frames_at(25, 46), [40])
        self.assertEqual([sum(row["shot"] == shot and row["objects"] for row in rows)
                          for shot in (0, 1)], [10, 6])
        self.assertEqual(len(rows), len({row["sourceFrame"] for row in rows}))
        self.assertEqual(sum(row["sectors"] for row in rows), 2)

    def test_small_detail_reuses_the_baseline_clock(self):
        frames = frames_at(30, 720)
        self.assertEqual(sample_plan(frames, [10, 411], "small-detail"), sample_plan(frames, [10, 411], "basic"))

    def test_long_shot_samples_objects_from_the_beginning(self):
        rows = sample_plan(frames_at(25, 500), [])
        objects = [row["timeMs"] for row in rows if row["objects"]]
        self.assertEqual(len(objects), 20)
        self.assertLess(objects[0], 1000)
        self.assertGreater(objects[-1], 19000)
        self.assertTrue(all(right - left <= 1000 for left, right in zip(objects, objects[1:])))

    def test_detailed_restarts_native_stride_at_each_shot(self):
        rows = sample_plan(frames_at(60, 92), [46], "every-third")
        expected = list(range(0, 46, 3)) + list(range(46, 92, 3))
        self.assertEqual([row["sourceFrame"] for row in rows if row["objects"]], expected)
        basic = sample_plan(frames_at(60, 92), [46])
        self.assertEqual([row["sourceFrame"] for row in rows if row["ocr"]],
                         [row["sourceFrame"] for row in basic])
        self.assertEqual(rows[0]["timeMs"], 0)
        self.assertIn(50, [row["timeMs"] for row in rows if row["objects"]])

    def test_detailed_tops_up_short_shots_and_preserves_cadence_on_variable_timestamps(self):
        stamps = [0, 8, 22, 58, 73, 93, 135, 154, 200, 222, 260, 281, 309]
        rows = sample_plan(list(enumerate(stamps)), [], "every-third")
        checked = [row["sourceFrame"] for row in rows if row["objects"]]
        self.assertEqual(len(checked), 10)
        self.assertTrue(set(range(0, len(stamps), 3)).issubset(checked))
        self.assertTrue(all(stamps[row["sourceFrame"]] == row["timeMs"] for row in rows))

    def test_detailed_keeps_its_cadence_even_below_three_fps(self):
        rows = sample_plan(frames_at(2, 90), [], "every-third")
        self.assertEqual([row["sourceFrame"] for row in rows if row["objects"]], list(range(0, 90, 3)))

    def test_basic_budget_keeps_the_minimum_across_many_cuts(self):
        rows = sample_plan(frames_at(30, 480), range(30, 480, 30))
        self.assertEqual(sum(row["objects"] for row in rows), 160)

    def test_ocr_budget_and_visible_faces_still_use_timestamps(self):
        frames = frames_at(25, 1500)
        rows = sample_plan(frames, [])
        self.assertEqual(len(rows), 60)
        self.assertEqual([row["timeMs"] for row in evenly_spaced(rows, 4)], [480, 19480, 39480, 59480])
        label_visible_faces(rows, {"stepMs": 40, "frames": [[], *[[] for _ in range(11)],
                                                           [[1, .1, .2, .3, .4, 1, 8]]]})
        self.assertEqual(rows[0]["faces"], [[.1, .2, .3, .4]])
        self.assertEqual(rows[1]["faces"], [])

    def test_scene_decodes_planned_native_frames_past_one_minute(self):
        def inspect_samples(script, arguments, imports, timeout):
            """Check the real FFmpeg files before model inference would begin."""
            folder = arguments[0]
            rows = json.loads((folder / "samples.json").read_bytes())["samples"]
            self.assertEqual(len(list(folder.glob("*.jpg"))), len(rows))
            return {"status": "ok", "times": [row["timeMs"] for row in rows if row["objects"]]}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long.mp4"
            ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=64x64:r=3:d=65", str(source)])
            with patch('voice_studio.vision.video_scene.run_worker', side_effect=inspect_samples):
                full = review_scene(source, root, FORMATS, object_sampling="every-third")
                partition = review_scene(source, root, FORMATS, window=(60_000, 64_000))
        self.assertEqual(full["times"][-1], 64_000)
        self.assertEqual((partition["times"][0], partition["times"][-1]), (0, 3667))
