"""Check exact scene OCR deduplication, coordinates and shared work limits."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio import mod_runtime
from voice_studio.vision import scene_ocr


READ = {"text": "Open", "box": [.1, .2, .3, .4], "score": .95}
HEALTHY = {"apple-vision": "ok"}


class SceneOCRReuseTests(TestCase):
    def setUp(self):
        """Keep OCR settings and cache files independent from the installed studio."""
        scratch = self.enterContext(TemporaryDirectory())
        self.root = Path(scratch)
        self.enterContext(patch.dict("os.environ", {"VOICE_MEDIA_MODS_DIR": str(self.root / "mods")}))
        self.enterContext(patch.object(mod_runtime, "REGISTRY", None))

    def job(self, stamp, name, *, content=b"exact encoded image", bounds=scene_ocr.FULL, width=160):
        """Create an isolated OCR crop with the requested source-coordinate transform."""
        path = self.root / f"{name}.png"
        path.write_bytes(content)
        return stamp, path, bounds, width, 90

    def test_identical_crops_reproject_at_each_requested_position(self):
        """Share raw text while retaining equal labels at distinct positions and times."""
        first = self.job(100, "left", bounds=(0, 0, .5, .5))
        second = self.job(100, "right", bounds=(.5, 0, .5, .5))
        third = self.job(200, "later", bounds=(0, .5, .5, .5))
        with patch.object(scene_ocr, "read_batches", return_value=([[READ]], HEALTHY)) as worker, \
                scene_ocr.scene_ocr_scope(reuse=True) as work:
            grouped, first_statuses = scene_ocr.recognize([first, second])
            later, second_statuses = scene_ocr.recognize([third], line_limit=8, keep_numbers=True)
            stats = work.summary()
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[0], [first])
        self.assertEqual([row["box"] for row in grouped[100]], [[.05, .1, .15, .2], [.55, .1, .15, .2]])
        self.assertEqual(later[200], [{"text": "Open", "box": [.05, .6, .15, .2]}])
        self.assertEqual((first_statuses, second_statuses), (HEALTHY, HEALTHY))
        self.assertEqual({key: stats[key] for key in ("requested", "dispatched", "reused", "dropped", "batches",
                                                     "cacheEntries")},
                         {"requested": 3, "dispatched": 1, "reused": 2, "dropped": 0, "batches": 1, "cacheEntries": 1})

    def test_distinct_pixels_dimensions_and_psm_are_independent(self):
        """A new glyph, raster dimension or OCR page mode gets a fresh read."""
        first = self.job(0, "a")
        second = self.job(1, "b", content=b"one changed symbol")
        third = self.job(2, "c", width=161)

        def read(jobs, **kwargs):
            return [[READ] for _job in jobs], HEALTHY

        with patch.object(scene_ocr, "read_batches", side_effect=read) as worker, \
                scene_ocr.scene_ocr_scope(reuse=True) as work:
            scene_ocr.recognize([first, second, third])
            scene_ocr.recognize([first], page_mode=7)
            scene_ocr.recognize([first], backend="tesseract")
            scene_ocr.recognize([first])

        self.assertEqual([len(call.args[0]) for call in worker.call_args_list], [3, 1, 1])
        self.assertEqual(work.summary()["dispatched"], 5)
        self.assertEqual(work.summary()["reused"], 1)

    def test_original_and_reused_layout_filter_cases_match(self):
        """Apply edge clipping and line limits after each crop is mapped."""
        edge = {**READ, "box": [0, .25, .2, .4]}
        first = self.job(100, "whole")
        second = self.job(200, "tile", bounds=(.2, .2, .5, .5))
        with patch.object(scene_ocr, "read_batches",
                          side_effect=lambda jobs, **_kwargs: ([[edge] for _job in jobs], HEALTHY)):
            expected = scene_ocr.recognize([first, second])
            with scene_ocr.scene_ocr_scope(reuse=True) as work:
                actual = scene_ocr.recognize([first, second])
        self.assertEqual(expected, actual)
        self.assertEqual(list(actual[0]), [100])
        self.assertEqual(work.summary()["reused"], 1)

    def test_reserve_targeted_capacity_without_failing_healthy_batches(self):
        """Distinguish dropping crops from backend failures and use the whole allowed budget."""
        jobs = [self.job(i, str(i), content=str(i).encode()) for i in range(5)]
        with patch.object(scene_ocr, "read_batches", side_effect=lambda jobs, **_kwargs: (
                [[READ] for _job in jobs], HEALTHY)) as worker, \
                scene_ocr.scene_ocr_scope(4, reserve=2) as work:
            normal, statuses = scene_ocr.recognize(jobs[:3])
            self.assertEqual(scene_ocr.recognize(jobs[2:3]), ({}, {}))
            with work.targeted():
                targeted, limited_statuses = scene_ocr.recognize(jobs[2:])
            self.assertEqual(scene_ocr.recognize(jobs[4:]), ({}, {}))

        self.assertEqual([len(call.args[0]) for call in worker.call_args_list], [2, 2])
        self.assertEqual((statuses, limited_statuses), (HEALTHY, HEALTHY))
        self.assertEqual((list(normal), list(targeted)), ([0, 1], [2, 3]))
        self.assertEqual((work.requested, work.dispatched, work.reused, work.dropped), (8, 4, 0, 4))

    def test_partial_batch_keeps_duplicate_eligible_for_its_own_attempt(self):
        """Do not confuse absent results with cached empty text."""
        first = self.job(1, "absent")
        second = self.job(2, "repair")
        partial = {"apple-vision": "partial"}
        outputs = [([None], partial), ([[READ]], HEALTHY), ([[]], HEALTHY)]
        with patch.object(scene_ocr, "read_batches", side_effect=outputs) as worker, \
                scene_ocr.scene_ocr_scope(reuse=True) as work:
            first_read, status = scene_ocr.recognize([first, second])
            self.assertEqual(first_read, {2: [{"text": "Open", "box": READ["box"]}]})
            self.assertEqual(status, partial)
            self.assertEqual(scene_ocr.recognize([first]), ({}, HEALTHY))
            self.assertEqual(scene_ocr.recognize([second]), ({}, HEALTHY))
        self.assertEqual(worker.call_count, 3)
        self.assertEqual((work.requested, work.dispatched, work.reused, work.dropped), (4, 3, 1, 0))

    def test_engine_metrics_include_native_launch_but_exclude_hits(self):
        """Count real backend work and clear every cache when leaving its clip."""
        first = self.job(1, "native")
        second = self.job(2, "identical")
        with patch.object(scene_ocr, "vision_binary", return_value="vision"), \
                patch.object(scene_ocr.subprocess, "check_output", return_value=json.dumps([[READ]])) as launch:
            with scene_ocr.scene_ocr_scope(reuse=True) as work:
                scene_ocr.recognize([first])
                scene_ocr.recognize([second])
            with scene_ocr.scene_ocr_scope(reuse=True) as next_scene:
                scene_ocr.recognize([second])
        self.assertEqual((work.engine_calls, next_scene.engine_calls, launch.call_count), (1, 1, 2))
        self.assertEqual((work.reused, next_scene.reused), (1, 0))

    def test_replacement_receives_only_admitted_original_jobs(self):
        """Meter the provider boundary without rewriting its inputs or returned layout."""
        jobs = [self.job(stamp, f"external-{stamp}") for stamp in (1, 2)]
        result = {1: [{"text": "Custom result", "box": [.3, .2, .2, .1]}]}, {"custom": "ok"}
        with patch.object(scene_ocr, "is_replacement", return_value=True), \
                scene_ocr.scene_ocr_scope(1, reuse=True, reserve=0) as work:
            with patch("voice_studio.vision.scene_ocr.read_batches") as builtin:
                from unittest.mock import Mock

                operation = Mock(return_value=result)
                invoke = scene_ocr.measure_scene_ocr(operation)
                self.assertIs(invoke(jobs, keep_numbers=True), result)
                builtin.assert_not_called()
        operation.assert_called_once_with([jobs[0]], keep_numbers=True)
        self.assertEqual((work.requested, work.dispatched, work.reused, work.dropped), (2, 1, 0, 1))
