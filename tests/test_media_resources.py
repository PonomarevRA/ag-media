"""Cover resource guards and optional native correlation without running heavy models."""

import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.core.media_resources import MediaResources, batch_media_job, batch_slot, stage_threads
from voice_studio.audio import pitch_vdsp
from voice_studio.audio.speech_intonation import python_scores


class MediaResourceTests(TestCase):
    def test_bounded_admission_never_runs_after_expiry(self):
        """The caller deadline applies before taking the native batch allocation."""
        from time import monotonic
        from unittest.mock import Mock
        from voice_studio.core.media_resources import run_bounded_batch_stage
        operation = Mock()
        with self.assertRaises(TimeoutError):
            run_bounded_batch_stage(operation, deadline=monotonic() - 1)
        operation.assert_not_called()

    def test_speech_leaves_room_for_sherpa_and_checks_memory(self):
        self.assertEqual(MediaResources(14, 24_000).speech(True), (8, True))
        self.assertEqual(MediaResources(6, 8_192).speech(True), (2, True))
        self.assertEqual(MediaResources(14, 2_048).speech(True), (8, False))
        self.assertEqual(MediaResources(3, 8_192).speech(True), (3, False))

    def test_native_stage_presets_respect_pilot_limit(self):
        with patch('voice_studio.core.media_resources.pilot_resources', return_value=MediaResources(1, 8_192)):
            self.assertEqual(stage_threads(2), 1)
            self.assertEqual(stage_threads(4), 1)

    def test_nested_batch_reuses_one_slot_and_recovers_after_failure(self):
        @batch_media_job
        def inner():
            raise ValueError("expected")

        @batch_media_job
        def outer():
            inner()

        with TemporaryDirectory() as folder:
            with (patch.dict(os.environ, {"VOICE_RESOURCE_PROFILE": "throughput"}),
                  patch('voice_studio.core.media_resources.SLOT', Path(folder) / "lock"),
                  patch('voice_studio.core.media_resources.batch_slot', wraps=batch_slot) as slot):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "expected"):
                        outer()
                self.assertEqual(slot.call_count, 2)

    def test_vdsp_matches_reference_across_short_lags(self):
        if pitch_vdsp.correlate() is None:
            self.skipTest("Accelerate is only available on macOS")
        frame = [math.cos(index * .15) + math.sin(index * .27) * .06 for index in range(180)]
        baseline = python_scores(frame, 8, 58)
        native = pitch_vdsp.scores(frame, 8, 58)
        self.assertEqual(len(baseline), len(native))
        for expected, actual in zip(baseline, native):
            self.assertAlmostEqual(expected, actual, places=12)
