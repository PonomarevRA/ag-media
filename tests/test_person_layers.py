"""Check routing and the frame-clock and face-track mask contract."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from voice_studio.vision.person_layers import dispatch_layers, layer_plan, validate_mask_request


class PersonLayerTests(TestCase):
    def test_body_still_runs_without_mediapipe(self):
        """A missing gaze runtime keeps NanoDet data without claiming gaze was measured."""
        plan = layer_plan(body=True, gaze=True)
        with TemporaryDirectory() as folder, \
                patch("voice_studio.vision.person_layers.mediapipe_interpreter", return_value=(None, None)), \
                patch("voice_studio.vision.person_layers.subprocess.run", return_value=SimpleNamespace(stdout='{"frames": [{"timeMs":0}]}')):
            result = dispatch_layers(Path(folder), [], {}, plan)
            request = json.loads((Path(folder) / "request.json").read_text())
        self.assertEqual(request["plan"], layer_plan(body=True))
        self.assertEqual(result["plan"], plan)
        self.assertTrue(result["gazeUnavailable"])
        self.assertEqual(result["frames"], [{"timeMs": 0}])

    def test_tasks_load_only_their_dependencies(self):
        self.assertEqual(layer_plan(), {"body": False, "gaze": False, "masks": False})
        self.assertEqual(layer_plan(gaze=True), {"body": False, "gaze": True, "masks": False})
        self.assertEqual(layer_plan(gaze=True, masks=True), {"body": True, "gaze": False, "masks": True})

    def test_mask_request_validates_window_and_face_identity(self):
        timeline = {"stepMs": 40, "frames": [[[1, .1, .2, .3, .4, 1, 8]]]}
        spec = {"faceId": 8, "startMs": 0, "endMs": 250, "timeline": timeline}
        self.assertEqual(validate_mask_request(spec), (8, 0, 250, timeline))
        for patch in ({"faceId": 4}, {"endMs": 6001}, {"timeline": {**timeline, "stepMs": 20}}):
            with self.assertRaises(ValueError):
                validate_mask_request({**spec, **patch})
