"""Check routing and the frame-clock and face-track mask contract."""

from unittest import TestCase

from voice_studio.vision.person_layers import layer_plan, validate_mask_request


class PersonLayerTests(TestCase):
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
