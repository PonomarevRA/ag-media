"""Verify source pixel use and figure counts when faces are missing."""

import importlib.util
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch


@skipUnless(importlib.util.find_spec("cv2"), "Needs the vision interpreter")
class PersonFacePilotTests(TestCase):
    def test_visible_bodies_without_faces_and_scale_back_from_native_pixels(self):
        import numpy as np
        from voice_studio.vision.person_face_pilot import PersonFacePilot

        with patch("voice_studio.vision.person_face_pilot.ObjectDetector") as detector, \
             patch("voice_studio.vision.person_face_pilot.cv2.FaceDetectorYN.create") as yunet, \
             patch("voice_studio.vision.person_face_pilot.frame_at") as decode:
            detector.return_value.detect_view.return_value = [
                {"label": "person", "score": .9, "box": [.1, .1, .3, .8]},
                {"label": "person", "score": .9, "box": [.6, .1, .3, .8]},
            ]
            face = np.array([[60, 64, 60, 64, 72, 76, 100, 76, 86, 92, 72, 102, 100, 102, .90]],
                            dtype=np.float32)
            yunet.return_value.detect.side_effect = [(1, face), (0, None)]
            decode.return_value = np.zeros((400, 800, 3), np.uint8)
            sample = SimpleNamespace(number=25, cut=False, suspect_cut=False,
                                     color=np.zeros((100, 200, 3), np.uint8))
            pilot = PersonFacePilot("source.mp4", "mov", 7000, "nanodet", "yunet")
            hits = pilot.augment(sample, None, True)
        self.assertEqual(len(hits), 1)
        self.assertEqual(pilot.data()["samples"][0]["count"], 2)
        decode.assert_called_once_with("source.mp4", "mov", 8000, native=True)
        self.assertEqual(yunet.return_value.detect.call_args_list[0].args[0].shape, (180, 288, 3))
        np.testing.assert_array_equal(hits[0][:6], [29, 19.5, 15, 16, 32, 22.5])

    def test_a_shared_face_can_only_claim_one_of_two_head_regions(self):
        from voice_studio.vision.person_face_pilot import unmatched_heads

        heads = [(0, 0, .5, .5), (.1, 0, .6, .5)]
        self.assertEqual(len(unmatched_heads(heads, [[22, 20, 10, 10]], (100, 100))), 1)

    def test_nested_face_boxes_are_duplicates_only_when_centers_agree(self):
        from voice_studio.vision.person_face_pilot import duplicate_face

        self.assertTrue(duplicate_face((45, 45, 10, 10), (40, 40, 20, 20)))
        self.assertTrue(duplicate_face((.7804, .7607, .0503, .0835), (.7724, .778, .0624, .1434)))
        self.assertFalse(duplicate_face((40, 40, 10, 10), (40, 40, 20, 20)))
