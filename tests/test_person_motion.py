"""Check motion compensation, shot boundaries and conservative relative depth."""

from unittest import TestCase

from voice_studio.vision.person_motion import measure_people_motion


def hit(track, x, *, y=.23, height=.10, face=None):
    """Create a normalized face box in the existing 40 ms track format."""
    return [track, x, y, height * .8, height, 0, face or track]


def evidence(frames, *, cuts=(), estimated=(), body=()):
    """Build the smallest report used by the motion estimator."""
    return {"timeline": {"frames": frames, "stepMs": 40, "estimatedFrames": list(estimated)},
            "editorTimeline": {"shots": [{"startMs": stamp} for stamp in cuts]},
            "personLayer": {"frames": list(body)}}


def pan(first=0, last=400, *, shift=.10, scale=1.0, confidence=.85):
    """Describe screen content moving right under a camera pan to the left."""
    return {"startMs": first, "endMs": last, "shiftX": shift, "shiftY": 0,
            "scale": scale, "confidence": confidence}


class PersonMotionTests(TestCase):
    def test_panning_background_does_not_move_static_person_and_preserves_counter_motion(self):
        """Identical screen motion cancels, while a counter-walking person moves left."""
        frames = [[hit(1, .20 + index * .01), hit(2, .60 + index * .006)] for index in range(12)]
        rows = measure_people_motion(evidence(frames), [pan()], 480)
        self.assertEqual([(row["horizontal"], row["basis"]) for row in rows],
                         [("still", "camera-compensated"), ("left", "camera-compensated")])
        self.assertAlmostEqual(rows[1]["deltaX"], -.04)
        self.assertTrue(all(row["depth"] == "unknown" for row in rows))

    def test_zoom_uses_normalized_image_center(self):
        """A stationary point away from the center moves in pixels during an optical zoom."""
        frames = [[hit(1, .5 + (.20 - .5) * (1 + index * .02),
                       y=.5 + (.23 - .5) * (1 + index * .02), height=.10 * (1 + index * .02))]
                  for index in range(12)]
        first, = measure_people_motion(evidence(frames), [pan(shift=0, scale=1.20)], 480)
        self.assertEqual((first["horizontal"], first["vertical"]), ("still", "still"))
        self.assertEqual(first["basis"], "camera-compensated")

    def test_low_confidence_camera_uses_screen_space_direction(self):
        """An unsupported global estimate does not suppress actual screen displacement."""
        frames = [[hit(1, .20 + index * .01, y=.25 - index * .005)] for index in range(10)]
        row, = measure_people_motion(evidence(frames), [pan(confidence=.35)], 400)
        self.assertEqual((row["horizontal"], row["vertical"], row["basis"]), ("right", "up", "screen-space"))

    def test_missing_or_invalid_camera_records_leave_screen_space_motion(self):
        """An old or manually edited JSON report can omit optional camera evidence."""
        frames = [[hit(1, .20 + index * .01)] for index in range(10)]
        for camera in (None, [False, {}]):
            row, = measure_people_motion(evidence(frames), camera, 400)
            self.assertEqual((row["horizontal"], row["basis"]), ("right", "screen-space"))

    def test_cuts_gaps_and_repeated_identity_never_bridge_two_shots(self):
        """Two same-person appearances remain separate source tracks on opposite sides of a cut."""
        frames = [[hit(2, .70, face=1)] if index < 10 else [hit(3, .18, face=1)]
                  for index in range(20)]
        rows = measure_people_motion(evidence(frames), [pan(80, 520)], 800, shot_starts=[400])
        self.assertEqual([(row["faceId"], row["track"], row["startMs"], row["endMs"], row["horizontal"])
                          for row in rows], [(1, 2, 0, 400, "still"), (1, 3, 400, 800, "still")])
        frames[5] = []
        rows = measure_people_motion(evidence(frames, cuts=[400]), [], 800)
        self.assertEqual([row["track"] for row in rows], [3])

    def test_estimated_faces_are_not_motion_endpoints(self):
        """A highly shifted but interpolated edge frame cannot create apparent movement."""
        frames = [[hit(1, .20)] for _ in range(10)]
        frames[-1] = [hit(1, .43)]
        row, = measure_people_motion(evidence(frames, estimated=[9]), [], 400)
        self.assertEqual(row["horizontal"], "still")

    def test_legacy_track_without_separate_identity_keeps_its_displayed_id(self):
        """Prior face timelines used the track number as both IDs."""
        frames = [[hit(4, .20 + index * .01)[:6]] for index in range(10)]
        row, = measure_people_motion(evidence(frames), [], 400)
        self.assertEqual((row["faceId"], row["track"], row["horizontal"]), (4, 4, "right"))

    def test_relative_depth_uses_simultaneous_face_size_and_consistent_body_baseline(self):
        """Physical-looking depth remains ordinal and requires corroborating geometry."""
        frames = [[hit(1, .18, height=.14), hit(2, .60, height=.10)] for _ in range(10)]
        near = {"track": 1, "faceId": 1, "box": [.12, .19, .27, .69]}
        far = {"track": 2, "faceId": 2, "box": [.55, .19, .23, .48]}
        body = [{"timeMs": 200, "people": [near, far]}]
        rows = measure_people_motion(evidence(frames, body=body), [], 400)
        self.assertEqual([row["depth"] for row in rows], ["foreground", "background"])
        self.assertTrue(all(row["depthBasis"] == "relative-face-scale-body-position" for row in rows))
        near["box"][3], far["box"][3] = far["box"][3], near["box"][3]
        rows = measure_people_motion(evidence(frames, body=body), [], 400)
        self.assertEqual([row["depth"] for row in rows], ["unknown", "unknown"])

    def test_strong_face_scale_only_assigns_lower_confidence_and_three_levels(self):
        """Unambiguous size ordering is reported as a weaker shot-local hint."""
        frames = [[hit(1, .12, height=.20), hit(2, .45, height=.11),
                   hit(3, .74, height=.060)] for _ in range(10)]
        rows = measure_people_motion(evidence(frames), [], 400)
        self.assertEqual([row["depth"] for row in rows], ["foreground", "midground", "background"])
        self.assertTrue(all(row["depthConfidence"] <= .42 for row in rows))

    def test_similar_projected_sizes_can_share_a_layer_with_external_evidence(self):
        """Two peers share foreground when each separately compares as closer to a third."""
        frames = [[hit(1, .12, height=.19), hit(2, .45, height=.18),
                   hit(3, .74, height=.080)] for _ in range(10)]
        rows = measure_people_motion(evidence(frames), [], 400)
        self.assertEqual([row["depth"] for row in rows], ["foreground", "foreground", "background"])

    def test_duplicate_identity_and_unstable_face_size_abstain(self):
        """Keep distinct people or track fragments from making a confident depth claim."""
        frames = [[hit(1, .12, height=.20), hit(2, .45, height=.09, face=1),
                   hit(3, .74, height=.060 if index < 5 else .095)] for index in range(10)]
        row, = measure_people_motion(evidence(frames), [], 400)
        self.assertEqual((row["faceId"], row["depth"]), (3, "unknown"))
