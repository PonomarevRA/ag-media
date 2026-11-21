"""Exercise window continuity, partial output and source-clock evidence."""

from unittest import TestCase
from unittest.mock import Mock, patch

from voice_studio.vision.video_activity_chunks import ActivityTimeline, activity_windows
from voice_studio.vision.video_revolver import raw_activity_window, sampled_window_end, score_visible_faces, window_cues


def fixture(count, track, distance=.1):
    """Make known, simultaneously detected activity without any native models."""
    hit = [track, distance, .2, .2, .3, 1, track]
    return {"timeline": {"frames": [[hit] for _ in range(count)]},
            "editorTimeline": {"shots": [{"startMs": 0}], "facePhotos": {str(track): "jpeg"}},
            "faceTracking": {"status": "ok"}, "spans": []}


class ActivityChunkTests(TestCase):
    def test_speaker_span_reconciles_only_a_corroborated_face_on_the_same_track(self):
        """A returning face can correct an earlier ID without conflating co-visible faces."""
        for same_subject in (True, False):
            with self.subTest(same_subject=same_subject):
                timeline = ActivityTimeline(20_000)
                cue = (9400, 11420)
                candidate = {"track": 6, "face": 4, "representative": {"time_ms": 9600},
                             "frames": 15, "positive_frames": 14, "mean_logit": .3,
                             "track_start_ms": 9000, "track_end_ms": 10000}
                timeline._span({"candidates": [candidate]}, cue, 0, {6: 5}, {4: 4}, {})
                following = {**candidate, "face": 7, "frames": 35, "positive_frames": 32,
                             "representative": {"time_ms": 10040}, "mean_logit": .5}
                if not same_subject:
                    timeline.frames[246] = [[1, .1, .2, .2, .2, 1, 3], [5, .6, .2, .2, .2, 1, 4]]
                timeline._span({"candidates": [following]}, cue, 0, {6: 5}, {7: 3}, {7: (3, .704)})
                rows = list(timeline.spans[cue[0]]["candidates"].values())
                if same_subject:
                    self.assertEqual([(row["face"], row["positive_frames"]) for row in rows], [(3, 46)])
                    self.assertEqual(rows[0]["faceEvidence"], "cross-window-face")
                else:
                    self.assertEqual([(row["face"], row["positive_frames"]) for row in rows], [(4, 14), (3, 32)])

    @patch("voice_studio.vision.video_revolver.ffmpeg")
    def test_people_count_request_controls_native_worker_and_original_clock(self, ffmpeg):
        worker = Mock()
        args = ("proxy.mp4", None, 0, "mov", 800, 6000, [], None, worker)
        with patch.dict("os.environ", {"VOICE_YUNET_SAMPLING": "dense"}, clear=True):
            raw_activity_window(*args, people_origin=("original.mp4", 10000))
            options = worker.request.call_args.args[0]
            self.assertEqual(options[options.index("--person-pilot-source") + 1], "original.mp4")
            self.assertEqual(options[options.index("--person-pilot-offset") + 1], 10800)
            raw_activity_window(*args, people_count=False)
            self.assertNotIn("--person-pilot-source", worker.request.call_args.args[0])
        with patch.dict("os.environ", {"VOICE_YUNET_SAMPLING": "adaptive"}):
            raw_activity_window(*args, people_count=True)
            self.assertIn("--person-pilot-count-only", worker.request.call_args.args[0])

    def test_person_observations_keep_source_times_and_discard_context(self):
        result = fixture(145, 19)
        result["editorTimeline"]["personPilot"] = {
            "status": "ok", "samples": [{"timeMs": 0, "count": 1}, {"timeMs": 1000, "count": 2}],
        }
        timeline = ActivityTimeline(10_000)
        timeline.add(result, 5000, 10000, 4200, {})
        people = timeline.data(10000)["editorTimeline"]["personPilot"]
        self.assertEqual(people["samples"], [{"timeMs": 5200, "count": 2}])
        self.assertEqual(people["windows"], [{"status": "ok", "startMs": 5000, "endMs": 10000}])

    def test_dynamic_core_and_boundary_context(self):
        self.assertEqual(list(activity_windows(7_000)), [(0, 7000, 0, 7000)])
        self.assertEqual(list(activity_windows(11_000))[-1], (10000, 11000, 9200, 11000))
        self.assertEqual(list(activity_windows(83_870))[-1], (80000, 83870, 79200, 83870))

    def test_adjacent_overlaps_share_one_track_and_no_artificial_cut(self):
        timeline = ActivityTimeline(10_000)
        self.assertEqual(timeline.add(fixture(145, 4), 0, 5000, 0, {}), 5000)
        self.assertEqual(timeline.add(fixture(145, 19), 5000, 10000, 4200, {}), 10000)
        report = timeline.data(10000)
        self.assertEqual(report["timeline"]["frames"][124][0], report["timeline"]["frames"][125][0])
        self.assertEqual(report["faceTracking"]["faces"], 1)
        self.assertEqual(report["editorTimeline"]["shots"], [{"startMs": 0}])

    @patch("voice_studio.vision.video_revolver.raw_activity_window")
    def test_failed_middle_window_does_not_discard_first_or_last(self, worker):
        worker.side_effect = [fixture(145, 1), ValueError("decode"), fixture(70, 1)]
        first = []
        events = []
        _, visual = score_visible_faces("video", "wav", 0, [(0, 12_000)], "mov",
                                        duration_ms=12_000, emit=events.append, publish=first.append)
        coverage = visual["activityCoverage"]["windows"]
        self.assertEqual([row["status"] for row in coverage], ["ok", "failed", "ok"])
        self.assertTrue(visual["timeline"]["frames"][299])
        self.assertEqual(visual["timeline"]["frames"][125], [])
        self.assertEqual(len(first[0]["activityCoverage"]["windows"]), 1)
        self.assertEqual(len(first[0]["timeline"]["frames"]), 125)
        self.assertEqual(events[-1]["chunk"], 3)

    def test_clipped_vote_preserves_full_cue_denominator(self):
        spans, original = window_cues([(4500, 6200)], 5000, 10000, 4200)
        self.assertEqual(spans, [{"start_ms": 800, "end_ms": 2000}])
        self.assertEqual(original[800], (4500, 6200))

    def test_only_final_sub_frame_eof_is_quantized(self):
        self.assertEqual(sampled_window_end(25_000, 28_137, 28_120, final=True), 28_137)
        self.assertEqual(sampled_window_end(25_000, 28_137, 28_080, final=True), 28_080)
        self.assertEqual(sampled_window_end(25_000, 28_137, 28_120, final=False), 28_120)
        self.assertEqual(sampled_window_end(25_000, 25_030, 25_000, final=True), 25_000)
