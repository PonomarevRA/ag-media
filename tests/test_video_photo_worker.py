"""Keep exact-source phrase photos available when the shared activity pass fails."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from voice_studio.vision.video_revolver import collect_evidence


class VideoPhotoWorkerTests(TestCase):
    def test_shared_process_reuses_photo_payload_and_emits_progress(self):
        events = []
        photos = {"0": {"photoTimeMs": 500, "photos": ["example"]}}
        report = {"stages": {}}
        phrases = [{"startMs": 0, "endMs": 1000}]
        with patch('voice_studio.vision.video_revolver.score_visible_faces', return_value=({}, {"phrasePhotos": photos})) as score, \
             patch('voice_studio.vision.video_revolver.run_worker') as worker, \
             patch('voice_studio.vision.video_revolver.LEGACY_PYTHON', Path(__file__)), \
             patch('voice_studio.vision.video_revolver.ASD_WEIGHTS', Path(__file__)):
            collect_evidence("video", "audio", phrases, 0, "mov", report, events.append)
        self.assertEqual(score.call_args.kwargs["photo_phrases"], [[0, 1000]])
        self.assertEqual(report["phrases"], photos)
        self.assertEqual(report["stages"], {"light-asd": "ok", "yunet": "ok", "facial-expressions": "unavailable"})
        self.assertTrue(events[-1]["reused"])
        worker.assert_not_called()

    def test_failed_activity_pass_falls_back_to_original_source_photos(self):
        report = {"stages": {}}
        phrases = [{"startMs": 0, "endMs": 1000}]
        with patch('voice_studio.vision.video_revolver.score_visible_faces', side_effect=ValueError), \
             patch('voice_studio.vision.video_revolver.run_worker', return_value={}) as worker:
            collect_evidence("video", "audio", phrases, 0, "mov", report, lambda event: None)
        self.assertEqual(worker.call_args.args[0], "voice_studio.speech.speaker_photos")
        self.assertEqual(worker.call_args.args[1][0], "video")
        self.assertEqual(report["stages"], {"light-asd": "failed", "yunet": "ok", "facial-expressions": "unavailable"})
