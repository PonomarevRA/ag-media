"""Verify that on-screen name evidence is repeated and local before adjusting text."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.speech.media_transcript import transcribe_media
from voice_studio.vision.scene_transcript import reconcile_names
from voice_studio.app.transcript_editor import clean_cues


def cue():
    """Return a case with an emoji to exercise Python indices and UTF-16 person spans."""
    return {"startMs": 14200, "endMs": 15400, "text": "🎙 Блогер Максим Лучак.",
            "personMentions": [{"start": 10, "end": 22, "type": "PERSON"}]}


def frame(stamp, last="Лутчак"):
    """Return two aligned name lines from one OCR sample."""
    return {"timeMs": stamp, "text": [
        {"text": "Максим", "box": [.36, .43, .58, .15]},
        {"text": last, "box": [.36, .54, .56, .15]},
    ]}


class SceneTranscriptTests(TestCase):
    def test_updates_only_when_two_aligned_frames_agree(self):
        text = cue()
        scene = {"status": "ok", "frames": [frame(14520), frame(14920)]}
        self.assertEqual(reconcile_names([text], scene, "ru"), 1)
        self.assertEqual(text["text"], "🎙 Блогер Максим Лутчак.")
        self.assertEqual(text["asrCorrection"]["timestampsMs"], [14520, 14920])
        self.assertNotIn("personMentions", text)
        self.assertEqual(clean_cues([text])[0]["asrCorrection"], text["asrCorrection"])

    def test_respects_person_span_shot_time_and_competing_readings(self):
        for case in ("single", "duplicate", "different shot", "wrong time", "competing", "verbatim", "edited", "no name"):
            with self.subTest(case=case):
                text = cue()
                scene = {"status": "ok", "frames": [frame(14520), frame(14920)]}
                if case == "single":
                    scene["frames"] = [frame(14520)]
                if case == "duplicate":
                    scene["frames"] = [frame(14520), frame(14520)]
                if case == "different shot":
                    scene["shotStartsMs"] = [14800]
                if case == "wrong time":
                    scene["frames"] = [frame(15500), frame(15800)]
                if case == "competing":
                    scene["frames"].append(frame(15200, "Лущак"))
                if case == "verbatim":
                    scene["frames"].append(frame(15200, "Лучак"))
                if case == "edited":
                    text["asrText"] = "Другой исходный текст."
                if case == "no name":
                    text["personMentions"] = []
                self.assertEqual(reconcile_names([text], scene, "ru"), 0)
                self.assertEqual(text["text"], "🎙 Блогер Максим Лучак.")

    def test_surname_only_needs_a_large_title_line_with_stable_position(self):
        first = {"timeMs": 14500, "text": [{"text": "Лутчак", "box": [.37, .56, .51, .13]}]}
        second = {"timeMs": 14900, "text": [{"text": "Лутчак", "box": [.42, .48, .52, .14]}]}
        text = cue()
        scene = {"status": "ok", "frames": [first, second]}
        self.assertEqual(reconcile_names([text], scene, "ru"), 1)
        self.assertEqual(text["asrCorrection"]["timestampsMs"], [14500, 14900])
        second["text"][0]["box"][1] = .24
        self.assertEqual(reconcile_names([cue()], scene, "ru"), 0)

    def test_video_result_synchronizes_text_subtitles_and_mentions(self):
        baseline = {"segments": [cue()], "durationMs": 16000, "language": "ru"}
        scene = {"status": "ok", "frames": [frame(14520), frame(14920)]}
        with TemporaryDirectory() as folder, patch('voice_studio.speech.media_transcript.audio_offset', return_value=0), \
                patch('voice_studio.speech.media_transcript.ffmpeg'), patch('voice_studio.speech.media_transcript.transcribe_wav', return_value=baseline), \
                patch('voice_studio.speech.media_transcript.review_scene', return_value=scene), \
                patch('voice_studio.speech.media_transcript.annotate_person_mentions') as reannotate:
            source = Path(folder) / "mock.mp4"
            source.write_bytes(b"scene name reconciliation source fixture")
            result = transcribe_media(Path(folder), "ru", source, scene=True)
        self.assertEqual(result["text"], "🎙 Блогер Максим Лутчак.")
        self.assertIn("Максим Лутчак.", result["srt"])
        reannotate.assert_called_once_with(result["segments"], "ru")
