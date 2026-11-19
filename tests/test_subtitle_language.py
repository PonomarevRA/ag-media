"""Verify localized SRT and MP4 overlay labels without rewriting speech."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.speech.media_transcript import render_srt
from voice_studio.vision.video_export import write_overlay


class SubtitleLanguageTests(TestCase):
    def test_srt_follows_speech_locale_and_preserves_the_cue(self):
        cues = [{"startMs": 0, "endMs": 1200, "speaker": 2, "text": "Hello, Maya!"}]
        self.assertIn("Speaker 2: Hello, Maya!", render_srt(cues, "en"))
        self.assertIn("Спикер 2: Hello, Maya!", render_srt(cues, "ru"))

    def test_mp4_ass_uses_english_speaker_and_shot_labels(self):
        layer = {"segments": [{"startMs": 0, "endMs": 1200, "number": 2, "text": "Hello, Maya!"}], "shots": [0]}
        with TemporaryDirectory() as directory:
            file = Path(directory) / "overlay.ass"
            write_overlay(layer, file, 640, 360, "en")
            labels = file.read_text()
        self.assertIn("Speaker 2 · Hello, Maya!", labels)
        self.assertIn("Shot 1", labels)
        self.assertNotIn("Спикер", labels)
