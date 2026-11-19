"""Verify timeline boundaries and attribution for simultaneous speech."""

from unittest import TestCase

from voice_studio.speech.media_transcript import render_srt
from voice_studio.speech.speech_overlap import audio_overlaps, clip_overlaps, visual_overlaps


class SpeechOverlapTests(TestCase):
    def test_distinct_simultaneous_speakers_and_media_clock(self):
        turns = [
            {"startMs": 0, "endMs": 700, "speaker": 1},
            {"startMs": 200, "endMs": 600, "speaker": 1},
            {"startMs": 400, "endMs": 1000, "speaker": 2},
            {"startMs": 1000, "endMs": 1300, "speaker": 3},
        ]
        spans = audio_overlaps(turns, 100)
        self.assertEqual(spans, [{"startMs": 500, "endMs": 800, "speakers": [1, 2]}])
        cues = [{"startMs": 250, "endMs": 750, "speaker": 1, "text": "Подожди."}]
        clip_overlaps(cues, spans)
        self.assertEqual(cues[0]["overlaps"], [{"startMs": 500, "endMs": 750, "speakers": [1, 2]}])
        subtitles = render_srt(cues)
        self.assertEqual(subtitles.count("Подожди."), 1)
        self.assertIn("Голоса пересекаются · спикеры 1, 2", subtitles)
        self.assertIn("Спикер 2: Подожди.", render_srt([{**cues[0], "speakerNumber": 2}]))
        self.assertEqual(audio_overlaps([turns[2], turns[3]]), [])

    def test_visual_overlap_ignores_brief_coincidence(self):
        both = [[3, 0, 0, 1, 1, 1], [5, 0, 0, 1, 1, 1]]
        timeline = {"stepMs": 40, "frames": [both] * 2 + [[]] + [both] * 5}
        self.assertEqual(visual_overlaps(timeline), [{"startMs": 120, "endMs": 320, "tracks": [3, 5]}])
