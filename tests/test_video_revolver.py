"""Exercise abstention and preservation of the audio transcript during video analysis."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio.speech.media_transcript import SMALL_MODEL, transcribe_wav
from voice_studio.vision.video_revolver import classify_span


class VideoRevolverTests(TestCase):
    def test_offscreen_speaker_keeps_audio_label(self):
        span = {"start_ms": 1000, "end_ms": 2000,
                "candidates": [{"frames": 25, "positive_frames": 0}]}
        observation = classify_span(span)
        self.assertEqual(observation["state"], "unconfirmed")

        with TemporaryDirectory() as folder, patch('voice_studio.speech.media_transcript.ready', return_value=True), \
                patch('voice_studio.speech.media_transcript.detect_turns', return_value=[
                    {"startMs": 0, "endMs": 3000, "speaker": 1}]), \
                patch('voice_studio.speech.media_transcript.recognize_file', return_value=[
                    {"text": "В кадре другой.", "offsets": {"from": 0, "to": 1000}},
                    {"text": "Говорю я.", "offsets": {"from": 1000, "to": 2000}}]) as recognize:
            work = Path(folder)
            wav = work / "reference.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(96000))
            result = transcribe_wav(wav, work, "ru", quality=True, speakers=True)
            self.assertEqual([(row["speaker"], row["text"]) for row in result["segments"]],
                             [(1, "В кадре другой."), (1, "Говорю я.")])
            self.assertEqual(recognize.call_args.kwargs["model"], SMALL_MODEL)
            self.assertFalse(recognize.call_args.kwargs["words"])

    def test_requires_one_active_track_for_most_of_phrase(self):
        track = {"track": 7, "frames": 25, "positive_frames": 20,
                 "representative": {"time_ms": 1200, "box": [.2, .2, .4, .4]}}
        span = {"start_ms": 1000, "end_ms": 2000, "candidates": [track]}
        self.assertEqual(classify_span(span)["state"], "visible-active")
        competing = {**track, "track": 8}
        self.assertEqual(classify_span({**span, "candidates": [track, competing]})["state"],
                         "partial-or-ambiguous")
        self.assertEqual(classify_span({**span, "end_ms": 2700})["state"], "partial-or-ambiguous")

    def test_verified_cut_keeps_dominant_phrase_but_abstains_on_concurrent_faces(self):
        first = {"track": 1, "face": 3, "frames": 31, "positive_frames": 29,
                 "track_start_ms": 34920, "track_end_ms": 36200, "representative": {"time_ms": 35000}}
        second = {"track": 2, "face": 7, "frames": 5, "positive_frames": 5,
                  "track_start_ms": 36200, "track_end_ms": 37200, "representative": {"time_ms": 36200}}
        span = {"start_ms": 34900, "end_ms": 36400, "candidates": [first, second]}
        self.assertEqual(classify_span(span)["state"], "partial-or-ambiguous")
        self.assertEqual(classify_span(span, [36200])["face"], 3)
        second["track_start_ms"] = 36000
        self.assertEqual(classify_span(span, [36200])["state"], "partial-or-ambiguous")
