"""Verify selective restoration and fixed-clock behaviour of the optional clean track."""

from array import array
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch
import wave

from voice_studio.audio.voice_quality import measure_dialogue, restore_dialogue, restoration_filters
from voice_studio.synthesis.video_dub import RATE, open_pcm


class DubAudioQualityTests(TestCase):
    def test_speech_and_gaps_are_distinct_and_silence_does_not_trigger_denoise(self):
        spans = [(0, 2000), (3000, 5000)]
        levels = [-39] * 10 + [-29] * 10 + [float("-inf")] * 10 + [-29] * 20
        with (patch("voice_studio.audio.voice_quality.window_levels", return_value=levels),
              patch("voice_studio.audio.voice_quality.rms", return_value=-32),
              patch("voice_studio.audio.voice_quality.band_levels", return_value=[-33, -40, -58])):
            quality = measure_dialogue(Path("dialogue.wav"), spans)
            actions, _ = restoration_filters(quality, RATE)
            self.assertEqual(set(actions), {"equalizer", "quietSpeech"})
            self.assertIsNone(quality["steadyNoiseDbfs"])
            levels[20:30] = [-47] * 10
            quality = measure_dialogue(Path("dialogue.wav"), spans)
            actions, _ = restoration_filters(quality, RATE)
            self.assertEqual(actions["denoise"]["noiseFloorDbfs"], -47)

    def test_loud_balanced_dialogue_does_not_get_a_duplicate(self):
        quality = {"status": "measured", "quietSpeechDbfs": -26, "speechDbfs": -20,
                   "lowBandDb": -5, "midBandDb": -4, "presenceBandDb": -9, "steadyNoiseDbfs": None}
        self.assertEqual(restoration_filters(quality, RATE), ({}, []))
        actions, _ = restoration_filters(quality, RATE, ["denoise", "quiet-speech"])
        self.assertEqual(set(actions), {"denoise", "quietSpeech"})
        self.assertEqual(actions["denoise"]["basis"], "preset")

    @skipUnless(which("ffmpeg"), "FFmpeg is required")
    def test_fft_restoration_keeps_impulse_and_wave_length_on_the_clock(self):
        with TemporaryDirectory() as name:
            source, clean = (Path(name) / filename for filename in ("source.wav", "clean.wav"))
            samples = array("h", [0]) * (2 * RATE)
            samples[RATE] = 20_000
            with open_pcm(source) as output:
                output.writeframes(samples.tobytes())
            quality = {"status": "measured", "quietSpeechDbfs": -25, "speechDbfs": -20,
                       "lowBandDb": -5, "midBandDb": -4, "presenceBandDb": -9, "steadyNoiseDbfs": -45}
            with patch("voice_studio.audio.voice_quality.measure_dialogue", return_value=quality):
                result = restore_dialogue(source, clean, [(0, 2000)])
            self.assertEqual(result["status"], "enhanced")
            with wave.open(str(clean), "rb") as audio:
                self.assertEqual(audio.getnframes(), len(samples))
                restored = array("h", audio.readframes(audio.getnframes()))
            self.assertEqual(max(range(len(restored)), key=lambda index: abs(restored[index])), RATE)
