"""Protect mixed audio detections and pitch contours from timing regressions."""

from array import array
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio.core.analysis_stages import ignore_progress
from voice_studio.audio.audio_scene import analyze_audio_scene, select_sound_classifier, sound_spans
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.speech.media_transcript import transcribe_wav
from voice_studio.audio.speech_intonation import measure_intonation
from voice_studio.audio.zipformer_audio import BASIS as ZIPFORMER_BASIS


class AudioSceneTests(TestCase):
    def test_auto_selects_local_platform_backend(self):
        """Keep Apple on macOS and select Zipformer for other platforms."""
        with patch.dict("os.environ", {"VOICE_AUDIO_CLASSIFIER": "auto"}):
            for platform, expected in (("darwin", "apple"), ("linux", "zipformer"), ("win32", "zipformer")):
                with patch('voice_studio.audio.audio_scene.sys.platform', platform):
                    self.assertEqual(select_sound_classifier(), expected)
                    self.assertEqual(select_sound_classifier("zipformer"), "zipformer")

    def test_zipformer_music_seed_and_provenance(self):
        """Reject a weak isolated music tag and keep Zipformer provenance."""
        rows = [{"start": 0, "end": 1.5, "scores": [{"label": "music", "score": .82}]},
                {"start": .75, "end": 2.25, "scores": []},
                {"start": 1.5, "end": 3, "scores": [{"label": "cricket_chirp", "score": .30}]}]
        layers = sound_spans(rows, 3000, ZIPFORMER_BASIS)
        self.assertEqual([row["label"] for row in layers], ["cricket_chirp"])
        graph = build_graph({"segments": [], "audioLayers": layers, "language": "en"},
                            kind="audio", duration_ms=3000)
        node = next(node for node in graph["nodes"] if node["kind"] == "audio_background")
        self.assertIn(ZIPFORMER_BASIS, str(node))

    def test_music_does_not_hide_a_concurrent_sound_or_depend_on_speech_rank(self):
        """Classify music and the cricket in the same windows under dominant speech."""
        rows = [
            {"start": start, "end": start + 1.5, "scores": [
                {"label": "speech", "score": .9}, {"label": "music", "score": score},
                {"label": "cricket_chirp", "score": cricket}]}
            for start, score, cricket in [(0, .28, 0), (.75, .51, 0), (1.5, .66, .57), (2.25, .55, .58)]
        ]
        music, sound = sound_spans(rows, 3800)
        self.assertEqual((music["startMs"], music["endMs"], music["label"]), (1125, 3750, "music"))
        self.assertEqual((sound["startMs"], sound["endMs"], sound["label"]), (1875, 3750, "cricket_chirp"))

    def test_pitch_direction_uses_pcm_offset_and_skips_overlapping_voices(self):
        """Compare ascending and descending reference sweeps on the source clock."""
        rate = 16000
        cues = [{"startMs": 900, "endMs": 1900}, {"startMs": 1900, "endMs": 2900}]
        with TemporaryDirectory() as folder:
            wav = Path(folder) / "contours.wav"
            samples = array("h")
            phase = 0
            for position in range(rate * 2):
                seconds = position / rate
                hz = 120 + 70 * (1 - abs(1 - seconds))
                phase += math.tau * hz / rate
                samples.append(round(12000 * math.sin(phase)))
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(samples.tobytes())
            pitch = measure_intonation(wav, cues, 900)
            self.assertEqual([(r["startMs"], r["label"]) for r in pitch], [(900, "rising"), (1900, "falling")])
            self.assertEqual(measure_intonation(wav, cues, 900, [cues[0]]), pitch[1:])
            graph = build_graph({"segments": [{**cue, "text": "Contour"} for cue in cues],
                                 "audioLayers": pitch, "language": "en"}, kind="audio", duration_ms=2900)
            self.assertEqual(sum(edge[1] == "intonation_of" for edge in graph["edges"]), 2)

    def test_non_speech_files_still_return_their_audio_layer(self):
        """Keep sound-classification data even when Whisper returns no cues."""
        with TemporaryDirectory() as folder:
            work = Path(folder)
            wav = work / "sound.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(16000 * 2 * 2))
            windows = [{"start": 0, "end": 1.5, "scores": [{"label": "music", "score": .86}]}]
            with patch('voice_studio.speech.media_transcript.recognize_file', return_value=[]), \
                    patch('voice_studio.audio.audio_scene.classify', return_value=(windows, "done")):
                result = transcribe_wav(wav, work, "en", 250, audio_layers=True, audio_classifier="apple")
            self.assertEqual(result["segments"], [])
            self.assertEqual([(r["label"], r["startMs"]) for r in result["audioLayers"]], [("music", 250)])
            self.assertEqual(analyze_audio_scene(work / "missing.wav", [], 0, [], ignore_progress)["soundAnalysisStatus"],
                             "failed")
