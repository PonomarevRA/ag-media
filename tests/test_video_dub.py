"""Verify per-speaker references and exact placement of dubbed speech."""

import json
from pathlib import Path
from shutil import which
from subprocess import TimeoutExpired, check_output
from tempfile import TemporaryDirectory
from time import monotonic
from unittest import TestCase, skipUnless
from unittest.mock import patch
import wave
from zipfile import ZipFile

from voice_studio.synthesis.dub_background import BACKGROUND_WAV, RATE as MIX_RATE
from voice_studio.synthesis.dub_timing import source_speech_window
from voice_studio.synthesis.dub_voice_profiles import reference_candidates, verified_clips
from voice_studio.synthesis.engines import ffmpeg
from voice_studio.synthesis.video_dub import DIALOGUE_WAV, RATE, build_video_dub, clean_cues, open_pcm, render_cue


def cue(start, end, number):
    """Construct one translated cue with explicit speaker and source clock."""
    return {"startMs": start, "endMs": end, "speaker": number,
            "sourceText": f"Исходная реплика {number}", "englishText": f"Test line {number}"}


class VideoDubTests(TestCase):
    def test_pronunciation_mode_is_validated_and_retained_through_retry(self):
        rows, _ = clean_cues({'sourceLanguage': 'ru', 'pronunciationMode': 'neutral',
                              'cues': [cue(100, 1000, 1)]}, 2000)
        modes = []

        def generated(engine, job, demo):
            modes.append(job['pronunciationMode'])
            if len(modes) == 1:
                raise TimeoutExpired(engine, 45)
            with open_pcm(job['output']) as audio:
                audio.writeframes(b'\x10\x01' * (RATE // 2))

        with TemporaryDirectory() as name, patch('voice_studio.synthesis.video_dub.generate', side_effect=generated):
            render_cue({**rows[0], 'tempo': 1}, 1, {1: {}}, (100, 1000), Path(name), monotonic() + 90)
        self.assertEqual(modes, ['neutral', 'neutral'])
        with self.assertRaises(ValueError):
            clean_cues({'sourceLanguage': 'ru', 'pronunciationMode': 'unknown', 'cues': [cue(0, 1000, 1)]}, 2000)

    def test_reference_selection_keeps_the_original_audio_offset_and_speaker(self):
        cues = [cue(20, 1250, 1), cue(1300, 2300, 1), cue(2600, 3650, 2)]
        selected = reference_candidates(cues, 100, 4000)
        self.assertEqual(selected, {1: [cues[1]], 2: [cues[2]]})
        with self.assertRaisesRegex(ValueError, "спикера 2"):
            reference_candidates(cues, 100, 3000)

    def test_voice_profile_uses_multiple_confirmed_cues_and_excludes_an_outlier(self):
        first, second, outlier = cue(200, 2300, 1), cue(3000, 4900, 1), cue(5000, 8000, 1)
        candidates = [outlier, first, second]
        vectors = {200: (1, 0), 3000: (.8, .6), 5000: (-1, 0)}
        self.assertEqual(verified_clips(candidates, vectors), ([first, second], .8))
        self.assertEqual(verified_clips(candidates, {}), ([outlier], None))

    def test_rejects_mixed_speech_and_overlapping_slots(self):
        base = {"sourceLanguage": "ru"}
        with self.assertRaisesRegex(ValueError, "перекрывающуюся речь"):
            clean_cues({**base, "cues": [{**cue(0, 900, 1), "overlap": True}]}, 3000)
        with self.assertRaisesRegex(ValueError, "пересечение"):
            clean_cues({**base, "cues": [cue(0, 1800, 1), cue(1500, 2600, 2)]}, 3000)
        with self.assertRaisesRegex(ValueError, "перевод"):
            clean_cues({**base, "cues": [{**cue(0, 900, 1), "englishText": ""}]}, 3000)
        for tempo in (True, 0.79, 1.81, float("nan")):
            with self.assertRaisesRegex(ValueError, "темп"):
                clean_cues({**base, "cues": [{**cue(0, 900, 1), "tempo": tempo}]}, 3000)

    def test_source_window_retains_the_media_clock_and_speech_padding(self):
        with TemporaryDirectory() as name:
            path = Path(name) / "offset.wav"
            with open_pcm(path) as audio:
                audio.writeframes(b"\x00\x00" * (RATE // 5) + b"\x10\x01" * (RATE * 3 // 5) +
                                  b"\x00\x00" * (RATE // 5))
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(source_speech_window(audio, 100, 1100, 100, RATE), (280, 920))

    def test_stalled_take_retries_with_a_new_seed(self):
        seeds = []

        def generate(engine, job, demo):
            seeds.append(job["seed"])
            if len(seeds) == 1:
                raise TimeoutExpired(engine, 45)
            with open_pcm(job["output"]) as audio:
                audio.writeframes(b"\x10\x01" * (RATE // 2))

        with TemporaryDirectory() as name, patch('voice_studio.synthesis.video_dub.generate', side_effect=generate):
            row = {**cue(100, 1000, 1), "tempo": 1}
            _, start, end, tempo, _, _ = render_cue(row, 1, {1: {}}, (100, 1000), Path(name), monotonic() + 90)
        self.assertEqual(seeds, [420, 1429])
        self.assertEqual((start, end, tempo), (100, 600, 1))

    @skipUnless(which("ffmpeg"), "Requires local FFmpeg")
    def test_overlong_composite_can_return_to_single_reference(self):
        attempts = []
        primary = {"text": "First. Second.", "fallback": {"text": "First."}}

        def generate(engine, job, demo):
            attempts.append((job["references"][0]["text"], job["seed"]))
            length = 2 * RATE if len(attempts) == 1 else RATE // 2
            with open_pcm(job["output"]) as audio:
                audio.writeframes(b"\x10\x01" * length)

        with TemporaryDirectory() as name, patch('voice_studio.synthesis.video_dub.generate', side_effect=generate):
            row = {**cue(100, 1000, 1), "tempo": 1}
            _, start, end, tempo, _, fallback = render_cue(
                row, 1, {1: primary}, (100, 1000), Path(name), monotonic() + 90,
            )
        self.assertEqual(attempts, [("First. Second.", 420), ("First.", 420)])
        self.assertEqual((start, end, tempo, fallback), (100, 600, 1, True))

    @skipUnless(which("ffmpeg") and which("ffprobe"), "Requires local FFmpeg and FFprobe")
    def test_archive_preserves_original_and_english_tracks_and_dub_timing(self):
        selected = []

        def fake_generate(engine, job, demo):
            """Write discernible fixed PCM so assertions can inspect the timeline."""
            selected.append((engine, job["language"], job["references"][0]["text"]))
            with open_pcm(job["output"]) as audio:
                audio.writeframes(b"\x10\x01" * (RATE // 4))

        def separated_bed(source, duration, folder):
            """Supply an audible continuous tone to test the real FFmpeg mixer."""
            background = folder / BACKGROUND_WAV
            ffmpeg(["-f", "lavfi", "-i", f"aevalsrc=0.05*sin(2*PI*110*t)|0.025*sin(2*PI*110*t):s={MIX_RATE}",
                    "-t", str(duration / 1000), "-c:a", "pcm_f32le", str(background)])
            return background

        with TemporaryDirectory() as name:
            folder = Path(name)
            source = folder / "source.mp4"
            ffmpeg(["-f", "lavfi", "-i", "color=size=32x32:rate=25:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:v", "libx264", "-c:a", "aac", str(source)])
            archive = folder / "dub.zip"
            cues = [cue(250, 1250, 1), {**cue(1500, 2600, 2), "tempo": 1}]
            with (patch('voice_studio.synthesis.video_dub.ready', return_value=True),
                  patch('voice_studio.synthesis.video_dub.generate', side_effect=fake_generate),
                  patch('voice_studio.synthesis.video_dub.prepare_background', side_effect=separated_bed)):
                report = build_video_dub(source, {"sourceLanguage": "ru", "cues": cues}, folder, archive)
            self.assertEqual(selected, [("qwen", "en", row["sourceText"]) for row in cues])
            self.assertEqual(report["durationMs"], 3000)
            first, second = report["cues"]
            self.assertEqual((first["tempo"], second["tempo"]), (0.8, 1))
            self.assertGreater(first["renderedEndMs"] - first["renderedStartMs"], first["rawDurationMs"])
            for row in report["cues"]:
                self.assertLessEqual(row["renderedEndMs"], row["endMs"])
            with wave.open(str(folder / DIALOGUE_WAV), "rb") as audio:
                self.assertEqual(audio.getnframes(), 3 * RATE)
                # atempo blends PCM, so check audibility separately from the untouched cue.
                audio.setpos(350 * RATE // 1000)
                self.assertGreater(int.from_bytes(audio.readframes(1), "little", signed=True), 200)
                for stamp, expected in ((0, b"\x00\x00"), (600, b"\x00\x00"),
                                        (1500, b"\x10\x01"), (1750, b"\x00\x00")):
                    audio.setpos(stamp * RATE // 1000)
                    self.assertEqual(audio.readframes(1), expected)
            with wave.open(str(folder / "english.wav"), "rb") as audio:
                self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getnframes()),
                                 (MIX_RATE, 2, 3 * MIX_RATE))
                for stamp in (103, 603, 1753, 2653):
                    audio.setpos(stamp * MIX_RATE // 1000)
                    self.assertNotEqual(audio.readframes(1), bytes(6))
            stats = json.loads(check_output(["ffprobe", "-v", "error", "-select_streams", "a",
                                              "-show_streams", "-of", "json", str(folder / "original-and-english.mp4")]))
            tracks = [row["tags"]["language"] for row in stats["streams"]]
            self.assertEqual(tracks, ["rus", "eng"])
            with ZipFile(archive) as package:
                self.assertCountEqual(package.namelist(),
                                      ["original-and-english.mp4", "english.wav", DIALOGUE_WAV,
                                       BACKGROUND_WAV, "english.srt", "dub-report.json"])
                self.assertIn("00:00:01,500 --> 00:00:01,750", package.read("english.srt").decode())
