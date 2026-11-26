"""Check enrollment measurements, holdout isolation and durable TTS references."""

from array import array
from math import pi, sin
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.app.studio import Studio
from voice_studio.prosody.reference_script import reference_script
from voice_studio.synthesis.dub_voice_profiles import reference_report
from voice_studio.synthesis.prosody_alignment import RATE, write_pcm
from voice_studio.synthesis.reference_enrollment import check_references, delivery_comparison


def synthetic_take(path, hz=153, amplitude=4200, pause_ms=300):
    """Write two periodic sections with a known internal pause and no real voice."""
    part = array("h", (round(amplitude * sin(2 * pi * hz * index / RATE)) for index in range(RATE * 2)))
    if sys.byteorder != "little":
        part.byteswap()
    write_pcm(path, part.tobytes() + bytes(pause_ms * RATE // 500) + part.tobytes())


class VoiceEnrollmentTests(TestCase):
    def test_cleans_and_batches_before_scoring(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [{"id": key, "text": "Один пробный дубль.", "reference_prompt": prompt}
                    for key, prompt in (("style", "question"), ("neutral", "neutral-a"),
                                        ("duplicate", "neutral-a"), ("clipped", None))]
            for key, pitch, amp in (("style", 147, 4100), ("neutral", 161, 4200),
                                    ("duplicate", 161, 4200), ("clipped", 150, 32767)):
                synthetic_take(root / f"{key}.wav", pitch, amp)
            with patch("voice_studio.synthesis.reference_enrollment.extract_embeddings",
                       return_value={0: [[1, 0]], 1: [[.95, .1]]}) as encoder:
                report = check_references(rows, root)
            encoder.assert_called_once()
            self.assertEqual(set(encoder.call_args.args[1]), {0, 1})
            self.assertEqual(report["referenceIds"], ["neutral", "style"])
            self.assertEqual(report["takes"][2]["issues"], ["duplicate"])
            self.assertIn("clipping", report["takes"][3]["issues"])
            self.assertEqual(report["takes"][1]["pauseMs"], 300)
            self.assertGreater(report["pairs"][0]["cosine"], .98)

    def test_holdout_measures_duration_and_pause(self):
        with TemporaryDirectory() as raw:
            left, right = (Path(raw) / name for name in ("left.wav", "right.wav"))
            synthetic_take(left, pause_ms=320)
            synthetic_take(right, pause_ms=520)
            metrics = delivery_comparison(left, right)
            stable = delivery_comparison(left, left)
        self.assertEqual(metrics["pauseDifferenceMs"], 200)
        self.assertGreater(metrics["durationRatio"], 1)
        self.assertIsNone(metrics["pitchShapeSemitones"])
        self.assertAlmostEqual(stable["pitchShapeSemitones"], 0, places=1)

    def test_workspace_saves_prompt_ids_and_checks_holdout_isolation(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            studio = Studio(root / "workspace", demo=True)
            prompts = reference_script("ru")["prompts"]
            ids = []
            try:
                for index in (0, 1, len(prompts) - 1):
                    wav = root / f"input-{index}.wav"
                    synthetic_take(wav, hz=145 + index * 2)
                    prompt = prompts[index]
                    ids.append(studio.record("ru", prompt["text"], wav.read_bytes(), reference_prompt=prompt["id"]))
                text = prompts[0]["text"]
                with self.assertRaisesRegex(ValueError, "Расшифровка"):
                    studio.record("ru", text, b"", reference_prompt="question")
                with self.assertRaisesRegex(ValueError, "Только для проверки"):
                    studio.check_voice_profile("ru", [ids[-1]])
                with patch("voice_studio.synthesis.reference_enrollment.extract_embeddings",
                           return_value={0: [[.8, .1]], 1: [[1, .1]]}):
                    result = studio.check_voice_profile("ru", ids[:2], save=True, name="  Голос для интервью  ")
                version = result["profileVersion"]
                self.assertEqual(studio.state("ru")["profiles"][0]["version"], version)
                self.assertEqual(studio.list_dub_profiles("ru"), [{"version": version}])
                self.assertEqual(studio.list_dub_profiles("en"), [])
                legacy = studio.save_profile("ru", ids[:2])
                unmeasured, measured = studio.list_voice_catalog("ru")
                self.assertEqual(unmeasured["version"], legacy)
                self.assertTrue(unmeasured["synthesisReady"])
                self.assertFalse(unmeasured["dubbingReady"])
                self.assertIsNone(unmeasured["metrics"])
                self.assertEqual(measured["name"], "Голос для интервью")
                self.assertEqual(measured["previewId"], result["referenceIds"][0])
                self.assertTrue(measured["dubbingReady"])
                self.assertEqual(measured["metrics"]["pairCount"], 1)
                self.assertEqual(measured["metrics"]["medianCosine"], result["pairs"][0]["cosine"])
                self.assertEqual(studio.list_voice_catalog("en"), [])
                with self.assertRaisesRegex(ValueError, "Неизвестная версия"):
                    studio.rename_voice_profile("en", version, "Other language")
                studio.rename_voice_profile("ru", version, "Интервью")
                self.assertEqual(studio.list_voice_catalog("ru")[1]["name"], "Интервью")
                self.assertEqual(studio.state("ru")["voiceCatalog"][1]["version"], version)
                copied = studio.copy_dub_prompts("ru", {"7": version}, root)
                first = studio.references("ru", [result["referenceIds"][0]])[0]
                self.assertEqual(copied[7]["text"], first["text"])
                self.assertEqual(reference_report(copied), {7: {"profileVersion": version, "recordedLanguage": "ru"}})
                self.assertTrue(Path(copied[7]["path"]).is_file())
            finally:
                studio.worker.shutdown()
