"""Verify repeat batches, source integrity, and score persistence."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

from voice_studio.synthesis.speech_metrics import word_error_rate
from voice_studio.app.studio import Studio
from test_studio import sample


class ComparisonTests(TestCase):
    def test_repeats_share_text_and_keep_seeds_and_scores(self):
        with TemporaryDirectory() as folder:
            studio = Studio(Path(folder), demo=False)
            try:
                text = "Ёлки растут в саду."
                original = studio.record("ru", text, sample())
                with patch('voice_studio.app.studio.availability', return_value={"coqui": {"ready": True}}):
                    data = {"text": text, "refs": [original], "models": ["coqui"],
                            "comparison": original, "repeats": 2}
                    with self.assertRaises(ValueError):
                        studio.synthesize("ru", {**data, "text": "Новый текст."})
                    with self.assertRaises(ValueError):
                        studio.synthesize("ru", {**data, "repeats": 6})
                    with patch('voice_studio.app.studio.generate', side_effect=lambda _, job, __: Path(job["output"]).write_bytes(sample())), \
                            patch('voice_studio.app.studio.measure', return_value={"whisper_word_error_rate": 0.25,
                                                                 "xtts_encoder_cosine": 0.72}):
                        ids = studio.synthesize("ru", data)
                        replay = studio.synthesize("ru", data)
                        deadline = monotonic() + 15
                        while monotonic() < deadline:
                            jobs = studio.rows("SELECT * FROM jobs ORDER BY rowid")
                            if all(job["status"] in {"done", "failed"} for job in jobs):
                                break
                            sleep(.05)
                    self.assertEqual([job["id"] for job in jobs], ids + replay)
                    self.assertEqual(len({job["seed"] for job in jobs}), 2)
                    self.assertEqual([job["seed"] for job in jobs[:2]],
                                     [job["seed"] for job in jobs[2:]])
                    for job in jobs:
                        self.assertEqual(job["status"], "done")
                        self.assertEqual(job["comparison_id"], original)
                        run = studio.tracking.client.get_run(job["run_id"])
                        self.assertEqual(run.data.params["seed"], str(job["seed"]))
                        self.assertEqual(run.data.metrics["xtts_encoder_cosine"], 0.72)
                        self.assertEqual(job["word_error_rate"], 0.25)
            finally:
                studio.worker.shutdown()

    def test_word_error_rate_normalizes_orthography(self):
        self.assertEqual(word_error_rate("Ёлки, растут!", "елки растут"), 0)
        self.assertEqual(word_error_rate("один два три", "один три"), 1 / 3)
