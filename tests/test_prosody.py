"""Cover cautious annotations and compare real queued variants in demo mode."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.prosody.prosody import analyze
from voice_studio.prosody.revolver import review_prosody, synthesis_variants
from voice_studio.prosody.stress_context import load_context_model
from voice_studio.app.studio import Studio
from test_studio import sample


class ProsodyTests(TestCase):
    def test_invalid_context_checkpoint_leaves_preview_available(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "context.json"
            path.write_text('{"stage":2,"weights":{"белки":{"1":{}}}}', encoding="utf-8")
            load_context_model.cache_clear()
            try:
                with patch('voice_studio.prosody.stress_context.MODEL_PATH', path):
                    self.assertEqual(load_context_model(), {})
            finally:
                load_context_model.cache_clear()

    def test_shadow_context_uses_previous_word_and_keeps_synthesis_input(self):
        model = {"stage": 2, "threshold": 1, "weights": {
            "белки": {"1": {"previous:лесные": 2}, "4": {"previous:молочные": 2}}}}
        assets = ({}, {"белки": ["б+елки", "белк+и"]}, {})
        with (patch('voice_studio.prosody.prosody.resources', return_value=assets),
              patch('voice_studio.prosody.revolver.model_available', return_value=False),
              patch('voice_studio.prosody.revolver.load_context_model', return_value=model)):
            for previous, expected in [("лесные", "бе́лки"), ("молочные", "белки́")]:
                text = f"{previous} белки попали в кадр."
                result = review_prosody(text, "ru")
                self.assertEqual(result["markedText"], text)
                self.assertEqual(result["shadowStress"][0]["proposal"], expected)
                self.assertEqual(synthesis_variants(text, "ru", "f5", {"stress": True}),
                                 [("baseline", [(text, 0)])])
            self.assertEqual(review_prosody("Не лесные белки.", "ru")["shadowStress"], [])
            self.assertEqual(review_prosody("Лесные белки́.", "ru")["shadowStress"], [])

    def test_annotations_and_t9_preserve_source(self):
        assets = ({"замок": "з+амок"}, {"замок": ["з+амок", "зам+ок"]}, {})
        with patch('voice_studio.prosody.prosody.resources', return_value=assets):
            result = analyze("Для игры пять шеликов. Python и дверной замок.", "ru")
            self.assertEqual(result["corrections"][0]["replacement"], "шариков")
            self.assertIn("шеликов", result["markedText"])
            self.assertIn("замо́к", result["markedText"])
            self.assertEqual(result["terms"][0]["canonical"], "Python")
            self.assertFalse(analyze("Подпись Шеликов.", "ru")["corrections"])
            self.assertEqual(analyze("ша́риков", "ru")["markedText"], "ша́риков")
            self.assertEqual(analyze("Это замок.", "ru")["stress"][-1]["source"], "abstained")

    def test_variant_adapters(self):
        text = "Пять шариков. Есть ещё?"
        variants = synthesis_variants(text, "ru", "coqui", {"stress": True, "pauses": True})
        self.assertEqual([name for name, _ in variants], ["baseline", "pauses"])
        self.assertEqual(variants[1][1], [("Пять шариков.", 280), ("Есть ещё?", 0)])
        self.assertEqual(len(synthesis_variants("См. API v1.2", "ru", "f5", {"pauses": True})), 1)

    def test_shadow_batch_keeps_baseline_and_original_text(self):
        text = "Пять шариков. Есть ещё."
        with TemporaryDirectory() as folder:
            studio = Studio(Path(folder), demo=True)
            try:
                key = studio.record("ru", text, sample())
                jobs = studio.synthesize("ru", {"text": text, "models": ["f5"], "refs": [key],
                                               "comparison": key, "prosody": {
                                                   "stress": True, "pauses": True, "tone": "question"}})
                studio.worker.shutdown()
                rows = studio.rows("SELECT * FROM jobs ORDER BY rowid")
                self.assertEqual([r["variant"] for r in rows], ["baseline", "stress", "tone-question", "pauses"])
                self.assertEqual([r["baseline_id"] for r in rows], [None, jobs[0], jobs[0], jobs[0]])
                self.assertTrue(all(r["status"] == "done" and r["text"] == text for r in rows))
                self.assertEqual(len({r["seed"] for r in rows}), 1)
                self.assertAlmostEqual(rows[-1]["seconds"], 3.28, places=2)
                run = studio.tracking.client.get_run(rows[-1]["run_id"])
                self.assertEqual(run.data.params["prosody_variant"], "pauses")
            finally:
                studio.worker.shutdown()
