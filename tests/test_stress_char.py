"""Verify that the portable stress model scores vowels without dictionary resources."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from voice_studio.prosody.stress_char_runtime import affix_features, candidates, load_model, predict, predict_eligible


class CharacterStressTests(unittest.TestCase):
    def test_weighted_affixes_rank_positions(self):
        """Stress can be inferred for unseen words from learned suffix weights."""
        model = {"kind": "affix", "width": 2, "weights": {"s:ик:1": 2, "s:ик:3": 0.5}}
        self.assertEqual(predict("Валик", model), (3, 1.5))
        self.assertEqual(predict("Братик", model), (4, 1.5))

    def test_unusable_words_abstain(self):
        """Tokens without a valid choice are outside the model's input scope."""
        model = {"kind": "affix", "width": 1, "weights": {}}
        for word in ("", "API", "дом", "по-другому", "а" * 65):
            self.assertIsNone(predict(word, model), word)

    def test_shared_model_inputs(self):
        """Only vowels become candidates, with dimensions stable across word lengths."""
        rows = candidates("котёнок")
        self.assertEqual([index for index, _, _ in rows], [1, 3, 5])
        self.assertTrue(all(len(chars) == 17 and len(numeric) == 6 for _, chars, numeric in rows))
        self.assertIn("s:ок:1", set(affix_features("котёнок", 5, 3)))

    def test_missing_and_malformed_runtime_file_fails_closed(self):
        """A damaged checkpoint cannot affect pronunciation analysis."""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.packed"
            with patch('voice_studio.prosody.stress_char_runtime.PACKED_PATH', path):
                load_model.cache_clear()
                self.assertIsNone(load_model())
                path.write_bytes(b"VSAFFIX1\x01")
                load_model.cache_clear()
                self.assertIsNone(load_model())
            load_model.cache_clear()

    def test_calibrated_margin(self):
        """Uncertain proposals remain out of production shadow results."""
        with patch('voice_studio.prosody.stress_char_runtime.load_model', return_value={"margin": 3.5}), patch(
            'voice_studio.prosody.stress_char_runtime.predict_loaded', return_value=(3, 3.49)
        ) as candidate:
            self.assertIsNone(predict_eligible("Валик"))
            candidate.return_value = (3, 3.5)
            self.assertEqual(predict_eligible("Валик"), (3, 3.5))
