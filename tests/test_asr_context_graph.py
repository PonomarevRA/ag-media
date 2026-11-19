"""Protect evidence links and abstention in local transcript suggestions."""

from unittest import TestCase

from voice_studio.speech.asr_context_graph import apply_corrections, corrections


class AsrContextTests(TestCase):
    def test_token_offsets_and_evidence(self):
        text = "🎙 На столе пять ШЕЛИКОВ, рядом шесть шериков."
        edits = corrections(text, "ru")
        self.assertEqual([item["word"] for item in edits], ["ШЕЛИКОВ", "шериков"])
        self.assertIn({"kind": "count", "word": "пять"}, edits[0]["evidence"])
        self.assertEqual(apply_corrections(text, edits), "🎙 На столе пять ШАРИКОВ, рядом шесть шариков.")

    def test_conflicting_and_missing_evidence_preserve_source(self):
        for text in ("Капитан Шеликов привёз пять предметов.",
                     "Подпись: Шеликов. Пять минут спустя.", "Шеликов позвонил.",
                     "В коробке пять кубиков. Потом пришёл Шериков."):
            with self.subTest(text=text):
                self.assertEqual(corrections(text, "ru"), [])
                self.assertEqual(len(corrections(text, "ru", contextual=False)), 1)

    def test_same_draft_from_other_language_is_preserved(self):
        self.assertFalse(corrections("пять шериков", "en"))
