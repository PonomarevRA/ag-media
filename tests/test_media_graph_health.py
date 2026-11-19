"""Distinguish script conflicts from allowed English terminology and unknown text quality."""

from unittest import TestCase

from voice_studio.graph.media_graph_health import script_check


class MediaGraphHealthTests(TestCase):
    def test_bengali_label_with_telugu_text_is_a_script_warning_not_an_accuracy_score(self):
        result = script_check("bn", ["మారిని " * 20])
        self.assertEqual(result["status"], "unexpected-script")
        self.assertEqual(result["expectedLetters"], 0)
        self.assertGreater(result["otherLetters"], 20)

    def test_russian_with_english_terms_and_short_text_are_not_false_script_failures(self):
        result = script_check("ru", ["Сегодня проверяем новый интерфейс настройки GPU и WebNN на локальной машине."])
        self.assertEqual(result["status"], "no-script-conflict")
        self.assertGreater(result["allowedLatinLetters"], 0)
        self.assertEqual(script_check("ru", ["Привет"])["status"], "not-enough-evidence")
