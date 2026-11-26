"""Exercise offline translation status and the ordered cue contract."""

from unittest import TestCase
from unittest.mock import patch

from voice_studio.mods import ModExecutionError
from voice_studio.synthesis.dub_text_translation import translate_text, translation_status

MODULE = "voice_studio.synthesis.dub_text_translation"
BATCH = {"sourceLanguage": "en", "targetLanguage": "ru", "texts": ["First line.", "Second line."]}


class DubTextTranslationTests(TestCase):
    def test_translations_are_forwarded_in_cue_order(self):
        with patch(MODULE + ".request_translation",
                   return_value={"status": "ready", "translations": ["Первая строка.", "Вторая строка."]}) as native:
            self.assertEqual(translate_text(BATCH), {"translations": ["Первая строка.", "Вторая строка."]})
            native.assert_called_once_with(BATCH)

    def test_missing_language_has_a_system_settings_hint(self):
        with patch(MODULE + ".request_translation", return_value={"status": "needs-languages"}) as native:
            status = translation_status("en", "ru")
            self.assertEqual(status["status"], "needs-languages")
            self.assertIn("Языки перевода", status["message"])
            self.assertEqual(native.call_args.args[0]["texts"], [])
            with self.assertRaisesRegex(ValueError, "Языки перевода"):
                translate_text(BATCH)

    def test_incomplete_batch_and_changed_provider_are_rejected(self):
        with patch(MODULE + ".request_translation",
                   return_value={"status": "ready", "translations": ["Первая строка."]}) as native:
            with self.assertRaisesRegex(ModExecutionError, "exactly one"):
                translate_text(BATCH)
            with self.assertRaisesRegex(ValueError, "Мод перевода изменился"):
                translate_text({**BATCH, "provider": "example.changed"})
            self.assertEqual(native.call_count, 1)
