"""Exercise UTF-16 spans and the optional local person-model adapter."""

import json
from pathlib import Path
import subprocess
from unittest import TestCase
from unittest.mock import patch

from voice_studio.speech.media_transcript import transcript_result
from voice_studio.speech.person_mentions import annotate_person_mentions, clean_mentions


class PersonMentionTests(TestCase):
    def test_dialogue_acknowledgements_are_not_people_on_import(self):
        for text in ["Что такое", "Понял", "Понятно.", "Добрый день"]:
            with self.subTest(text=text):
                span = {"start": 0, "end": len(text), "type": "PERSON"}
                cue = {"text": text, "personMentions": [span]}
                annotate_person_mentions([cue], "ru")
                self.assertEqual(cue["personMentions"], [])
                self.assertEqual(cue["text"], text)

    def test_invalid_spans_cannot_split_surrogate_pairs_or_overlap(self):
        text = "🎙 Иван Петров"
        candidates = [{"start": 1, "end": 5}, {"start": 3, "end": 14}, {"start": 7, "end": 14}]
        self.assertEqual(clean_mentions(text, candidates), [{"start": 3, "end": 14, "type": "PERSON"}])

    def test_generation_phrase_keeps_the_word_unmarked_but_a_full_name_is_marked(self):
        text = "У Зумеров всё иначе. Иван Зумеров ответил."
        spans = [{"start": 2, "end": 9}, {"start": 21, "end": 33}]
        self.assertEqual(clean_mentions(text, spans), [{"start": 21, "end": 33, "type": "PERSON"}])

    @patch.object(Path, "is_file", return_value=True)
    @patch('voice_studio.speech.person_mentions.subprocess.run')
    def test_first_last_and_patronymic_are_annotations_on_the_original_text(self, run, _):
        text = "Иван Петрович Иванов ответил."
        span = {"start": 0, "end": 20}
        run.return_value.stdout = json.dumps({"rows": [[span]]})
        cue = {"startMs": 0, "endMs": 500, "text": text}
        result = transcript_result([cue], 500, "ru")
        self.assertEqual(result["segments"][0]["personMentions"], [{**span, "type": "PERSON"}])
        self.assertEqual(result["text"], text)
        self.assertTrue(result["srt"].endswith(text + "\n"))
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), [text])
        annotate_person_mentions([cue], "ru")
        run.assert_called_once()

    @patch.object(Path, "is_file", return_value=True)
    @patch('voice_studio.speech.person_mentions.subprocess.run', side_effect=subprocess.TimeoutExpired("dotnet", 5))
    def test_model_failure_preserves_usable_transcript(self, _, __):
        cue = {"startMs": 0, "endMs": 500, "text": "Готово."}
        with self.assertLogs('voice_studio.speech.person_mentions', level="WARNING"):
            result = transcript_result([cue], 500, "ru")
        self.assertEqual(result["text"], "Готово.")
        self.assertNotIn("personMentions", cue)

    @patch('voice_studio.speech.person_mentions.subprocess.run')
    def test_english_cues_are_not_sent_to_the_russian_detector(self, run):
        cue = {"text": "John Smith"}
        annotate_person_mentions([cue], "en")
        run.assert_not_called()
