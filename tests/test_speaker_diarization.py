"""Verify speaker assignment, ambiguous boundaries and subtitle formatting."""

from unittest import TestCase

from voice_studio.speech.media_transcript import transcript_result
from voice_studio.speech.speaker_diarization import label_phrases, label_words, speaker_for


class SpeakerTests(TestCase):
    def test_returning_speaker_and_srt(self):
        words = [
            {"startMs": start, "endMs": start + 400, "text": text}
            for start, text in [(0, "Привет"), (440, "всем!"), (1400, "Слушаю."), (2400, "Спасибо.")]
        ]
        turns = [
            {"startMs": 0, "endMs": 900, "speaker": 1},
            {"startMs": 1200, "endMs": 1900, "speaker": 2},
            {"startMs": 2200, "endMs": 2900, "speaker": 1},
        ]
        cues = label_words(words, turns)
        self.assertEqual([cue["speaker"] for cue in cues], [1, 2, 1])
        self.assertEqual(cues[0]["text"], "Привет всем!")
        result = transcript_result(cues, 3000, "ru")
        self.assertIn("00:00:01,400 --> 00:00:01,800\nСпикер 2: Слушаю.", result["srt"])
        self.assertEqual(result["text"], "Привет всем! Слушаю. Спасибо.")

    def test_overlap_differs_from_adjacent_turns(self):
        first = {"startMs": 0, "endMs": 500, "speaker": 1}
        adjacent = {"startMs": 500, "endMs": 1000, "speaker": 2}
        overlapping = {**adjacent, "startMs": 300}
        self.assertEqual(speaker_for([first, adjacent], 300, 700), (None, False))
        self.assertEqual(speaker_for([first, overlapping], 300, 700), (2, True))
        self.assertEqual(speaker_for([first, adjacent], 1100, 1500), (None, False))


    def test_recover_voice_in_a_padded_phrase_but_leave_competing_turns_unknown(self):
        phrase = {"startMs": 0, "endMs": 4000, "text": "One isolated phrase."}
        turn = {"startMs": 1950, "endMs": 3100, "speaker": 2}
        cue = label_phrases([phrase], [turn])[0]
        self.assertEqual(cue["speaker"], 2)
        self.assertEqual(cue["speakerBasis"], "single-voiced-turn")
        self.assertEqual(cue["speakerAudioWindow"], {"startMs": 1950, "endMs": 3100})
        competing = {"startMs": 100, "endMs": 840, "speaker": 1}
        self.assertIsNone(label_phrases([phrase], [competing, turn])[0]["speaker"])
        self.assertIsNone(label_phrases([{**phrase, "text": "- Hi. - Hello."}], [turn])[0]["speaker"])
