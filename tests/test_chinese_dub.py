"""Verify Chinese source cues, offline script conversion and advisory OCR matches."""

from unittest import TestCase

from voice_studio.speech.ocr_checks import measure_ocr_checks
from voice_studio.speech.subtitle_script import simplified_subtitles
from voice_studio.synthesis.video_dub import AUDIO_LANGUAGES, clean_cues


class ChineseDubTests(TestCase):
    def test_chinese_source_accepts_reviewed_russian_cues(self):
        rows, language = clean_cues({"sourceLanguage": "zh", "dubLanguage": "ru", "cues": [
            {"startMs": 700, "endMs": 4730, "speaker": 1, "sourceText": "LIGO宣佈第二次探測到重力波",
             "targetText": "ЛИГО обнаружила гравитационные волны во второй раз."},
        ]}, 7000)
        self.assertEqual((language, AUDIO_LANGUAGES[language]), ("zh", "zho"))
        self.assertEqual(rows[0]["startMs"], 700)
        self.assertIn("探測", rows[0]["sourceText"])

    def test_traditional_srt_preserves_cue_numbers_timestamps_and_identifiers(self):
        clock = "00:00:00,700 --> 00:00:04,730"
        srt = f"1\r\n{clock}\r\nLIGO宣佈第二次探測到重力波 GW151226\r\n"
        result = simplified_subtitles({"language": "zh", "srt": srt})
        self.assertEqual(result["srt"], f"1\r\n{clock}\r\nLIGO宣布第二次探测到重力波 GW151226\r\n")
        with self.assertRaises(ValueError):
            simplified_subtitles({"language": "ru", "srt": srt})

    def test_ocr_checks_names_measurements_and_nearby_identifier_without_editing_cue(self):
        original = "GW15094，Virggo Collaboration，Benett Thomas，3.5厘米"
        cues = [{"startMs": 43_000, "endMs": 53_000, "text": original}]
        scene = {"status": "ok", "frames": [
            {"timeMs": 56_137, "text": [{"text": "GW150914"}, {"text": "3,6 cm"}, {"text": "Benett Thomass"}]},
            {"timeMs": 59_200, "text": [{"text": "GW150904"}]},
        ]}
        checks = measure_ocr_checks(cues, scene)
        self.assertEqual({(row["kind"], row["observed"]) for row in checks}, {
            ("identifier", "GW150914"), ("name", "Benett Thomass"), ("number", "3,6 cm"),
        })
        self.assertTrue(all(row["timeMs"] == 56_137 and row["status"] == "review" for row in checks))
        self.assertEqual(cues[0]["text"], original)
        scene["frames"][1]["timeMs"] = 57_500
        self.assertNotIn("identifier", {row["kind"] for row in measure_ocr_checks(cues, scene)})
