"""Check subtitle formatting and imperfect Whisper timestamps."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio.speech.media_transcript import segments_from_whisper, srt_time, transcribe_media, transcribe_stream, transcribe_wav


class MediaTranscriptTests(TestCase):
    def test_clamps_and_keeps_media_clock_order(self):
        raw = [{"text": "  один  ", "offsets": {"from": 300, "to": 1000}},
               {"text": "два", "offsets": {"from": 900, "to": 1700}},
               {"text": "три", "offsets": {"from": 2000, "to": 2300}},
               {"text": "", "offsets": {"from": 0, "to": 100}}]
        self.assertEqual(segments_from_whisper(raw, 1500), [
            {"text": "один", "startMs": 300, "endMs": 1000},
            {"text": "два", "startMs": 1000, "endMs": 1500},
        ])
        self.assertEqual(srt_time(3723004), "01:02:03,004")

    def test_media_offset_is_added_to_whisper_clock(self):
        raw = [{"text": "фраза", "offsets": {"from": 0, "to": 800}}]
        self.assertEqual(segments_from_whisper(raw, 3000, 1900), [
            {"text": "фраза", "startMs": 1900, "endMs": 2700},
        ])

    def test_silent_chunk_preserves_duration_without_invoking_whisper(self):
        with TemporaryDirectory() as folder, patch('voice_studio.speech.media_transcript.recognize_file') as recognize:
            work = Path(folder)
            wav = work / "silent.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(32000))
            result = transcribe_wav(wav, work, "ru", allow_empty=True)
            self.assertEqual((result["durationMs"], result["segments"], result["srt"]), (1000, [], ""))
            recognize.assert_not_called()

    def test_auto_uses_detected_language_and_preserves_foreign_terms(self):
        """Read RU or EN from native Whisper metadata without rewriting the text."""
        with TemporaryDirectory() as folder:
            work = Path(folder)
            wav = work / "sample.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(32000))
            for language, text in [("ru", "Проверим GitHub API."), ("en", "Check the GitHub API.")]:
                with self.subTest(language=language), patch('voice_studio.speech.media_transcript.recognize_result', return_value={
                    "result": {"language": language}, "transcription": [
                        {"text": text, "offsets": {"from": 0, "to": 600}},
                    ],
                }) as recognize:
                    result = transcribe_wav(wav, work, "auto")
                    self.assertEqual((result["language"], result["languageMode"]), (language, "auto"))
                    self.assertEqual(result["text"], text)
                    self.assertIn(text, result["srt"])
                    self.assertEqual(recognize.call_args.args[2], "auto")

    def test_uploaded_clip_languages_extend_auto_without_changing_capture(self):
        """Five additional scripts reach timed subtitles through the clip language policy."""
        examples = {"ar": "مرحبًا", "ur": "سلام", "bn": "নমস্কার", "hi": "नमस्ते", "ps": "پښتو"}
        with TemporaryDirectory() as folder:
            work = Path(folder)
            wav = work / "sample.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(32000))
            for language, text in examples.items():
                raw = [{"text": text, "offsets": {"from": 0, "to": 600}}]
                with self.subTest(language=language), patch(
                    "voice_studio.speech.media_transcript.recognize_result",
                    return_value={"result": {"language": language}, "transcription": raw},
                ):
                    result = transcribe_wav(wav, work, "auto", clip_languages=True)
                    self.assertEqual((result["language"], result["languageMode"]), (language, "auto"))
                    self.assertIn(text, result["srt"])
                    with self.assertRaisesRegex(ValueError, "русский или английский"):
                        transcribe_wav(wav, work, "auto")

    def test_browser_pcm_skips_decode_and_truncated_wav_does_not(self):
        """The direct recognizer receives only complete WAVs with the capture format."""
        with TemporaryDirectory() as folder:
            work = Path(folder)
            wav = work / "chunk"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(32000))
            with patch('voice_studio.speech.media_transcript.audio_offset', wraps=lambda source: 0) as probe, \
                    patch('voice_studio.speech.media_transcript.ffmpeg', side_effect=ValueError("bad WAV")):
                result = transcribe_media(work, "ru", wav, allow_empty=True)
                self.assertEqual(result["durationMs"], 1000)
                probe.assert_not_called()
                wav.write_bytes(wav.read_bytes()[:-1])
                with self.assertRaisesRegex(ValueError, "bad WAV"):
                    transcribe_media(work, "ru", wav, allow_empty=True)
                probe.assert_called_once_with(wav)

    def test_recovery_updates_the_initial_report_with_stable_media_timecodes(self):
        """Publish a stable prefix first, then finish speech at original source timestamps."""
        with TemporaryDirectory() as folder:
            work = Path(folder)
            wav = work / "sample.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(32000))
            first = {"startMs": 0, "endMs": 500, "text": "এখানে দেখুন"}
            second = {"startMs": 500, "endMs": 1000, "text": "Click Upload."}

            def recover(_wav, _work, _cues, lang, *, emit, on_prefix):
                """Simulate recovery that emits the first validated portion immediately."""
                self.assertEqual(lang, "bn")
                on_prefix([first])
                return [first, second], {"retried": [{"startMs": 0, "endMs": 1000}], "unresolved": []}

            events = []
            raw = [{"text": "మారిని మారిని మారిని మారిని మారిని మారిని",
                    "offsets": {"from": 0, "to": 1000}}]
            with patch("voice_studio.speech.media_transcript.ready", return_value=True), patch(
                "voice_studio.speech.media_transcript.recognize_file", return_value=raw,
            ), patch("voice_studio.speech.media_transcript.recover_clip", side_effect=recover):
                report = transcribe_wav(
                    wav, work, "bn", audio_start_ms=1200, quality=True, clip_languages=True,
                    publish=lambda phase, data: events.append((phase, data)),
                )
            self.assertEqual([phase for phase, _ in events], ["base", "speech"])
            self.assertEqual(events[0][1]["segments"], report["segments"][:1])
            self.assertEqual(report["segments"][1]["startMs"], 1700)
            self.assertEqual(report["recognitionRecovery"]["retried"], [{"startMs": 1200, "endMs": 2200}])

    def test_stream_rejects_unsupported_sources_and_durations(self):
        with patch('voice_studio.speech.media_transcript.ffmpeg') as ffmpeg:
            for url, seconds in (("file:///tmp/speech.m3u8", 30), ("rtsp://", 30),
                                 ("https://example.org/live.m3u8", 121)):
                with self.subTest(url=url, seconds=seconds), self.assertRaises(ValueError):
                    transcribe_stream(Path("."), "ru", url, seconds)
            ffmpeg.assert_not_called()
