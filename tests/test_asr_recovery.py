"""Keep selective clip recovery bounded and aligned to the source signal."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio.speech.asr_recovery import accepted_cues, recover_clip, retry_spans, window_plan
from voice_studio.speech.recognition import watch_whisper_windows


def cue(start, end, text):
    """Build a transcript phrase in the original PCM clock."""
    return {"startMs": start, "endMs": end, "text": text}


def report(text, start=0, end=10_000, language="bn"):
    """Build a minimal Whisper JSON output for one independent window."""
    return {"result": {"language": language},
            "transcription": [{"text": text, "offsets": {"from": start, "to": end}}]}


class ASRRecoveryTests(TestCase):
    def test_retries_decoder_loops_without_removing_isolated_english_or_native_text(self):
        rows = [cue(0, 1000, "গুগল Chrome খুলুন"), cue(1000, 2000, "Check Google Chrome."),
                cue(2000, 4000, "మారిని మారిని మారిని మారిని మారిని మారిని")]
        repeats = ("Here you can see passport photo maker", "Go down")
        rows.extend(cue(i * 2000, (i + 1) * 2000, repeats[i % 2]) for i in range(2, 8))
        self.assertEqual(retry_spans(rows, "bn"), ([(2000, 16000)], set(range(2, 9))))

    def test_windows_split_on_nearby_pause_and_are_bounded(self):
        windows = window_plan([(3000, 29000), (31000, 1000000)], [10720, 19000])
        self.assertEqual(windows[:3], [(3000, 10720), (10720, 19000), (19000, 29000)])
        self.assertLessEqual(len(windows), 12)
        self.assertLessEqual(sum(end - start for start, end in windows), 100000)
        self.assertTrue(all(0 < end - start <= 10000 for start, end in windows))
        tail = window_plan([(0, 23870)], [])
        self.assertTrue(all(5000 <= end - start <= 10000 for start, end in tail))
        self.assertEqual(tail[-1][1], 23870)

    def test_accepted_cues_keep_english_and_native_text_without_using_padded_timestamps(self):
        native = report("বাংলা লেখা এবং Passport Photo ছোট\ufffd")
        self.assertEqual(accepted_cues(native, 7000, 9000, "BENGALI", True),
                         [cue(7000, 9000, "বাংলা লেখা এবং Passport Photo")])
        english = report("Click the upload button.", language="en")
        self.assertEqual(accepted_cues(english, 7000, 9000, "BENGALI", True),
                         [cue(7000, 9000, "Click the upload button.")])
        hindi = report("नीचे चले आइए।", language="hi")
        self.assertEqual(accepted_cues(hindi, 7000, 9000, "BENGALI", True), [])

    def test_retry_preserves_neighboring_text_and_reports_unresolved_span(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "full.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(16000 * 2 * 5))
            clean = cue(0, 1000, "Look at this example")
            bad = cue(1000, 5000, "মোটামোটামোটামোটামোটামোটামোটামোটা")
            with patch("voice_studio.speech.asr_recovery.ready", return_value=True), patch(
                "voice_studio.speech.asr_recovery.run_whisper",
                return_value=[report("এখানে দেখুন", 0, 2200)],
            ) as decode:
                rows, evidence = recover_clip(wav, root, [clean, bad], "bn")
            self.assertEqual(rows, [clean, cue(1000, 3200, "এখানে দেখুন")])
            self.assertEqual(decode.call_args.args[1], "bn")
            self.assertEqual(evidence["unresolved"], [{"startMs": 3200, "endMs": 5000}])
            with wave.open(str(root / "asr-retry-0.wav")) as excerpt:
                self.assertEqual(excerpt.getnframes(), 64000)

    def test_subscribers_see_validated_prefix_before_remaining_windows(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "full.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(16000 * 2 * 24))
            bad = cue(0, 24000, "మారిని మారిని మారిని మారిని మారిని మారిని")
            order = []
            prefix = []

            def decode(jobs, *args, on_window):
                """Record when a retry starts and return valid independent outputs."""
                order.append(("decode", len(jobs)))
                return [report("বাংলা এবং English") for _ in jobs]

            def publish(cues):
                """Capture only fully decoded cues available at the first boundary."""
                order.append(("publish", len(cues)))
                prefix.extend(cues)

            with patch("voice_studio.speech.asr_recovery.ready", return_value=True), patch(
                "voice_studio.speech.asr_recovery.run_whisper", side_effect=decode,
            ):
                rows, _ = recover_clip(wav, root, [bad], "bn", on_prefix=publish)
            self.assertEqual(order, [("decode", 2), ("publish", 2), ("decode", 1)])
            self.assertEqual(prefix, rows[:2])

    def test_live_run_keeps_ready_window_when_later_input_times_out(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = [(root / f"{i}.wav", root / f"{i}") for i in range(3)]
            ready_file = jobs[0][1].with_suffix(".json")
            incomplete = jobs[1][1].with_suffix(".json")
            script = (
                "import json, sys, time; from pathlib import Path; "
                "Path(sys.argv[1]).write_text(json.dumps({'transcription': []})); "
                "Path(sys.argv[2]).write_text('{'); time.sleep(60)"
            )
            progress = []
            result = watch_whisper_windows([sys.executable, "-c", script, str(ready_file), str(incomplete)],
                                          jobs, 1.5, progress.append)
            self.assertEqual(result, [{"transcription": []}, None, None])
            self.assertEqual(progress, [1])

    def test_recovery_emits_measured_counts_and_preserves_unfinished_ranges(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "full.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(16000 * 2 * 16))
            bad = cue(0, 16000, "మారిని మారిని మారిని మారిని మారిని మారిని")
            events = []

            def decode(jobs, *args, on_window):
                """Announce one readable first window and leave the second unfinished."""
                self.assertEqual(len(jobs), 2)
                on_window(1)
                return [report("এখানে দেখুন"), None]

            with patch("voice_studio.speech.asr_recovery.ready", return_value=True), patch(
                "voice_studio.speech.asr_recovery.run_whisper", side_effect=decode,
            ):
                rows, diagnostics = recover_clip(wav, root, [bad], "bn", emit=events.append)
            self.assertEqual(rows, [cue(0, 8000, "এখানে দেখুন")])
            self.assertEqual(diagnostics["unresolved"], [{"startMs": 8000, "endMs": 16000}])
            counted = [event for event in events if "windowsDone" in event]
            self.assertEqual([(event["state"], event["windowsDone"], event["windowsTotal"])
                              for event in counted], [("running", 0, 2), ("running", 1, 2), ("partial", 1, 2)])
            self.assertEqual(diagnostics['status'], 'partial')
            self.assertEqual((diagnostics['windowsDone'], diagnostics['windowsTotal']), (1, 2))

    def test_failed_alternate_decode_is_not_a_completed_attempt(self):
        """A finished primary process cannot conceal a timed-out alternate-language retry."""
        from voice_studio.speech.asr_recovery import review_windows
        health = {}
        jobs = [(Path('input.wav'), Path('transcript'))]
        outputs = iter([[{'transcription': []}], []])
        with patch('voice_studio.speech.asr_recovery.decode_windows',
                   side_effect=lambda batch, *args: next(outputs) if batch else []):
            _, done = review_windows(jobs, [(0, 1000)], 'bn', Path('model.bin'), True, True,
                                     None, lambda event: None, health=health)
        self.assertEqual(done, 1)
        self.assertTrue(health['incomplete'])
