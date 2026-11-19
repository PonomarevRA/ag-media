"""Keep transcript inputs, progress and native output order across performance modes."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from voice_studio.speech.media_transcript import analyze_detail_layers, recognize_and_diarize
from voice_studio.speech.recognition import translate_clips


class RecognitionPerformanceTests(TestCase):
    def test_parallel_scene_reads_the_face_timeline_without_sharing_mutations(self):
        started = Event()
        evidence = {"timeline": {"frames": [[1]]}}

        def scene(source, work, captured, sampling, emit, publish_mosaic=None):
            self.assertIsNot(evidence, captured)
            self.assertIs(evidence["timeline"], captured["timeline"])
            started.set()
            return {"status": "ok"}

        def people(source, work, target, **kwargs):
            self.assertTrue(started.wait(timeout=3))
            target["personLayer"] = {"frames": [1]}

        with (patch.dict(os.environ, {"VOICE_VIDEO_PARALLEL": "1"}), patch('voice_studio.speech.media_transcript.os.cpu_count', return_value=8),
              patch('voice_studio.speech.media_transcript.collect_scene', side_effect=scene),
              patch('voice_studio.speech.media_transcript.attach_person_evidence', side_effect=people)):
            result = analyze_detail_layers(Path("source"), Path("work"), evidence, body=True, gaze=True,
                                           scene=True, object_sampling="basic", emit=lambda event: None)
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(evidence["personLayer"], {"frames": [1]})

    def test_small_and_sherpa_overlap_without_changing_either_input(self):
        ready = Event()
        turns = [{"startMs": 0, "endMs": 2000, "speaker": 1}]

        def diarize(wav, count):
            self.assertTrue(ready.wait(timeout=3), "Whisper must start before diarization completes")
            return turns

        def decode(wav, output, language, *, words, timeout, model):
            self.assertFalse(words)
            ready.set()
            return [{"text": "Строка"}]

        events = []
        with (patch.dict(os.environ, {"VOICE_ASR_PARALLEL": "1"}), patch('voice_studio.speech.media_transcript.os.cpu_count', return_value=8),
              patch('voice_studio.speech.media_transcript.detect_turns', side_effect=diarize),
              patch('voice_studio.speech.media_transcript.recognize_file', side_effect=decode)):
            result = recognize_and_diarize(Path("input.wav"), Path("output"), "ru", True, True, None, False,
                                           events.append)
        self.assertEqual(result, ([{"text": "Строка"}], "ru", turns, None))
        self.assertCountEqual([(event["stage"], event["state"]) for event in events],
                              [("diarization", "running"), ("diarization", "done"),
                               ("whisper", "running"), ("whisper", "done")])

    def test_translation_batch_keeps_file_order_and_defers_to_single_mode_by_default(self):
        def run(command, **kwargs):
            for index, flag in enumerate(command):
                if flag == "-of":
                    prefix = Path(command[index + 1])
                    prefix.with_suffix(".json").write_text(json.dumps({"transcription": [{"text": prefix.name}]}))

        with TemporaryDirectory() as folder:
            clips = [Path(folder) / name for name in ["second.wav", "first.wav"]]
            with patch('voice_studio.speech.recognition.ready', return_value=True), patch('voice_studio.speech.recognition.subprocess.run', side_effect=run) as call:
                with patch.dict(os.environ, {"VOICE_DUB_BATCH": "1"}):
                    self.assertEqual(translate_clips(clips, "ru"), ["second", "first"])
                self.assertEqual(call.call_count, 1)
                call.reset_mock()
                with patch.dict(os.environ, {"VOICE_DUB_BATCH": "0"}):
                    self.assertEqual(translate_clips(clips, "ru"), ["second", "first"])
                self.assertEqual(call.call_count, 2)
