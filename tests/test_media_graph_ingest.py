"""Verify source-clock partitioning and identity boundaries for long graphs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_graph_ingest import (
    ID_BLOCK, MAX_DURATION_MS, SPEECH_WINDOW_MS, analyze_source, restore_imported_person_layer,
    probe_source, scan_shots, shifted_spans, spans,
)


class MediaGraphIngestTests(TestCase):
    def test_shifted_matroska_and_mp4_join_at_the_playback_origin(self):
        """Container durations have distinct time bases at a positive start."""
        with TemporaryDirectory() as root:
            source = Path(root) / "media"
            source.touch()
            for container, duration, expected in (
                ("matroska,webm", "13.008", 3015),
                ("mov,mp4,m4a,3gp,3g2,mj2", "3.023", 3023),
            ):
                probe = {
                    "format": {"format_name": container, "start_time": "9.993", "duration": duration},
                    "streams": [{"index": 0, "codec_type": "audio", "start_time": "9.993"}],
                }
                with patch('voice_studio.graph.media_graph_ingest.subprocess.check_output', return_value=json.dumps(probe).encode()):
                    result = probe_source(source)
                self.assertEqual(result["durationMs"], expected)
                self.assertEqual(result["streams"][0]["startMs"], 0)

    def test_two_hours_is_partitioned_contiguously(self):
        self.assertEqual(len(spans(MAX_DURATION_MS)), MAX_DURATION_MS // SPEECH_WINDOW_MS)
        self.assertEqual(spans(MAX_DURATION_MS)[-1], (MAX_DURATION_MS - SPEECH_WINDOW_MS, MAX_DURATION_MS))
        self.assertEqual(spans(SPEECH_WINDOW_MS + 475)[-1], (SPEECH_WINDOW_MS, SPEECH_WINDOW_MS + 475))

    def test_shift_clips_overlap_and_changes_all_local_identity_references(self):
        local = {"startMs": 400, "endMs": 1500, "speaker": 1, "linkedSpeakerNumber": 2,
                 "visualFace": 3, "visualSpeaker": 4,
                 "overlaps": [{"startMs": 500, "endMs": 1300, "speakers": [1, 2]}]}
        offset = SPEECH_WINDOW_MS
        shifted = shifted_spans([local], offset, offset + 700, ID_BLOCK)
        self.assertEqual(shifted, [
            {"startMs": offset + 400, "endMs": offset + 700, "speaker": ID_BLOCK + 1,
             "linkedSpeakerNumber": ID_BLOCK + 2, "visualFace": ID_BLOCK + 3, "visualSpeaker": ID_BLOCK + 4,
             "overlaps": [{"startMs": offset + 500, "endMs": offset + 700,
                           "speakers": [ID_BLOCK + 1, ID_BLOCK + 2]}]},
        ])
        self.assertEqual(local["overlaps"][0]["startMs"], 500)

    def test_window_scoped_speaker_ids_do_not_merge_and_samples_stay_bounded(self):
        end = SPEECH_WINDOW_MS + 3000
        metadata = {"kind": "video", "durationMs": end, "hasAudio": True,
                    "streams": [{"kind": "audio", "index": 0, "startMs": 0, "durationMs": end}]}

        def audio_window(source, work, start, stop, stream):
            return work / "fake.wav", 0, stop - start

        def speech(*args, **kwargs):
            return {"segments": [{"startMs": 1000, "endMs": 2000, "text": "Example", "speaker": 1}]}

        def video(source, work, window, analysis, visual, scene, *, object_sampling, revolver, people_count, emit, body=False, gaze=False):
            self.assertEqual(object_sampling, "every-third")
            timeline = {"stepMs": 40, "frames": [[[1, .1, .2, .3, .4, 1, 3]]] * 12, "estimatedFrames": [4]}
            counts = {"samples": [{"timeMs": 1000, "count": 2}], "windows": [{"startMs": 0, "endMs": 2000}]}
            return {"timeline": timeline, "faceSpeakers": {3: {"speaker": 1, "basis": "audio-visual"}},
                    "editorTimeline": {"personPilot": counts}}, {
                "status": "ok", "frames": [
                    {"timeMs": stamp, "shot": 0, "text": [], "objects": []}
                    for stamp in (1200, 62_000) if stamp < window[1] - window[0]
                ], "cameraMotion": [{"startMs": 100, "endMs": 420, "confidence": .9,
                                      "shiftX": 0, "shiftY": 0, "scale": 1}],
            }

        with TemporaryDirectory() as root, patch('voice_studio.graph.media_graph_ingest.probe_source', return_value=metadata), \
                patch('voice_studio.graph.media_graph_ingest.extract_speech', side_effect=audio_window), \
                patch('voice_studio.graph.media_graph_ingest.transcribe_wav', side_effect=speech), \
                patch('voice_studio.graph.media_graph_ingest.collect_visual', side_effect=video), \
                patch('voice_studio.graph.media_graph_ingest.analyze_audio_scene', return_value={"soundAnalysisStatus": "done",
                      "audioLayers": [{"startMs": 100, "endMs": 900, "kind": "music", "label": "music"}]}), \
                patch('voice_studio.graph.media_graph_ingest.scan_shots', return_value=[]):
            (Path(root) / 'test.mov').write_bytes(b'fixture-source')
            report = analyze_source(Path(root), Path(root) / "test.mov", "video", "en",
                                    object_sampling="every-third")
        self.assertEqual(report["speakerScope"], "window")
        self.assertEqual([row["startMs"] for row in report["audioLayers"]], [100, SPEECH_WINDOW_MS + 100])
        self.assertEqual([cue["speaker"] for cue in report["segments"]], [1, ID_BLOCK + 1])
        self.assertEqual([cue["speakerLocalNumber"] for cue in report["segments"]], [1, 1])
        self.assertEqual([cue["speakerWindowStartMs"] for cue in report["segments"]], [0, SPEECH_WINDOW_MS])
        self.assertEqual([cue["speakerProfileId"] for cue in report["segments"]],
                         [profile["id"] for profile in report["speakerProfileIndex"]])
        self.assertEqual(len({profile["id"] for profile in report["speakerProfileIndex"]}), 2)
        self.assertTrue(all(profile["scope"] == "window" for profile in report["speakerProfileIndex"]))
        self.assertEqual([cue["startMs"] for cue in report["segments"]], [1000, SPEECH_WINDOW_MS + 1000])
        self.assertEqual([sample["timeMs"] for sample in report["sceneEvidence"]["frames"]],
                         [1200, 62_000, SPEECH_WINDOW_MS + 1200])
        timeline = report["videoEvidence"]["timeline"]["frames"]
        self.assertFalse(timeline[500])
        self.assertEqual(timeline[SPEECH_WINDOW_MS // 40][0][-1], ID_BLOCK + 3)
        self.assertEqual(report["videoEvidence"]["timeline"]["estimatedFrames"], [4, SPEECH_WINDOW_MS // 40 + 4])
        self.assertEqual([sample["timeMs"] for sample in report["videoEvidence"]["editorTimeline"]["personPilot"]["samples"]],
                         [1000, SPEECH_WINDOW_MS + 1000])
        self.assertEqual([row["startMs"] for row in report["sceneEvidence"]["cameraMotion"]],
                         [100, SPEECH_WINDOW_MS + 100])
        motion = report["videoEvidence"]["editorTimeline"]["personMotion"]
        self.assertEqual([(row["startMs"], row["faceId"], row["track"], row["basis"]) for row in motion],
                         [(100, 3, 1, "camera-compensated"),
                          (SPEECH_WINDOW_MS + 100, ID_BLOCK + 3, ID_BLOCK + 1, "camera-compensated")])
        self.assertEqual([row["startMs"] for row in report["coverage"]["face"]], [0, SPEECH_WINDOW_MS])
        self.assertEqual([(row["startMs"], row["endMs"], row["status"]) for row in report["coverage"]["scene"]],
                         [(0, SPEECH_WINDOW_MS, "sampled"), (SPEECH_WINDOW_MS, end, "sampled")])
        self.assertEqual(report["videoEvidence"]["editorTimeline"]["shots"], [{"startMs": 0, "basis": "ffmpeg-5fps"}])

    def test_import_recovers_requested_directions_on_existing_face_tracks(self):
        timeline = {"stepMs": 40, "frames": [[[1, .1, .2, .3, .4, 1, 3]]]}
        old_box = [.05, .1, .5, .8]
        plan = {"body": True, "gaze": True}
        report = {"videoEvidence": {"timeline": timeline, "personLayer": {"plan": plan, "frames": [
            {"timeMs": 0, "people": [{"track": 1, "faceId": 3, "box": old_box}]}
        ]}}}

        def recover(source, work, evidence, *, body, gaze):
            self.assertTrue(body and gaze)
            self.assertIs(evidence["timeline"], timeline)
            evidence["personLayer"] = {"status": "ok", "plan": plan, "frames": [
                {"timeMs": 0, "people": [{"track": 1, "faceId": 3, "box": [.2, .2, .2, .2],
                    "headPose": {"horizontal": "right"}, "gaze": {"horizontal": "left"}}]}
            ]}

        with TemporaryDirectory() as folder, patch('voice_studio.graph.media_graph_ingest.attach_person_evidence', side_effect=recover):
            root = Path(folder)
            self.assertTrue(restore_imported_person_layer(root, root / "source.mp4", report))
            self.assertFalse(restore_imported_person_layer(root, root / "source.mp4", report))
        person = report["videoEvidence"]["personLayer"]["frames"][0]["people"][0]
        self.assertEqual(person["box"], old_box)
        self.assertEqual(person["headPose"]["horizontal"], "right")
        self.assertEqual(person["gaze"]["horizontal"], "left")

    def test_coarse_cuts_read_the_original_offsets_and_reject_boundary_duplicates(self):
        class Worker:
            stderr = "pts_time:0.2\npts_time:1.2\npts_time:1.4\npts_time:2.0\n"

        with patch('voice_studio.graph.media_graph_ingest.subprocess.run', return_value=Worker()) as runner:
            self.assertEqual(scan_shots(Path("/example.mp4"), 1000, 2700), [1600, 2400])
        options = runner.call_args.args[0]
        self.assertLess(options.index("-t"), options.index("-i"))
        self.assertEqual(options[options.index("-protocol_whitelist") + 1], "file,pipe")
