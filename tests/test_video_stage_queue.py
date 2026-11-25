"""Check that video stages wait for shared workers and finish the original request."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Event
from unittest import TestCase
from unittest.mock import patch

from voice_studio.core.analysis_stages import queued_stage_slot, stage
from voice_studio.speech.media_transcript import attach_person_evidence, collect_scene
from voice_studio.vision import person_layers, video_revolver, video_scene


class VideoStageQueueTests(TestCase):
    def run_while_occupied(self, resource, operation, stage_name):
        """Hold the real semaphore until the caller reports waiting."""
        waiting = Event()
        events = []

        def notify(event):
            """Signal when the original request has joined the queue."""
            events.append(event)
            if event["state"] == "queued":
                waiting.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with resource:
                result = pool.submit(operation, notify)
                self.assertTrue(waiting.wait(2), "The stage should wait instead of returning busy")
                self.assertFalse(result.done())
            result = result.result(timeout=3)
        observed = [row for row in events if row["stage"] == stage_name]
        queued = next(index for index, row in enumerate(observed) if row["state"] == "queued")
        self.assertEqual([row["state"] for row in observed[queued:]], ["queued", "running", "done"])
        self.assertTrue(observed[queued]["reason"])
        return result

    def test_face_analysis_resumes_in_the_same_request(self):
        def collect(source, wav, phrases, offset, formats, result, emit, **kwargs):
            """Mimic the worker completion while keeping the public stage events."""
            with stage(emit, "light-asd"):
                result["stages"]["yunet"] = "ok"
            return result

        def review(emit):
            return video_revolver.review_video(Path(__file__), None, [], 0, "mov", emit=emit)

        with (patch.object(video_revolver, "probe_video", return_value=(True, 1000)),
              patch.object(video_revolver, "MODEL", Path(__file__)),
              patch.object(video_revolver, "collect_evidence", side_effect=collect)):
            result = self.run_while_occupied(video_revolver.VISUAL_SLOT, review, "light-asd")
        self.assertEqual(result["stages"]["yunet"], "ok")

    def test_scene_analysis_waits_for_the_same_visual_slot(self):
        def review(emit):
            return collect_scene(Path(__file__), Path(), {}, "baseline", emit, None)

        with (patch.object(video_scene, "probe_video", return_value=True),
              patch.object(video_scene, "scan_scene", return_value=([(0, 0)], [], None)),
              patch.object(video_scene, "sample_plan", return_value=[{"timeMs": 0}]),
              patch.object(video_scene, "label_visible_faces"),
              patch.object(video_scene, "decode_samples"),
              patch.object(video_scene, "run_worker", return_value={"status": "ok"})):
            result = self.run_while_occupied(video_revolver.VISUAL_SLOT, review, "scene")
        self.assertEqual(result["status"], "ok")

    def test_person_preview_queues_behind_other_person_work(self):
        def preview(emit):
            evidence = {"timeline": {"frames": [[]], "stepMs": 40}}
            attach_person_evidence(Path(__file__), Path(), evidence, body=True, gaze=False, emit=emit)
            return evidence["personLayer"]

        with (patch.object(person_layers, "select_frames", return_value=[]),
              patch.object(person_layers, "dispatch_layers", return_value={"status": "ok"})):
            result = self.run_while_occupied(person_layers.SLOT, preview, "person-layer")
        self.assertEqual(result["status"], "ok")

    def test_disconnected_waiter_does_not_consume_a_slot(self):
        slot = BoundedSemaphore(1)

        def disconnect(event):
            raise BrokenPipeError("disconnected")

        with slot:
            with self.assertRaises(BrokenPipeError):
                with queued_stage_slot(slot, disconnect, "scene"):
                    self.fail("Disconnected request entered a video worker")
        self.assertTrue(slot.acquire(blocking=False))
        slot.release()
