"""Exercise shared status tracking without running local models."""

from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from voice_studio.app.server import Handler
from voice_studio.core.analysis_runs import AnalysisRun, AnalysisRuns
from voice_studio.core.analysis_stages import stage
from voice_studio.mcp.media_graph_mcp import GraphBuildQueue


class AnalysisRunTests(TestCase):
    def test_wall_clock_is_not_the_sum_of_parallel_work(self):
        """Keep actual elapsed time, cache hits, and recoverable errors as separate evidence."""
        with patch("voice_studio.core.analysis_runs.monotonic") as time:
            time.return_value = 10
            run = AnalysisRun("studio")
            run.start()
            run.observe({"stage": "scene", "state": "running"})
            run.observe({"stage": "person-layer", "state": "running"})
            time.return_value = 12
            run.observe({"stage": "scene", "state": "failed", "reason": "OCR failed"})
            run.observe({"stage": "scene", "state": "done", "elapsedMs": 80, "cacheHit": True})
            run.observe({"stage": "person-layer", "state": "done"})
            run.finish()
        result = run.snapshot()
        scene, people = result["stages"]
        self.assertEqual((result["elapsedMs"], people["elapsedMs"]), (2000, 2000))
        self.assertEqual((scene["state"], scene["elapsedMs"], scene["cacheHit"]), ("done", 80, True))
        self.assertEqual(scene["lastFailure"]["message"], "OCR failed")
        self.assertEqual(scene["retryGroup"], "scene")
        self.assertEqual(result["failedStages"], [])

    def test_studio_stream_and_status_share_the_same_terminal_failure(self):
        """Correlate the browser upload record with the agent-facing run and exact stage error."""
        registry = AnalysisRuns()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "media"
            handler = Mock()
            handler.server = SimpleNamespace(studio=SimpleNamespace(root=path.parent), analysis_runs=registry)
            handler.uploaded_media.return_value = nullcontext(path)
            handler.wfile = BytesIO()
            handler.stream_analysis.side_effect = lambda analyze, run: Handler.stream_analysis(handler, analyze, run)

            def failure(*args, emit, **kwargs):
                """Cancel the run with a diagnostic from the actual failing step."""
                with stage(emit, "whisper"):
                    raise ValueError("Whisper fixture failed")

            with patch("voice_studio.app.server.transcribe_uploaded_media", side_effect=failure):
                Handler.receive_media(handler, "en", progress=True)
        events = [json.loads(row) for row in handler.wfile.getvalue().splitlines()]
        run = registry.status(events[0]["runId"])
        self.assertEqual(run, events[-1]["pipeline"])
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["failedStages"], ["whisper"])
        self.assertEqual(run["stages"][0]["state"], "done")
        self.assertEqual(run["stages"][1]["error"], {"type": "ValueError", "message": "Whisper fixture failed"})

    def test_graph_job_reports_running_and_failed_stages_without_partial_graphs(self):
        """A regular agent job has live stages even with progressive snapshots disabled."""
        queue = GraphBuildQueue(track_stages=True)
        entered = Event()
        release = Event()

        def operation(*, publication):
            """Expose one active stage while blocked on an independent fixture event."""
            with stage(publication.activity, "scene"):
                entered.set()
                release.wait(3)
                raise ValueError("Scene fixture failed")

        job = queue.submit(operation)["jobId"]
        try:
            self.assertTrue(entered.wait(3))
            current = queue.status(job)
            self.assertEqual(current["pipeline"]["stages"][0]["state"], "running")
            self.assertNotIn("snapshot", current)
        finally:
            release.set()
            queue.worker.shutdown(wait=True)
        final = queue.status(job)
        self.assertEqual(final["pipeline"]["failedStages"], ["scene"])
        self.assertEqual(final["pipeline"]["stages"][0]["error"]["message"], "Scene fixture failed")
