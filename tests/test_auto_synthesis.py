"""Exercise automatic stopping, promotion and rollback with controlled scores."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.synthesis.auto_synthesis import assess
from voice_studio.app.studio import Studio
from test_studio import sample


def scores(cosine, wer):
    """Return a fixed pair of independent local metrics."""
    return {"xtts_encoder_cosine": cosine, "whisper_word_error_rate": wer}


class AutoSynthesisTests(TestCase):
    def test_one_regressed_metric_blocks_promotion_even_if_score_increases(self):
        best = {"cosine": .75, "wer": .2, "quality": 77.0}
        self.assertEqual(assess({"cosine": .9, "wer": .21}, best)[1], "regression")
        self.assertEqual(assess({"cosine": None, "wer": 0}, best), (None, "unscored"))

    def test_live_queue_reaches_target_and_preserves_edges(self):
        with TemporaryDirectory() as folder:
            studio = Studio(Path(folder))
            try:
                source = studio.record("ru", "Пять шариков лежат на столе.", sample())
                refs = [studio.record("ru", f"Образец номер {number}.", sample()) for number in range(3)]

                def generate(model, job, demo):
                    Path(job["output"]).write_bytes(sample())

                with patch.object(studio.automatic, "enabled", return_value=True), \
                     patch('voice_studio.app.studio.availability', return_value={"coqui": {"ready": True}}), \
                     patch('voice_studio.app.studio.generate', side_effect=generate), \
                     patch('voice_studio.app.studio.measure', side_effect=[scores(.72, .2), scores(.68, .1), scores(.9, .05)]) as metric:
                    key = studio.automatic.start("ru", {
                        "comparison": source, "refs": [source, *refs], "models": ["coqui"],
                        "target": 90, "max_attempts": 6, "minutes": 2,
                    })
                    studio.worker.submit(lambda: None).result(timeout=20)
                run, = studio.automatic.state("ru")
                first, second, third = run["attempts"]
                self.assertEqual(key, run["id"])
                self.assertEqual((run["status"], run["reason"], run["best_id"]),
                                 ("stopped", "target", third["job_id"]))
                self.assertEqual([a["outcome"] for a in run["attempts"]], ["promoted", "regression", "promoted"])
                self.assertEqual(second["incumbent_id"], first["job_id"])
                self.assertEqual(third["incumbent_id"], first["job_id"])
                self.assertEqual(third["previous_job"], second["job_id"])
                self.assertEqual(metric.call_count, 3)
                self.assertEqual([a["references"] for a in run["attempts"]], [3, 2, 3])
                for node in run["attempts"]:
                    self.assertTrue((studio.media / f"{node['job_id']}.wav").is_file())
                studio.delete("ru", "result", third["job_id"])
                self.assertEqual(studio.automatic.state("ru")[0]["best_id"], first["job_id"])
            finally:
                studio.worker.shutdown()

    def test_auto_run_expires_in_queue_without_starting_inference(self):
        with TemporaryDirectory() as folder:
            studio = Studio(Path(folder), demo=True)
            try:
                with patch('voice_studio.synthesis.auto_synthesis.time', return_value=100), patch.object(studio, "run_synthesis") as run:
                    studio.write("""INSERT INTO auto_runs(id,language,demo,text,refs,original_id,target,max_attempts,
                                 created,deadline,status) VALUES ('test','ru',1,'Фраза','[]','original',85,5,0,60,'queued')""")
                    studio.automatic.work("test", "ru", {}, [], [], {}, 85, 5, 60)
                result, = studio.automatic.state("ru")
                self.assertEqual((result["status"], result["reason"], result["attempts"]), ("stopped", "time_limit", []))
                run.assert_not_called()
            finally:
                studio.worker.shutdown()
