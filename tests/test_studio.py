"""Exercise persistence, model routing, MLflow, and local audio handling."""

from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from time import monotonic, sleep
import unittest
from unittest.mock import patch
import wave
from zipfile import ZipFile

import msgpack
from voice_studio.synthesis.engines import generate
from voice_studio.app.studio import PROMPTS, Studio


def sample():
    """Build a synthetic recording without using a real voice."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x10\x00" * 24000)
    return output.getvalue()


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.studio = Studio(self.root, demo=True)
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(self.studio.worker.shutdown)

    def wait_for(self, ids):
        """Wait for persisted completion, reporting errors from any failed jobs."""
        deadline = monotonic() + 20
        while monotonic() < deadline:
            jobs = {row["id"]: row for row in self.studio.state("ru")["jobs"]}
            if all(jobs.get(key, {}).get("status") in {"done", "failed"} for key in ids):
                self.assertEqual([jobs[key]["error"] for key in ids], [None] * len(ids))
                return
            sleep(.05)
        self.fail(f"Jobs did not finish: {ids}")

    def test_record_iterate_compare_assemble_and_track(self):
        first = self.studio.record("ru", PROMPTS["ru"][0], sample())
        self.assertEqual(self.studio.state("ru")["prompt"], PROMPTS["ru"][1])
        second = self.studio.record("ru", PROMPTS["ru"][1], sample())
        refs = [first, second]
        version = self.studio.save_profile("ru", refs)
        data = {"text": "Фраза для проверки.", "models": ["f5", "fish", "coqui"],
                "refs": refs, "profile": version}
        ids = self.studio.synthesize("ru", data)
        self.wait_for(ids)
        self.studio.rate("ru", ids[0], 4)
        stitched = self.studio.assemble("ru", [ids[2], ids[0]])
        with wave.open(str(self.studio.media / f"{stitched}.wav")) as audio:
            self.assertAlmostEqual(audio.getnframes() / audio.getframerate(), 3.18, places=2)
        archive = self.root / "corpus.zip"
        self.studio.export("ru", refs, archive)
        with ZipFile(archive) as exported:
            rows = [json.loads(line) for line in exported.read("manifest.jsonl").splitlines()]
            self.assertEqual([row["text"] for row in rows], PROMPTS["ru"][:2])
            self.assertEqual(len(exported.namelist()), 3)
        with closing(sqlite3.connect(self.root / "mlflow.db")) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM runs WHERE status='FINISHED'").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT value FROM metrics WHERE key='listener_similarity'").fetchone()[0], 4)
            self.assertIsNone(db.execute("SELECT 1 FROM params WHERE value=?", (data["text"],)).fetchone())
        restarted = Studio(self.root, demo=True)
        self.addCleanup(restarted.worker.shutdown)
        self.assertEqual(len(restarted.state("ru")["recordings"]), 2)
        self.assertEqual(len(restarted.state("ru")["jobs"]), 4)

    def test_russian_story_uses_existing_recording_and_profile_flow(self):
        """The full script and its parts must fit the transcript limit and preserve corpus prompts."""
        before = self.studio.state("ru")
        story = before["readingStory"]
        self.assertEqual(story["text"], " ".join(part["text"] for part in story["parts"]))
        self.assertEqual(len(story["parts"]), 5)
        ids = [self.studio.record("ru", text, sample())
               for text in (story["text"], story["parts"][0]["text"])]
        self.studio.save_profile("ru", ids)
        self.assertEqual(self.studio.state("ru")["promptNumber"], before["promptNumber"])
        self.assertIsNone(self.studio.state("en")["readingStory"])

    def test_reject_wrong_language_and_invalid_audio(self):
        key = self.studio.record("ru", "Это запись.", sample())
        with self.assertRaises(ValueError):
            self.studio.save_profile("en", [key])
        with self.assertRaises(ValueError):
            self.studio.record("ru", "Не аудио.", b"not audio")
        with self.assertRaises(ValueError):
            self.studio.synthesize("ru", {"models": [["f5"]], "text": "Тест", "refs": [key]})

    def test_synthesis_preserves_exact_checkpoint(self):
        reference = self.studio.record("ru", "Образец.", sample())
        self.studio.write("INSERT INTO checkpoints VALUES(?,?,?,?,?,?)",
                          ("checkpoint-one", "f5", "ru", 2, "/test/weights", ""))
        ids = self.studio.synthesize("ru", {"models": ["f5"], "text": "Тест.",
                                            "refs": [reference], "checkpoints": {"f5": "checkpoint-one"}})
        self.wait_for(ids)
        job = self.studio.state("ru")["jobs"][0]
        self.assertEqual((job["checkpoint_id"], job["checkpoint_profile"]), ("checkpoint-one", 2))
        run = self.studio.tracking.client.get_run(job["run_id"])
        self.assertEqual(run.data.params["checkpoint_id"], "checkpoint-one")

    def test_lapshi_uses_paired_seeds_and_held_out_original(self):
        """Changing the TTS must preserve the seed, conditioning take and evaluation text."""
        original = self.studio.record("ru", "Мы стоим у лапши.", sample(), evaluation_only=True)
        reference = self.studio.record("ru", "Образец голоса.", sample())
        with (patch.object(self.studio, "demo", False),
              patch('voice_studio.app.studio.F5_ACCENT_WEIGHTS', self.root / "studio.db"),
              patch.object(self.studio, "synthesis_plan"),
              patch.object(self.studio, "enqueue_synthesis") as queue):
            self.studio.compare_lapshi("ru", {"comparison": original, "refs": [reference]})
        language, script, rows, _, candidates, comparison = queue.call_args.args
        self.assertEqual((language, script, rows[0]["id"], comparison),
                         ("ru", "Мы стоим у лапши.", reference, original))
        self.assertEqual(len(candidates), 9)
        for offset, seed in enumerate((17027, 88219, 451007)):
            stock, accent, coqui = candidates[offset * 3:offset * 3 + 3]
            self.assertEqual([take[:1] for take in (stock, accent, coqui)], [("f5",), ("f5",), ("coqui",)])
            self.assertEqual([take[2] for take in (stock, accent, coqui)], [seed] * 3)
            self.assertEqual(accent[1]["architecture"], "F5TTS_v1_Base")
            self.assertEqual(accent[4], [("Мы стоим у лапш+и.", 0)])
            self.assertEqual(stock[4], coqui[4])

    def test_accent_retry_persists_checkpoint_policy_and_legacy_fallback(self):
        """A repeat of a legacy accent take and its child both retain the plus policy."""
        reference = self.studio.record("ru", "Образец голоса.", sample())
        text = "Мы стоим у лапши."
        with patch('voice_studio.app.studio.F5_ACCENT_WEIGHTS', self.root / "studio.db"):
            parent, = self.studio.enqueue_synthesis("ru", text, self.studio.references("ru", [reference]), 0,
                [("f5", None, 10, "lapshi-accent", [(text, 0)])], None)
            self.wait_for([parent])
            first, = self.studio.retry_words("ru", parent, 3, 1, "wide")
            self.wait_for([first])
            first_job = self.studio.synthesis_result("ru", first)
            self.assertEqual(first_job["accent_mode"], "lapshi-plus")
            # Emulate a fragment saved by the older implementation before the new column existed.
            self.studio.write("UPDATE jobs SET accent_mode=NULL WHERE id=?", (first,))
            with patch('voice_studio.app.studio.generate', wraps=generate) as observed:
                second, = self.studio.retry_words("ru", first, 3, 1, "wide")
                self.wait_for([second])
            second_job = self.studio.synthesis_result("ru", second)
            self.assertEqual(second_job["accent_mode"], "lapshi-plus")
            for job in (first_job, second_job):
                self.assertEqual(self.studio.tracking.client.get_run(job["run_id"]).data.params["builtin_checkpoint"],
                                 "f5-accent-tune")
            self.assertEqual(observed.call_args.args[1]["architecture"], "F5TTS_v1_Base")
            self.assertIn("лапш+и", observed.call_args.args[1]["text"])

    def test_prosody_pair_rating_is_persisted_in_mlflow(self):
        """The plain run stores the pair verdict returned through studio state."""
        reference = self.studio.record("ru", "Образец голоса.", sample())
        text = "У лапши."
        plain, plus = self.studio.enqueue_synthesis("ru", text, self.studio.references("ru", [reference]), 0, [
            ("f5", None, 17027, "pilot-plain", [(text, 0)]),
            ("f5", None, 17027, "pilot-plus", [("У лапш+и.", 0)]),
        ], None, batch_id="test-pair")
        self.wait_for([plain, plus])
        with self.assertRaises(ValueError):
            self.studio.rate_prosody_pair("ru", plus, "both")
        self.studio.rate_prosody_pair("ru", plain, "unclear")
        pair = next(row for row in self.studio.state("ru")["jobs"] if row["id"] == plain)
        self.assertEqual(pair["listener_choice"], "unclear")
        self.assertEqual(self.studio.tracking.client.get_run(pair["run_id"]).data.tags["listener_stress_choice"],
                         "unclear")

    def test_fish_request_uses_binary_references_and_transcripts(self):
        key = self.studio.record("ru", "Пример.", sample())
        reference = self.studio.media / f"{key}.wav"
        output = self.root / "fish.wav"
        with patch('voice_studio.synthesis.engines.fish_request', return_value=sample()) as request:
            generate("fish", {"output": str(output), "text": "Новая фраза.", "seed": 42,
                              "references": [{"path": str(reference), "text": "Пример."}]}, False)
        route, body, timeout = request.call_args.args
        self.assertEqual((route, timeout), ("/v1/tts", 600))
        payload = msgpack.unpackb(body)
        self.assertEqual(payload["references"], [{"audio": reference.read_bytes(), "text": "Пример."}])
        self.assertEqual(output.read_bytes(), sample())


if __name__ == "__main__":
    unittest.main()
