"""Verify guided recording and safe deletion using synthetic PCM."""

from test_studio import sample
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.synthesis.corpus import HELD_OUT
from voice_studio.app.studio import Studio


class RecorderTests(TestCase):
    def setUp(self):
        """Open an isolated workspace for each test."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.studio = Studio(Path(directory.name), demo=True)
        self.addCleanup(self.studio.worker.shutdown)

    def test_corrected_transcript_advances_prompt(self):
        """Use the prompt identity even when the transcript differs from the script."""
        first = self.studio.state("ru")
        self.assertEqual(first["promptCount"], 24)
        self.assertEqual(first["promptNumber"], 1)
        self.studio.record("ru", "Другими словами.", sample(), prompt_number=1)
        second = self.studio.state("ru")
        self.assertEqual(second["promptNumber"], 2)
        self.assertEqual(second["promptDone"], 1)
        self.assertEqual(second["heldOut"], HELD_OUT["ru"])
        self.assertEqual(self.studio.state("en")["promptNumber"], 1)

    def test_deleting_saved_reference_requires_removing_profile(self):
        """Keep stored profile references valid and remove the WAV after approval."""
        key = self.studio.record("ru", "Проверка.", sample(), prompt_number=1)
        profile = self.studio.save_profile("ru", [key])
        with self.assertRaisesRegex(ValueError, "профиль"):
            self.studio.delete("ru", "recording", key)
        self.studio.delete("ru", "profile", profile)
        self.studio.delete("ru", "recording", key)
        self.assertFalse((self.studio.media / f"{key}.wav").exists())
        self.assertEqual(self.studio.state("ru")["promptNumber"], 1)

    def test_failed_unlink_preserves_recording(self):
        """Leave the recording selectable when the WAV cannot be removed."""
        key = self.studio.record("ru", "Проверка.", sample())
        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.studio.delete("ru", "recording", key)
        self.assertEqual(self.studio.state("ru")["recordings"][0]["id"], key)
        self.assertTrue((self.studio.media / f"{key}.wav").is_file())

    def test_evaluation_recording_is_only_a_comparison_target(self):
        """Keep held-out audio out of progress, profiles, exports and conditioning."""
        state = self.studio.state("ru")
        text = state["prompt"]
        target = self.studio.record("ru", text, sample(), prompt_number=1, evaluation_only=True)
        self.assertEqual(self.studio.state("ru")["promptNumber"], 1)
        with self.assertRaisesRegex(ValueError, "Только для проверки"):
            self.studio.save_profile("ru", [target])
        with self.assertRaisesRegex(ValueError, "Только для проверки"):
            self.studio.export("ru", [target], self.studio.root / "corpus.zip")
        with self.assertRaisesRegex(ValueError, "Только для проверки"):
            self.studio.synthesize("ru", {"text": text, "models": ["f5"], "refs": [target]})
        source = self.studio.record("ru", "Образец голоса.", sample())
        ids = self.studio.synthesize("ru", {"text": text, "models": ["f5"],
                                           "refs": [source], "comparison": target})
        self.studio.worker.shutdown()
        result = next(row for row in self.studio.state("ru")["jobs"] if row["id"] == ids[0])
        self.assertEqual((result["status"], result["comparison_id"]), ("done", target))
