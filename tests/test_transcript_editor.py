"""Verify per-cue names, source evidence and concurrent edit protection."""

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.app.transcript_editor import RevisionConflict, TranscriptEditor


class TranscriptEditorTests(TestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory())

        def connect():
            db = sqlite3.connect(Path(folder) / "test.db")
            db.row_factory = sqlite3.Row
            return db

        self.editor = TranscriptEditor(connect)
        self.initial = self.editor.create("ru", "Диалог", [
            {"startMs": 0, "endMs": 2000, "text": "Максим, это так?", "speaker": 1},
            {"startMs": 2000, "endMs": 3000, "text": "Ну это правда.", "speaker": 1, "visualSpeaker": 10},
            {"startMs": 3000, "endMs": 4000, "text": "Продолжим.", "speaker": 1},
        ])

    def test_voice_abstention_and_source_profile_survive_save_without_raw_id_collision(self):
        """Persist the source namespace guard and discard private nested evidence."""
        from voice_studio.speech.speaker_profiles import cue_number
        voice = 'a' * 32
        saved = self.editor.create('ru', 'Long voice material', [
            {'startMs': 0, 'endMs': 2000, 'text': 'Assigned.', 'speaker': 9000001,
             'profileSpeakerNumber': 1, 'voiceProfileId': voice,
             'voiceProfileEvidence': {'state': 'matched', 'basis': 'audio', 'secret': {'vector': [1, 2]}}},
            {'startMs': 2000, 'endMs': 1800000, 'text': 'Unknown.', 'speaker': 1,
             'voiceProfileEvidence': {'state': 'abstained'}}])
        cues = self.editor.get('ru', saved['id'])['segments']
        self.assertEqual([cue_number(cue) for cue in cues], [1, None])
        self.assertEqual(cues[0]['speaker'], 9000001)
        self.assertNotIn('secret', cues[0]['voiceProfileEvidence'])
        self.assertEqual(cues[1]['endMs'], 1800000)

    def test_named_cue_takes_priority_without_relabeling_other_cues(self):
        self.assertIn("Спикер 1: Ну это правда.", self.initial["srt"])
        updated = self.editor.patch("ru", {
            "id": self.initial["id"], "revision": 1, "startMs": 2000, "speakerNumber": 2,
            "contextStartMs": 0, "source": "agent", "text": "Ну, это правда.",
        })
        before, edited, after = updated["segments"]
        self.assertNotIn("speakerNumber", before)
        self.assertNotIn("speakerNumber", after)
        self.assertEqual((edited["speakerNumber"], edited["speaker"], edited["visualSpeaker"]), (2, 1, 10))
        self.assertEqual(edited["asrText"], "Ну это правда.")
        self.assertEqual(edited["speakerNumberEvidence"]["contextText"], "Максим, это так?")
        self.assertEqual(updated["revision"], 2)
        self.assertIn("Спикер 2: Ну, это правда.", updated["srt"])
        self.assertIn("Спикер 1: Продолжим.", updated["srt"])

    def test_stale_edit_and_forward_evidence_are_rejected(self):
        data = {"id": self.initial["id"], "revision": 1, "startMs": 2000, "speakerName": "Максим"}
        with self.assertRaisesRegex(ValueError, "раннюю"):
            self.editor.patch("ru", {**data, "contextStartMs": 3000})
        self.editor.patch("ru", {**data, "contextStartMs": 0})
        with self.assertRaises(RevisionConflict):
            self.editor.patch("ru", {**data, "speakerName": "Виктор"})
        cleared = self.editor.patch("ru", {**data, "revision": 2, "speakerName": ""})
        self.assertNotIn("speakerName", cleared["segments"][1])
        self.assertIn("Спикер 1: Ну это правда.", cleared["srt"])

    def test_recognition_evidence_round_trips_until_manual_text_edit(self):
        source = "Максим Лучак."
        fix = {"word": "Лучак", "replacement": "Лутчак", "start": 7, "end": 12,
               "timestampsMs": [250, 650], "basis": "repeated-title-card-ocr"}
        initial = self.editor.create("ru", "Титр", [
            {"startMs": 0, "endMs": 1500, "text": "Максим Лутчак.", "asrText": source, "asrCorrection": fix,
             "visualSpeaker": 10, "visualSpeakerBasis": "short-reply-face-and-voice"},
        ])
        saved = initial["segments"][0]
        self.assertEqual(saved["asrCorrection"], fix)
        self.assertEqual(saved["visualSpeakerBasis"], "short-reply-face-and-voice")
        manual = self.editor.patch("ru", {"id": initial["id"], "startMs": 0, "revision": 1,
                                          "text": "Проверено редактором."})
        cue = manual["segments"][0]
        self.assertEqual((cue["text"], cue["asrText"]), ("Проверено редактором.", source))
        self.assertNotIn("asrCorrection", cue)

    def test_delete_respects_language_and_revision_and_preserves_other_materials(self):
        key = self.initial["id"]
        other = self.editor.create("en", "A transcript", self.initial["segments"])
        changed = self.editor.patch("ru", {
            "id": key, "revision": 1, "startMs": 0, "text": "Правка.",
        })
        with self.assertRaisesRegex(ValueError, "не найден"):
            self.editor.delete("en", key, changed["revision"])
        with self.assertRaises(RevisionConflict):
            self.editor.delete("ru", key, 1)
        self.assertEqual(self.editor.get("ru", key)["segments"][0]["text"], "Правка.")
        self.editor.delete("ru", key, changed["revision"])
        self.assertEqual(self.editor.list("ru"), [])
        self.assertEqual(self.editor.get("en", other["id"])["title"], "A transcript")

    def test_recovers_only_exact_cue_of_same_media_and_keeps_model_labels(self):
        digest = "a" * 64
        cues = self.initial["segments"]
        material = self.editor.create("ru", "Исходник", cues, digest)
        self.editor.patch("ru", {
            "id": material["id"], "revision": 1, "startMs": 2000, "speakerNumber": 2,
            "contextStartMs": 0, "source": "agent",
        })
        incoming = {"segments": [dict(cue) for cue in cues]}
        graph = self.editor.recover("ru", digest, incoming)
        self.assertEqual([row["state"] for row in graph["decisions"]], ["applied"])
        self.assertEqual(incoming["segments"][1]["speaker"], 1)
        self.assertEqual(incoming["segments"][1]["speakerNumber"], 2)
        self.assertNotIn("speakerNumber", incoming["segments"][2])
        self.assertIn("Спикер 2: Ну это правда.", incoming["srt"])
        for other, text in [(digest, "Совсем другой текст."), ("b" * 64, cues[1]["text"])]:
            incoming = {"segments": [dict(cue) for cue in cues]}
            incoming["segments"][1]["text"] = text
            self.assertEqual(self.editor.recover("ru", other, incoming)["decisions"], [])

    def test_matching_profile_keeps_previously_measured_appearance(self):
        digest = "a" * 64

        def analysis(face, traits):
            return {"speakerCount": 1, "segments": [
                {"startMs": 100, "endMs": 1100, "text": "Голос", "speaker": 1, "visualFace": face}],
                "videoEvidence": {"timeline": {"frames": [[]]},
                                  "faceSpeakers": {str(face): {"speaker": 1, "basis": "audio-visual"}},
                                  "editorTimeline": {"faceAttributes": {str(face): traits}}}}

        initial = analysis(1, {"age": 25, "ageSamples": 2, "appearanceClass": "male",
                               "hairSwatch": "#2d221d", "hairSource": "mediapipe-hair-mask"})
        self.editor.remember_profiles("ru", digest, initial)
        newer = analysis(1, {"age": 26, "ageSamples": 3})
        self.editor.remember_profiles("ru", digest, newer)
        first, = initial["speakerProfileIndex"]
        second, = newer["speakerProfileIndex"]
        self.assertEqual(second["id"], first["id"])
        material = self.editor.create("ru", "Исходник", newer["segments"], digest)
        saved, = self.editor.get("ru", material["id"])["speakerProfileIndex"]
        self.assertEqual(saved["appearance"], {"age": 26, "ageSamples": 3, "appearanceClass": "male",
                                               "hairSwatch": "#2d221d", "hairSource": "mediapipe-hair-mask"})
        different = analysis(2, {})
        self.editor.remember_profiles("ru", digest, different)
        row, = different["speakerProfileIndex"]
        self.assertNotIn("appearance", row)

    def test_conflicting_anchors_are_preserved_without_overriding_the_model(self):
        digest = "a" * 64
        cues = self.initial["segments"]
        for number in (2, 3):
            draft = [dict(cue) for cue in cues]
            draft[1]["speakerNumber"] = number
            saved = self.editor.create("ru", "Материал", draft, digest)
            self.assertEqual(saved["segments"][1]["speakerNumber"], number)
        incoming = {"segments": [dict(cue) for cue in cues]}
        decision, = self.editor.recover("ru", digest, incoming)["decisions"]
        self.assertEqual(decision["state"], "conflicting")
        self.assertNotIn("speakerNumber", incoming["segments"][1])
        self.assertIn("Спикер 1: Ну это правда.", incoming["srt"])
        self.assertEqual({edge["to"] for edge in decision["edges"] if edge["relation"] == "assigns"},
                         {"speaker:2", "speaker:3"})
