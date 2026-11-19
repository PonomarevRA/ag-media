"""Verify the offline MP4 renderer's timed overlay contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.vision.video_export import write_overlay, write_preview_overlay


class VideoExportTests(TestCase):
    def test_export_uses_visible_label_and_escapes_caption_markup(self):
        data = {
            "stepMs": 40,
            "frames": [[[1, .1, .2, .3, .4, 1, 3]]],
            "activeTracks": [[1]],
            "faceSpeakers": {"3": {"speaker": 9, "basis": "visible-only"}},
            "segments": [{"startMs": 320, "endMs": 890, "text": r"Тест{\pos(1,2)}", "number": 2}],
            "shots": [0],
        }
        with TemporaryDirectory() as folder:
            overlay = Path(folder) / "layer.ass"
            write_overlay(data, overlay, 1280, 720)
            ass = overlay.read_text()
        self.assertIn("0:00:00.00,0:00:00.04,Box", ass)
        self.assertIn(r"\1c&H4EB3FF&", ass)
        self.assertIn("Лицо 3", ass)
        self.assertIn("0:00:00.32,0:00:00.89,Caption", ass)
        self.assertIn("Спикер 2 · Тест｛＼pos(1,2)｝", ass)
        self.assertIn("План 1", ass)

    def test_export_labels_overlap_only_within_its_timed_window(self):
        data = {
            "segments": [{
                "startMs": 100,
                "endMs": 900,
                "text": "Reply",
                "number": 1,
                "overlap": True,
                "overlaps": [{"startMs": 300, "endMs": 500, "speakers": [1, 2]}],
            }],
        }
        with TemporaryDirectory() as folder:
            overlay = Path(folder) / "layer.ass"
            write_overlay(data, overlay, 1280, 720, "en")
            captions = [line for line in overlay.read_text().splitlines() if ",Caption,," in line]
        self.assertEqual(captions, [
            "Dialogue: 2,0:00:00.10,0:00:00.30,Caption,,0,0,0,,Speaker 1 · Reply",
            "Dialogue: 2,0:00:00.30,0:00:00.50,Caption,,0,0,0,,Overlapping speech · speakers 1, 2 · Reply",
            "Dialogue: 2,0:00:00.50,0:00:00.90,Caption,,0,0,0,,Speaker 1 · Reply",
        ])


    def test_preview_commands_preserve_geometry_color_and_direction_labels(self):
        plan = {"width": 480, "height": 270, "samples": [[500, 900, [
            ["stroke", "rgba(153, 225, 152, .50)", 300, 150, 48, 55, 1.25, []],
            ["fill", "rgba(16, 13, 10, .94)", 48, 50, 170, 20],
            ["text", "#f5f1e8", 54, 64, 12, r"Взгляд ← · Голова →↑ {unsafe}"],
        ]]]}
        with TemporaryDirectory() as folder:
            overlay = Path(folder) / "layer.ass"
            write_preview_overlay(plan, overlay)
            ass = overlay.read_text()
        self.assertIn("PlayResX: 480", ass)
        self.assertIn("0:00:00.50,0:00:00.90", ass)
        self.assertIn(r"\1c&H98E199&\1a&H7F&", ass)
        self.assertIn(r"\pos(54,52)\fs12", ass)
        self.assertIn("Взгляд ← · Голова →↑ ｛unsafe｝", ass)
