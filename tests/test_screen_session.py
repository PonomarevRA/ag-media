"""Verify session reuse, deletion limits and caller-visible screenshot metadata."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from voice_studio.mcp import screen_mcp
from voice_studio.vision.screen_session import MAX_GENERATIONS, SESSION_TTL_SECONDS, ScreenSessions, save_layout


class ScreenSessionTests(TestCase):
    def test_disposes_expired_sessions_and_evicts_old_layout_generations(self):
        sessions = ScreenSessions()
        folder, _ = sessions.get("edit", now=20)
        try:
            for generation in range(MAX_GENERATIONS + 2):
                save_layout(folder / f"{generation}.json", {"structure": {}, "generation": generation})
            self.assertEqual(len(list(folder.glob("*.json"))), MAX_GENERATIONS)
            self.assertFalse((folder / "0.json").exists())
        finally:
            sessions.get(None, now=20 + SESSION_TTL_SECONDS)
        self.assertFalse(folder.exists())

    def test_idle_session_expires_without_another_request(self):
        sessions = ScreenSessions()
        callback = sessions.expire
        done = Event()

        def expire():
            """Observe real expiry without relying on a fixed sleep."""
            callback()
            done.set()

        with patch.object(sessions, "expire", expire), patch(
            "voice_studio.vision.screen_session.SESSION_TTL_SECONDS", .05,
        ):
            folder, _ = sessions.get("idle")
            try:
                self.assertTrue(done.wait(timeout=2))
                self.assertFalse(folder.exists())
            finally:
                if sessions.timer:
                    sessions.timer.cancel()

    def test_equal_file_bytes_reuse_analysis_without_reusing_the_original_path(self):
        result = {"ocr": [], "ocrStatuses": {"tesseract": "ok"}, "interface": None,
                  "structure": {"windows": [], "lists": [], "timelineClips": []},
                  "ocrJobs": 1, "mode": "overview"}
        sessions = ScreenSessions()
        try:
            with TemporaryDirectory() as directory, patch.object(screen_mcp, "SESSIONS", sessions), patch(
                "voice_studio.mcp.screen_mcp.analyze_image_bytes", return_value=result,
            ) as recognize:
                first, second = Path(directory) / "first.png", Path(directory) / "second.png"
                first.write_bytes(b"same input")
                second.write_bytes(b"same input")
                initial = screen_mcp.analyze_screenshot(str(first), "ru", mode="overview", session_id="edit")
                repeated = screen_mcp.analyze_screenshot(str(second), "en", mode="overview", session_id="edit")
                recognize.assert_called_once()
                self.assertEqual(initial["source"]["sha256"], repeated["source"]["sha256"])
                self.assertEqual(repeated["source"]["path"], str(second.resolve()))
                self.assertEqual((repeated["reuse"], repeated["ocrJobs"]), ("exact-image", 0))
                self.assertEqual(repeated["draftLanguage"], "en")
        finally:
            for _, folder, _ in sessions.sessions.values():
                folder.cleanup()
