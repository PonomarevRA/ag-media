"""Protect the still-image upload contract without loading production models in unit tests."""

from hashlib import sha256
import http.client
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from voice_studio.app.image_analysis import analyze_image, image_options
from voice_studio.app.server import Handler, StudioServer
from voice_studio.vision.image_input import verify_image
from voice_studio.vision.image_status import ocr_coverage


class ImageAnalysisTests(TestCase):
    """Test validation, resource admission, worker arguments, source identity and explicit partial results."""

    def test_ocr_groups_preserve_partial_coverage_and_recovered_fallback(self):
        """A successful control crop cannot certify unread main text, while recovered fallback is complete."""
        cases = [
            ({"tesseract": "unavailable", "controls-tesseract": "ok"}, "partial"),
            ({"tesseract": "partial"}, "partial"),
            ({"apple-vision": "partial", "tesseract": "ok"}, "ok"),
            ({"apple-vision": "partial", "tesseract": "unavailable"}, "partial"),
            ({"apple-vision": "ok", "controls-tesseract": "unavailable"}, "partial"),
            ({"tesseract": "unavailable"}, "unavailable"),
        ]
        for statuses, expected in cases:
            with self.subTest(statuses=statuses):
                self.assertEqual(ocr_coverage(statuses), expected)

    def test_options_require_known_mode_and_at_least_one_layer(self):
        """Unknown options and an empty layer set cannot start a worker."""
        self.assertEqual(image_options({}), {"mode": "full", "include_ui": True, "include_objects": True})
        for query in ({"mode": ["remote"]}, {"ui": ["yes"]}, {"ui": ["0"], "objects": ["0"]}):
            with self.subTest(query=query), self.assertRaises(ValueError):
                image_options(query)

    def test_worker_reuses_screenshot_path_and_keeps_source_hash_without_temp_path(self):
        """UI success plus unavailable object weights is a partial result with stable source identity."""
        report = {"imageSize": [640, 480], "objectStatus": "unavailable", "uiStatus": "ok", "objects": [], "ocr": []}
        with TemporaryDirectory() as folder:
            source = Path(folder) / "uploaded"
            source.write_bytes(b"fixture-image")
            with patch("voice_studio.app.image_analysis.verify_image"), patch(
                "voice_studio.app.image_analysis.run_worker", return_value=report,
            ) as worker:
                result = analyze_image(source, image_options({}))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["source"], {"sha256": sha256(b"fixture-image").hexdigest(), "bytes": 13})
            self.assertNotIn("path", result["source"])
            args = worker.call_args.args
            self.assertEqual(args[0], "video_scene.py")
            self.assertEqual(args[1][:2], ["--screenshot", source])
            settings = json.loads(args[1][2])
            self.assertTrue(settings["include_objects"])
            self.assertTrue(settings["include_ui"])

    def test_invalid_input_and_worker_failure_release_shared_admission(self):
        """Decoder and worker errors must not permanently block later browser or MCP screenshots."""
        slot = BoundedSemaphore(1)
        with TemporaryDirectory() as folder, patch("voice_studio.app.image_analysis.IMAGE_SLOT", slot):
            path = Path(folder) / "source"
            path.write_bytes(b"image")
            with patch("voice_studio.app.image_analysis.verify_image", side_effect=ValueError("bad image")):
                with self.assertRaisesRegex(ValueError, "bad image"):
                    analyze_image(path, image_options({}))
            self.assertTrue(slot.acquire(blocking=False))
            with self.assertRaisesRegex(ValueError, "другое изображение"):
                analyze_image(path, image_options({}))
            slot.release()
            with patch("voice_studio.app.image_analysis.verify_image"), patch(
                "voice_studio.app.image_analysis.run_worker", side_effect=subprocess.TimeoutExpired("worker", 110),
            ):
                with self.assertRaisesRegex(ValueError, "быстрый режим"):
                    analyze_image(path, image_options({}))
            self.assertTrue(slot.acquire(blocking=False))
            slot.release()

    def test_decoder_rejects_oversized_pixels_before_inference(self):
        """The decoded pixel budget is checked even when the compressed input is tiny."""
        probe = {"streams": [{"codec_name": "png", "width": 8000, "height": 8000}]}
        with TemporaryDirectory() as folder, patch("voice_studio.vision.image_input.subprocess.check_output", return_value=json.dumps(probe).encode()):
            image = Path(folder) / "input.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            with self.assertRaisesRegex(ValueError, "Размеры"):
                verify_image(image)

    def test_decoder_uses_single_image_format_from_bytes_not_filename(self):
        """JPEG bytes select jpeg_pipe even under a PNG name; non-images never reach FFprobe."""
        probe = {"streams": [{"codec_name": "mjpeg", "width": 80, "height": 60}]}
        with TemporaryDirectory() as folder, patch("voice_studio.vision.image_input.subprocess.check_output", return_value=json.dumps(probe).encode()) as decoder:
            image = Path(folder) / "wrong.png"
            image.write_bytes(b"\xff\xd8\xfffixture")
            verify_image(image)
            command = decoder.call_args.args[0]
            self.assertEqual(command[command.index("-f") + 1], "jpeg_pipe")
            image.write_bytes(b"not-an-image")
            with self.assertRaisesRegex(ValueError, "PNG"):
                verify_image(image)
            self.assertEqual(decoder.call_count, 1)


class ImageHTTPTests(TestCase):
    """Exercise the actual same-origin upload and NDJSON transport with an isolated workspace."""

    def test_image_route_streams_result_and_rejects_invalid_options_and_origins(self):
        """The real HTTP route uses a transient upload and admits only same-origin, valid work."""
        with TemporaryDirectory() as folder:
            server = StudioServer(("127.0.0.1", 0), Handler)
            server.studio = SimpleNamespace(root=Path(folder))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            headers = {"Origin": f"http://127.0.0.1:{port}", "X-Voice-Studio": "1"}
            try:
                with patch("voice_studio.app.image_analysis.analyze_image", return_value={"status": "ok", "objects": []}) as model:
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                    connection.request("POST", "/api/analyze-image?progress=1", b"image-bytes", headers)
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertIn("img-src 'self' data: blob:", response.getheader("Content-Security-Policy"))
                    events = [json.loads(line) for line in response.read().splitlines()]
                    self.assertEqual(events[-1]["result"]["status"], "ok")
                    source = model.call_args.args[0]
                    self.assertFalse(source.exists(), "Upload is removed after the response")
                    connection.close()
                    for query, request_headers, expected in [
                        ("?ui=0&objects=0", headers, 400),
                        ("", {**headers, "Origin": "https://external.invalid"}, 403),
                    ]:
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                        connection.request("POST", "/api/analyze-image" + query, b"image", request_headers)
                        response = connection.getresponse()
                        self.assertEqual(response.status, expected)
                        response.read()
                        connection.close()
                    self.assertEqual(model.call_count, 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
