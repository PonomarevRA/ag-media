"""Exercise local range responses and request validation through the studio handler."""

import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
from unittest import TestCase

from voice_studio.app.server import Handler, StudioServer


class HttpTransportTests(TestCase):
    """Test the actual studio routes with a synthetic WAV payload."""

    def test_file_ranges_and_local_write_validation(self):
        """Moving the transport preserves range bytes, response headers and request validation."""
        key = "a" * 32
        with TemporaryDirectory() as folder, StudioServer(("127.0.0.1", 0), Handler) as server:
            media = Path(folder)
            (media / f"{key}.wav").write_bytes(b"0123456789")
            server.studio = SimpleNamespace(root=media, media=media, demo=False,
                                            rows=lambda *_: [{"visible": True}])
            worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            worker.start()
            try:
                self.check_ranges(server.server_port, f"/media/{key}.wav")
                self.check_write_scope(server.server_port)
            finally:
                server.shutdown()
                worker.join(timeout=5)

    def check_ranges(self, port, path):
        """Verify whole files, standard and suffix ranges, and rejection of multiple ranges."""
        status, headers, body = self.request(port, "GET", path)
        self.assertEqual((status, body), (200, b"0123456789"))
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertTrue(headers["Content-Security-Policy"])

        cases = (("bytes=3-5", 206, "bytes 3-5/10", b"345"),
                 ("bytes=-2", 206, "bytes 8-9/10", b"89"),
                 ("bytes=1-2,4-5", 416, "bytes */10", b""))
        for header, expected_status, expected_range, expected_body in cases:
            with self.subTest(header=header):
                status, headers, body = self.request(port, "GET", path, Range=header)
                self.assertEqual((status, headers["Content-Range"], body),
                                 (expected_status, expected_range, expected_body))

    def check_write_scope(self, port):
        """An authorized empty upload reaches validation while an untagged write is rejected."""
        path = "/api/transcribe"
        status, _, body = self.request(port, "POST", path, b"")
        self.assertEqual(status, 403)
        self.assertIn("error", json.loads(body))
        headers = {"Origin": f"http://127.0.0.1:{port}", "X-Voice-Studio": "1"}
        status, _, body = self.request(port, "POST", path, b"", **headers)
        self.assertEqual(status, 400)
        self.assertIn("лимит", json.loads(body)["error"])

    @staticmethod
    def request(port, method, path, body=None, **headers):
        """Read one complete HTTP response and close the local connection."""
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()
