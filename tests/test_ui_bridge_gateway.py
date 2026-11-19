"""Verify a model-free companion gateway against an owned synthetic backend."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request

from voice_studio.app.ui_bridge_gateway import BridgeGatewayHandler, create_gateway
from voice_studio.app.static_assets import CONTENT_SECURITY_POLICY, STATIC

ROOT = Path(__file__).resolve().parents[1]


class GatewayTests(unittest.TestCase):
    """Exercise real byte streaming and origin checks without a Studio workspace."""

    def setUp(self):
        """Start only two test-owned listeners, with a controllable streaming response."""
        self.seen = []
        self.release = threading.Event()
        owner = self

        class Backend(BaseHTTPRequestHandler):
            """Return deterministic test bytes and record the forwarded trust boundary."""

            def do_GET(self):
                """Expose a stream or synthetic range response, never a local file."""
                owner.seen.append(dict(self.headers))
                if self.path == "/stream":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"first\n")
                    self.wfile.flush()
                    owner.release.wait(2)
                    self.wfile.write(b"last\n")
                    return
                self.send_response(206)
                self.send_header("Content-Length", "3")
                self.send_header("Content-Range", "bytes 1-3/5")
                self.end_headers()
                self.wfile.write(b"bcd")

            def do_POST(self):
                """Echo only the fixture body and selected forwarding headers."""
                owner.seen.append(dict(self.headers))
                content = self.rfile.read(int(self.headers["Content-Length"]))
                reply = json.dumps({"body": content.decode(), "origin": self.headers.get("Origin"),
                                    "marker": self.headers.get("X-Voice-Studio")}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        self.backend.daemon_threads = True
        self.gateway = create_gateway(0, self.backend.server_port)
        self.threads = [threading.Thread(target=server.serve_forever, daemon=True)
                        for server in (self.backend, self.gateway)]
        for worker in self.threads:
            worker.start()
        self.origin = f"http://127.0.0.1:{self.gateway.server_port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        """Close the gateway and synthetic backend; never target the real Studio."""
        self.release.set()
        for server in (self.gateway, self.backend):
            server.shutdown()
            server.server_close()
        for worker in self.threads:
            worker.join(2)

    def test_protected_forwarding_and_shared_assets(self):
        """Rewrite origin only after checking the incoming browser and preserve exact JSON bytes."""
        request = urllib.request.Request(self.origin + "/api/fixture", data=b"fixture",
            headers={"Origin": self.origin, "X-Voice-Studio": "1"})
        with self.opener.open(request) as response:
            result = json.load(response)
            self.assertEqual(response.headers["Content-Security-Policy"], CONTENT_SECURITY_POLICY)
        self.assertEqual(result, {"body": "fixture", "origin": f"http://127.0.0.1:{self.backend.server_port}", "marker": "1"})
        count = len(self.seen)
        for headers in ({}, {"Origin": "https://example.invalid", "X-Voice-Studio": "1"}):
            request = urllib.request.Request(self.origin + "/api/fixture", data=b"rejected", headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.opener.open(request)
            self.assertEqual(rejected.exception.code, 403)
        self.assertEqual(len(self.seen), count)
        with self.opener.open(self.origin + "/mcp-bridge.js") as response:
            self.assertEqual(response.read(), (ROOT / "web" / STATIC["/mcp-bridge.js"][0]).read_bytes())
        self.assertEqual(len(self.seen), count)

    def test_wiki_directory_redirect_keeps_relative_assets(self):
        """Resolve wiki entry points to a document URL so relative CSS and links remain inside /wiki/."""
        with self.opener.open(self.origin + "/wiki?ui=en") as response:
            self.assertTrue(response.url.endswith("/wiki/Home.html"))
            self.assertIn(b'href="wiki.css"', response.read())
        with self.opener.open(self.origin + "/wiki/wiki.css") as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/css", response.headers["Content-Type"])
        self.assertEqual(self.seen, [])

    def test_range_and_live_progress_are_not_buffered(self):
        """Media range headers survive, and progress reaches a client before backend completion."""
        request = urllib.request.Request(self.origin + "/media/fixture", headers={"Range": "bytes=1-3"})
        with self.opener.open(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 1-3/5")
            self.assertEqual(response.read(), b"bcd")
        self.assertEqual(self.seen[-1]["Range"], "bytes=1-3")
        with ThreadPoolExecutor() as workers:
            def first_chunk():
                """Read one complete progress item while the upstream remains blocked."""
                with self.opener.open(self.origin + "/stream", timeout=2) as response:
                    return response.read(6)
            result = workers.submit(first_chunk)
            try:
                self.assertEqual(result.result(timeout=1), b"first\n")
            finally:
                self.release.set()

    def test_gateway_import_is_lightweight(self):
        """A fresh interpreter imports the gateway without Studio/model/tracking frameworks."""
        code = "import sys;sys.path.insert(0,sys.argv[1]);import voice_studio.app.ui_bridge_gateway;assert not any(x in sys.modules for x in ('voice_studio.app.server','torch','mlflow','numpy'))"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_closing_gateway_preserves_backend(self):
        """The companion owns only its socket, so its teardown cannot stop the backend."""
        self.gateway.shutdown()
        self.gateway.server_close()
        with self.opener.open(f"http://127.0.0.1:{self.backend.server_port}/still-running") as response:
            self.assertEqual(response.read(), b"bcd")

    def test_close_drains_short_handler_finalization(self):
        """A fully delivered response can still own a daemon worker; close waits for that worker's cleanup."""
        entered, release = threading.Event(), threading.Event()

        class FinalizingHandler(BridgeGatewayHandler):
            """Hold cleanup after sending a real response to expose the shutdown race deterministically."""

            def finish(self):
                """Simulate bounded handler cleanup after HTTP bytes are already delivered."""
                super().finish()
                entered.set()
                release.wait(3)

        self.gateway.RequestHandlerClass = FinalizingHandler
        with self.opener.open(self.origin + "/mcp-bridge.js") as response:
            response.read()
        self.assertTrue(entered.wait(1))
        self.gateway.shutdown()
        with ThreadPoolExecutor() as workers:
            closed = workers.submit(self.gateway.server_close)
            try:
                with self.assertRaises(TimeoutError):
                    closed.result(timeout=.05)
            finally:
                release.set()
            closed.result(timeout=1)


if __name__ == "__main__":
    unittest.main()
