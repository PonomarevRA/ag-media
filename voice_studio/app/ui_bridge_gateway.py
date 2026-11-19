"""Serve the updated UI/bridge while relaying domain requests to an untouched local Studio."""

import argparse
import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition
from urllib.parse import parse_qs, urlsplit

from voice_studio.app.static_assets import STATIC, CONTENT_SECURITY_POLICY, ROOT
from voice_studio.app.ui_bridge import UIBridge, handle_bridge
from voice_studio.app.wiki_assets import wiki_asset

HOP_HEADERS = frozenset({"connection", "transfer-encoding", "keep-alive", "proxy-authenticate",
                         "proxy-authorization", "te", "trailer", "upgrade"})
LOCAL_HEADERS = frozenset({"content-security-policy", "cache-control", "x-content-type-options"})
FORWARD_HEADERS = ("Content-Type", "Range", "X-Voice-Transcript")


class BridgeGatewayHandler(BaseHTTPRequestHandler):
    """Use current static assets and the in-memory broker, without opening a second Studio database."""

    def log_message(self, format, *args):
        """Keep normal request diagnostics, but never compete for stderr during interpreter teardown."""
        if not self.server.closing:
            super().log_message(format, *args)

    def do_GET(self):
        """Serve owned assets locally; stream other read routes from the fixed backend."""
        url = urlsplit(self.path)
        if self.authorize():
            if not self.serve_asset(url):
                self.forward("GET", 0)

    def do_HEAD(self):
        """Keep backend media/preview availability checks working without a response body."""
        if self.authorize():
            if not self.serve_asset(urlsplit(self.path), head=True):
                self.forward("HEAD", 0)

    def do_POST(self):
        """Handle page/MCP channels here; relay only protected same-origin domain writes."""
        if urlsplit(self.path).path.startswith("/api/ui-bridge/"):
            handle_bridge(self, urlsplit(self.path).path.rsplit("/", 1)[1])
            return
        if not self.authorize(write=True):
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked request uploads are not supported by this local gateway.")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= 2 * 1024 ** 3 + 2 * 1024 ** 2:
                raise ValueError("Invalid gateway upload size.")
            self.forward("POST", size)
        except ValueError:
            self.reply(400, {"error": "invalid_gateway_upload"})

    def authorize(self, *, write=False):
        """Require the gateway's own exact Host and browser write origin before rewriting anything."""
        host = f"127.0.0.1:{self.server.server_port}"
        allowed = self.headers.get("Host") == host
        if write:
            allowed = allowed and self.headers.get("Origin") == "http://" + host and self.headers.get("X-Voice-Studio") == "1"
        if not allowed:
            self.reply(403, {"error": "gateway_origin_required"})
        return allowed

    def end_headers(self):
        """Use the exact same CSP/static response policy as the full Studio."""
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        super().end_headers()

    def reply(self, status, data):
        """Return a small UTF-8 JSON error/result without retaining request contents."""
        content = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def body(self, limit):
        """Bound bridge JSON before reading it; domain media uses the streaming path instead."""
        size = int(self.headers.get("Content-Length", "0"))
        if self.headers.get("Transfer-Encoding") or not 0 <= size <= limit:
            raise ValueError("Invalid bridge request length.")
        content = self.rfile.read(size)
        if len(content) != size:
            raise ValueError("Incomplete bridge request.")
        return content

    def serve_asset(self, url, *, head=False):
        """Reuse the allowlisted catalogue/wiki resolver; user media always stays on the backend."""
        if url.path in {"/wiki", "/wiki/"}:
            locale = parse_qs(url.query).get("ui", ["ru"])[0]
            self.send_response(302)
            self.send_header("Location", "/wiki/Home.html" if locale == "en" else "/wiki/Home-ru.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True
        if url.path in STATIC:
            name, mime = STATIC[url.path]
            asset = (ROOT / "web" / name, mime)
        elif url.path.startswith("/wiki/"):
            asset = wiki_asset(url.path, parse_qs(url.query).get("ui", ["ru"])[0])
        else:
            return False
        if not asset or not asset[0].is_file():
            self.reply(404, {"error": "ui_asset_unavailable"})
            return True
        path, mime = asset
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        if not head:
            with path.open("rb") as source:
                while chunk := source.read(65_536):
                    self.wfile.write(chunk)
        return True

    def forward(self, method, size):
        """Stream literal bytes in bounded buffers to one configured loopback port, never a supplied URL."""
        url = urlsplit(self.path)
        if url.scheme or url.netloc or not self.path.startswith("/"):
            self.reply(400, {"error": "invalid_gateway_route"})
            return
        connection = http.client.HTTPConnection("127.0.0.1", self.server.backend_port, timeout=1020)
        responded = False
        try:
            headers = {key: self.headers[key] for key in FORWARD_HEADERS if key in self.headers}
            if method == "POST":
                # This is reached only after the gateway's exact Host/Origin/write check.
                headers.update({"Content-Length": str(size), "X-Voice-Studio": "1",
                                "Origin": f"http://127.0.0.1:{self.server.backend_port}"})
            connection.putrequest(method, self.path)
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            remaining = size
            while remaining:
                chunk = self.rfile.read(min(65_536, remaining))
                if not chunk:
                    raise ValueError("Incomplete gateway upload.")
                connection.send(chunk)
                remaining -= len(chunk)
            response = connection.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS | LOCAL_HEADERS:
                    self.send_header(key, value)
            self.end_headers()
            responded = True
            if method != "HEAD":
                while chunk := response.read1(65_536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not responded:
                self.reply(502, {"error": "ui_bridge_backend_unavailable"})
        finally:
            connection.close()


class GatewayServer(ThreadingHTTPServer):
    """Own only this gateway socket and broker; never instantiate or signal the backend."""

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, handler):
        """Create one small broker without importing any model or Studio data services."""
        self.ui_bridge = UIBridge()
        self.closing = False
        self.request_condition = Condition()
        self.request_count = 0
        super().__init__(address, handler)

    def process_request(self, request, address):
        """Count accepted workers before starting threads, including workers not yet scheduled."""
        with self.request_condition:
            if self.closing:
                self.shutdown_request(request)
                return
            self.request_count += 1
        try:
            super().process_request(request, address)
        except BaseException:
            self.request_finished()
            raise

    def process_request_thread(self, request, address):
        """Release the drain barrier only after the handler and socket have finished."""
        try:
            super().process_request_thread(request, address)
        finally:
            self.request_finished()

    def request_finished(self):
        """Wake the closing owner after each accepted worker releases its socket."""
        with self.request_condition:
            self.request_count -= 1
            self.request_condition.notify_all()

    def handle_error(self, request, address):
        """Report operational failures normally; teardown disconnects cannot write after process exit."""
        if not self.closing:
            super().handle_error(request, address)

    def server_close(self):
        """Wake broker callers and drain short handlers for up to two seconds, without stopping the backend."""
        with self.request_condition:
            self.closing = True
        self.ui_bridge.close()
        super().server_close()
        # Long upstream analyses remain daemon workers; they must not keep this companion alive.
        with self.request_condition:
            self.request_condition.wait_for(lambda: self.request_count == 0, timeout=2)


def create_gateway(port, backend_port):
    """Bind an independent socket and retain only the fixed backend port, with no workspace lock override."""
    if not 0 <= port <= 65535 or not 1 <= backend_port <= 65535 or port == backend_port:
        raise ValueError("Choose distinct valid gateway and backend ports.")
    server = GatewayServer(("127.0.0.1", port), BridgeGatewayHandler)
    if server.server_port == backend_port:
        server.server_close()
        raise ValueError("Gateway cannot relay to its own bound port.")
    server.backend_port = backend_port
    return server


def main():
    """Start a companion UI without restarting, signalling or loading models in the existing Studio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--backend-port", type=int, default=8766)
    options = parser.parse_args()
    with create_gateway(options.port, options.backend_port) as server:
        print(json.dumps({"schema": "ag-media-ui-bridge-gateway-v1", "pid": os.getpid(),
                          "origin": f"http://127.0.0.1:{server.server_port}",
                          "backendOrigin": f"http://127.0.0.1:{options.backend_port}",
                          "ownsBackend": False}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
