"""Exercise page identity, bounded delivery and real local HTTP/MCP transport boundaries."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import unittest
import urllib.error
import urllib.request

from voice_studio.app.ui_bridge import BridgeError, UIBridge, UI_TOOLS
from voice_studio.app.server import Handler, StudioServer

REVISION = "a" * 32
STATE = {"revision": REVISION, "ui": {"page": "/"}}
TOOLS = [{"name": name, "inputSchema": {"type": "object"}} for name in sorted(UI_TOOLS)]


class UiBridgeTests(unittest.TestCase):
    """Use synthetic page state; no recordings, model runs or live user workspaces."""

    def setUp(self):
        """Give each check a short independent deadline and fresh window channel."""
        self.bridge = UIBridge(call_timeout=.3, poll_timeout=.05)
        self.channel = self.bridge.register({"tools": TOOLS, "state": STATE})

    def call(self, **extra):
        """Form a semantic mutation using this window's last inspected revision."""
        return {"windowId": self.channel["windowId"], "name": "ui_navigate",
                "revision": REVISION, "arguments": {"page": "settings"}, **extra}

    def poll(self, **extra):
        """Use the private page channel, which is never returned to MCP clients."""
        return self.bridge.poll({**self.channel, "state": STATE, **extra})

    def test_correlated_delivery_and_result(self):
        """The page receives one command, and only its matching ID releases the caller."""
        with ThreadPoolExecutor() as workers:
            result = workers.submit(self.bridge.call, self.call())
            command = self.poll()["command"]
            self.assertEqual(command["name"], "ui_navigate")
            self.poll(completed={"id": "wrong", "result": {"ok": False}})
            self.assertFalse(result.done())
            self.poll(completed={"id": command["id"], "result": {"status": "ok", "revision": REVISION}})
            self.assertEqual(result.result()["status"], "ok")

    def test_wrong_token_context_and_tool_rejected(self):
        """A wrong page token, stale context and script-like tool name cannot dispatch."""
        for operation, code in ((lambda: self.poll(token="wrong"), "ui_page_token_invalid"),
                                (lambda: self.bridge.call(self.call(revision="old")), "ui_context_changed"),
                                (lambda: self.bridge.call(self.call(name="eval")), "ui_tool_not_allowed")):
            with self.assertRaisesRegex(BridgeError, code):
                operation()
        self.assertIsNone(self.poll()["command"])

    def test_page_requires_token_even_when_explicitly_null(self):
        """Absent/null/wrong page tokens cannot alter state or revoke a channel; internal MCP lookup still works."""
        for token in ({}, {"token": None}, {"token": ""}, {"token": "wrong"}):
            data = {"windowId": self.channel["windowId"], "state": {**STATE, "revision": "b" * 32}, **token}
            for operation in (self.bridge.poll, self.bridge.disconnect):
                with self.subTest(token=token, operation=operation.__name__):
                    with self.assertRaisesRegex(BridgeError, "ui_page_token_invalid"):
                        operation(data)
                    self.assertEqual(self.bridge.windows()["windows"][0]["state"], STATE)
        self.assertIsNone(self.poll()["command"])
        self.assertEqual(self.bridge.disconnect(self.channel), {"ok": True})

    def test_busy_timeout_no_replay_and_disconnect(self):
        """One outstanding command per window; delivered timed-out commands are not replayed."""
        with ThreadPoolExecutor() as workers:
            result = workers.submit(self.bridge.call, self.call())
            command = self.poll()["command"]
            self.assertIsNotNone(command)
            with self.assertRaisesRegex(BridgeError, "ui_window_busy"):
                self.bridge.call(self.call())
            with self.assertRaisesRegex(BridgeError, "ui_timeout_outcome_unknown"):
                result.result()
            self.assertIsNone(self.poll()["command"])
            second = workers.submit(self.bridge.call, self.call())
            self.poll()
            self.bridge.disconnect(self.channel)
            self.assertEqual(second.result()["error"], "ui_window_closed")
        with self.assertRaisesRegex(BridgeError, "ui_window_unavailable"):
            self.bridge.call(self.call())

    def test_capacity_expiry_and_private_tokens(self):
        """Window enumeration omits channel secrets; fresh sessions have a hard capacity."""
        for _ in range(7):
            self.bridge.register({"tools": TOOLS, "state": STATE})
        with self.assertRaisesRegex(BridgeError, "ui_window_capacity"):
            self.bridge.register({"tools": TOOLS, "state": STATE})
        self.assertNotIn(self.channel["token"], json.dumps(self.bridge.windows()))
        for entry in self.bridge.sessions.values():
            entry["seen"] -= 61
        self.assertEqual(self.bridge.windows()["windows"], [])

    def test_invalid_state_does_not_poison_session(self):
        """Rejected page state cannot replace the last valid context."""
        with self.assertRaises(BridgeError):
            self.poll(state={"revision": None})
        self.assertEqual(self.bridge.windows()["windows"][0]["state"], STATE)
        with self.assertRaises(BridgeError):
            self.bridge.register({"tools": [{"name": "voice_delete"}], "state": STATE})
        with self.assertRaises(BridgeError):
            self.bridge.register({"tools": TOOLS, "state": {**STATE, "large": "x" * 70_000}})

    def test_close_wakes_long_poll(self):
        """HTTP shutdown is not delayed by a page's idle long poll."""
        self.bridge.poll_timeout = 20
        with ThreadPoolExecutor() as workers:
            result = workers.submit(self.poll)
            deadline = time.monotonic() + 1
            while not self.bridge.sessions[self.channel["windowId"]]["polling"] and time.monotonic() < deadline:
                time.sleep(.001)
            self.bridge.close()
            with self.assertRaisesRegex(BridgeError, "ui_window_unavailable"):
                result.result(timeout=.5)


class UiBridgeHttpTests(unittest.TestCase):
    """Exercise actual protected routes without constructing a model or Studio database."""

    def setUp(self):
        """Bind only an isolated loopback server and broker."""
        self.server = StudioServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        """Release only this test listener and its long polls."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def post(self, route, data, **headers):
        """Send literal JSON with caller-selected transport headers."""
        request = urllib.request.Request(self.origin + "/api/ui-bridge/" + route,
            data=json.dumps(data).encode(), headers={"Content-Type": "application/json", **headers})
        with self.opener.open(request, timeout=2) as response:
            return json.load(response)

    def test_browser_and_mcp_are_separate(self):
        """Browser markers cannot reach MCP control, and MCP markers cannot forge a page channel."""
        page = {"Origin": self.origin, "X-Voice-Studio": "1"}
        mcp = {"X-Voice-Studio-MCP": "b" * 32}
        self.post("register", {"tools": TOOLS, "state": STATE}, **page)
        self.assertEqual(len(self.post("windows", {}, **mcp)["windows"]), 1)
        for route, headers, body in (("windows", page, {}), ("register", mcp, {}),
                                     ("windows", {**mcp, "Origin": "https://example.invalid"}, {}),
                                     ("register", {**page, "Origin": "https://example.invalid"}, {})):
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.post(route, body, **headers)
            self.assertEqual(rejected.exception.code, 403)

    def test_http_page_token_null_is_rejected_before_mutation(self):
        """JSON null must stay an invalid page credential through the actual HTTP handler."""
        page = {"Origin": self.origin, "X-Voice-Studio": "1"}
        mcp = {"X-Voice-Studio-MCP": "b" * 32}
        channel = self.post("register", {"tools": TOOLS, "state": STATE}, **page)
        for token in ({}, {"token": None}, {"token": ""}, {"token": "wrong"}):
            data = {"windowId": channel["windowId"], "state": {**STATE, "revision": "b" * 32}, **token}
            for operation in ("poll", "disconnect"):
                with self.subTest(token=token, operation=operation):
                    with self.assertRaises(urllib.error.HTTPError) as rejected:
                        self.post(operation, data, **page)
                    self.assertEqual(rejected.exception.code, 403)
                    self.assertEqual(self.post("windows", {}, **mcp)["windows"][0]["state"], STATE)
        self.server.ui_bridge.poll_timeout = 0
        self.assertEqual(self.post("poll", {**channel, "state": STATE}, **page), {"command": None})
        self.assertEqual(self.post("disconnect", channel, **page), {"ok": True})

    def test_fresh_mcp_schemas_and_ui_catalogue(self):
        """Fresh MCP discovery publishes the bridge tools under the UI category."""
        import asyncio
        import tempfile
        from voice_studio.mcp.voice_mcp import StudioClient, build_server
        from voice_studio.mcp.voice_catalog import read_catalog
        with tempfile.TemporaryDirectory() as folder:
            mcp = build_server(StudioClient(self.server.server_port, Path(folder)))
            catalogue = asyncio.run(read_catalog(mcp, "ui", "live"))
            self.assertTrue({"ui_windows", "ui_inspect", "ui_execute", "ui_measure"} <=
                            {row["name"] for row in catalogue["tools"]})

    def test_mcp_redirect_is_not_followed(self):
        """A redirect from the chosen endpoint cannot forward a command elsewhere."""
        from http.server import BaseHTTPRequestHandler
        from voice_studio.mcp.ui_bridge_mcp import NoBridgeRedirect
        observed = []

        class RedirectHandler(BaseHTTPRequestHandler):
            """Record requests and offer a redirect that the bridge must decline."""

            def do_GET(self):
                """Return one relative redirect; any second request would reveal forwarding."""
                observed.append(self.path)
                self.send_response(302)
                self.send_header("Location", "/unexpected")
                self.end_headers()

        self.server.RequestHandlerClass = RedirectHandler
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoBridgeRedirect())
        with self.assertRaises(urllib.error.HTTPError) as rejected:
            opener.open(self.origin + "/selected", timeout=2)
        self.assertEqual(rejected.exception.code, 302)
        self.assertEqual(observed, ["/selected"])


if __name__ == "__main__":
    unittest.main()
