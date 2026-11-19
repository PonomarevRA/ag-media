"""Bounded, in-memory rendezvous between local MCP and an explicitly connected page."""

import json
import re
import time
from threading import Condition, Event
from uuid import uuid4
from voice_studio.core.tool_names import UI_TOOL_NAMES

UI_TOOLS = UI_TOOL_NAMES
MAX_BYTES = 65_536
_MISSING_PAGE_TOKEN = object()


class BridgeError(ValueError):
    """Carry a finite HTTP status and machine-readable failure without private input."""

    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def bounded_object(value):
    """Require a JSON object of at most 64 KiB; reject non-finite numeric values."""
    if not isinstance(value, dict):
        raise BridgeError("invalid_bridge_object", 400)
    try:
        size = len(json.dumps(value, allow_nan=False).encode())
    except (ValueError, TypeError):
        raise BridgeError("invalid_bridge_object", 400) from None
    if size > MAX_BYTES:
        raise BridgeError("bridge_payload_too_large", 413)
    return value


def page_state(value):
    """Validate a page context before replacing the last known valid snapshot."""
    state = bounded_object(value)
    if not isinstance(state.get("revision"), str) or not re.fullmatch(r"[a-f0-9]{32}", state["revision"]):
        raise BridgeError("invalid_ui_revision", 400)
    return state


class UIBridge:
    """Keep at most eight live page sessions, one operation per page, and no replay queue."""

    def __init__(self, *, ttl=60, call_timeout=15, poll_timeout=20):
        self.ttl, self.call_timeout, self.poll_timeout = ttl, call_timeout, poll_timeout
        self.condition = Condition()
        self.sessions = {}
        self.closed = False

    def _retire(self, identity):
        """Wake callers with an explicit closed result before dropping a session."""
        entry = self.sessions.pop(identity)
        if entry["pending"]:
            entry["pending"]["result"] = {"status": "error", "error": "ui_window_closed"}
            entry["pending"]["done"].set()
        self.condition.notify_all()

    def _prune(self):
        """Expire abandoned pages only; never evict a fresh window to admit another."""
        now = time.monotonic()
        for identity, entry in list(self.sessions.items()):
            if now - entry["seen"] > self.ttl:
                self._retire(identity)

    def _entry(self, identity, token=_MISSING_PAGE_TOKEN):
        """Fence page ownership separately from the public MCP window identity."""
        self._prune()
        entry = self.sessions.get(identity)
        if self.closed or entry is None:
            raise BridgeError("ui_window_unavailable", 404)
        # Only an omitted internal argument skips page ownership. JSON null is an invalid token.
        if token is not _MISSING_PAGE_TOKEN and token != entry["token"]:
            raise BridgeError("ui_page_token_invalid", 403)
        return entry

    def register(self, data):
        """Connect a page with its bounded shared tool catalogue and context snapshot."""
        bounded_object(data)
        tools = data.get("tools")
        if (not isinstance(tools, list) or len(tools) > len(UI_TOOLS)
                or any(not isinstance(row, dict) or not isinstance(row.get("name"), str)
                       or row["name"] not in UI_TOOLS for row in tools)
                or len({row["name"] for row in tools}) != len(tools)):
            raise BridgeError("invalid_ui_tools", 400)
        state = page_state(data.get("state"))
        with self.condition:
            self._prune()
            if self.closed or len(self.sessions) >= 8:
                raise BridgeError("ui_window_capacity", 503)
            identity, token = uuid4().hex, uuid4().hex
            self.sessions[identity] = {"token": token, "seen": time.monotonic(), "tools": tools,
                                       "state": state, "pending": None, "polling": False}
            return {"windowId": identity, "token": token}

    def windows(self):
        """Expose live identities and bounded state, never private page-channel tokens."""
        with self.condition:
            self._prune()
            return {"windows": [{"windowId": key, "state": value["state"], "tools": value["tools"],
                                  "busy": value["pending"] is not None,
                                  "observedSecondsAgo": round(time.monotonic() - value["seen"], 2)}
                                 for key, value in self.sessions.items()]}

    def poll(self, data):
        """Accept a correlated completion and long-poll for one command without busy polling."""
        bounded_object(data)
        with self.condition:
            entry = self._entry(data.get("windowId"), data.get("token", ""))
            if entry["polling"]:
                raise BridgeError("ui_page_poll_busy")
            entry["state"] = page_state(data.get("state"))
            entry["seen"] = time.monotonic()
            completed, pending = data.get("completed"), entry["pending"]
            if completed is not None:
                if not isinstance(completed, dict):
                    raise BridgeError("invalid_ui_completion", 400)
                if pending and completed.get("id") == pending["command"]["id"]:
                    pending["result"] = bounded_object(completed.get("result"))
                    pending["done"].set()
            entry["polling"] = True
            deadline = time.monotonic() + self.poll_timeout
            try:
                while True:
                    self._entry(data.get("windowId"), data.get("token", ""))
                    pending = entry["pending"]
                    if pending and not pending["delivered"]:
                        pending["delivered"] = True
                        return {"command": pending["command"]}
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {"command": None}
                    self.condition.wait(remaining)
            finally:
                entry["polling"] = False

    def call(self, data):
        """Dispatch to an explicit window; timeout is an unknown outcome, never automatic retry."""
        bounded_object(data)
        name = data.get("name")
        if not isinstance(name, str) or name not in UI_TOOLS:
            raise BridgeError("ui_tool_not_allowed", 400)
        bounded_object(data.get("arguments", {}))
        with self.condition:
            entry = self._entry(data.get("windowId"))
            if name not in {row["name"] for row in entry["tools"]}:
                raise BridgeError("ui_tool_unavailable", 400)
            if name != "ui_inspect" and data.get("revision") != entry["state"].get("revision"):
                raise BridgeError("ui_context_changed")
            if entry["pending"]:
                raise BridgeError("ui_window_busy")
            command = {"id": uuid4().hex, "name": name, "arguments": data.get("arguments", {}),
                       "revision": data.get("revision"),
                       "expiresAtMs": int((time.time() + self.call_timeout) * 1000)}
            pending = {"command": command, "done": Event(), "result": None, "delivered": False}
            entry["pending"] = pending
            self.condition.notify_all()
        try:
            if not pending["done"].wait(self.call_timeout):
                raise BridgeError("ui_timeout_outcome_unknown", 504)
            return pending["result"]
        finally:
            with self.condition:
                if entry["pending"] is pending:
                    entry["pending"] = None
                self.condition.notify_all()

    def disconnect(self, data):
        """Revoke one page channel immediately, including its in-flight operation."""
        with self.condition:
            self._entry(data.get("windowId"), data.get("token", ""))
            self._retire(data["windowId"])
        return {"ok": True}

    def close(self):
        """Wake all long polls/callers before the HTTP server joins its request threads."""
        with self.condition:
            self.closed = True
            for identity in list(self.sessions):
                self._retire(identity)


def handle_bridge(handler, operation):
    """Separate same-origin page channels from non-browser local MCP control requests."""
    expected = f"127.0.0.1:{handler.server.server_port}"
    headers = handler.headers
    page = operation in {"register", "poll", "disconnect"}
    allowed = headers.get("Host") == expected
    if page:
        allowed = allowed and headers.get("Origin") == "http://" + expected and headers.get("X-Voice-Studio") == "1"
    else:
        allowed = allowed and "Origin" not in headers and "Sec-Fetch-Site" not in headers
        allowed = allowed and bool(re.fullmatch(r"[a-f0-9]{32}", headers.get("X-Voice-Studio-MCP", "")))
    if not allowed:
        handler.reply(403, {"error": "ui_bridge_transport_required"})
        return
    try:
        data = bounded_object(json.loads(handler.body(MAX_BYTES)))
        bridge = handler.server.ui_bridge
        operations = {"register": bridge.register, "poll": bridge.poll, "disconnect": bridge.disconnect,
                      "windows": lambda _: bridge.windows(), "call": bridge.call}
        if operation not in operations:
            raise BridgeError("unknown_bridge_operation", 404)
        handler.reply(200, operations[operation](data))
    except BridgeError as error:
        handler.reply(error.status, {"error": error.code})
    except (ValueError, TypeError, UnicodeError, RecursionError):
        handler.reply(400, {"error": "invalid_bridge_request"})
    except (BrokenPipeError, ConnectionResetError):
        pass
