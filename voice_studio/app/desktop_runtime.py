"""Owned loopback service for the native desktop hosts; never attaches to another instance."""

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
from uuid import uuid4
from urllib.parse import urlsplit

from voice_paths import application_data, ensure_temporary_root, workspace_temp
from voice_studio.core.workspace_lock import workspace_lock


READY_SCHEMA = "ag-media-desktop-ready-v1"


def parser():
    """Describe the identical service launch contract used by all native hosts."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=Path(os.environ.get("VOICE_STUDIO_DATA", application_data() / "studio")))
    result.add_argument("--port", type=int, default=18766)
    result.add_argument("--demo", action="store_true")
    return result


def create_server(root, port, demo):
    """Bind an owned loopback socket, falling back only when the preferred port is occupied."""
    from voice_studio.app.server import Handler, StudioServer

    class DesktopHandler(Handler):
        """Expose native commands within the owned service's same-origin boundary."""

        def do_GET(self):
            """Return only this bound service's identity, then delegate ordinary application routes."""
            if urlsplit(self.path).path == "/api/desktop-instance":
                with self.request_scope() as allowed:
                    if allowed:
                        self.reply(200, self.server.desktop_identity)
                return
            super().do_GET()

        def do_POST(self):
            """Check releases only on an explicit request to the native instance."""
            if self.path == "/api/desktop-update":
                with self.request_scope(write=True) as allowed:
                    if allowed:
                        from voice_studio.app.desktop_updates import check_desktop_update
                        self.reply(200, check_desktop_update())
                return
            super().do_POST()
    from voice_studio.app.studio import Studio
    from voice_studio.app.scroll_previews import ScrollPreviews
    from voice_studio.app.live_capture import LiveCapture
    from voice_studio.app.studio_storage import StudioStorage
    try:
        server = StudioServer(("127.0.0.1", port), DesktopHandler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE or not port:
            raise
        server = StudioServer(("127.0.0.1", 0), DesktopHandler)
    # Requests may be doing bounded model I/O; the native owner's final deadline
    # must not wait indefinitely for ThreadingHTTPServer's request thread joins.
    server.daemon_threads = True
    try:
        server.studio = Studio(root, demo)
        server.scroll_previews = ScrollPreviews(workspace_temp(root) / "scroll-previews")
        server.live_capture = LiveCapture(root)
        server.storage = StudioStorage(root, root)
        server.stop_storage = server.storage.start()
    except BaseException:
        server.server_close()
        raise
    return server


def readiness(server, root):
    """Report the bound origin and actual core tools without loading or downloading models."""
    port = server.server_port
    return {"schema": READY_SCHEMA, "pid": os.getpid(), "port": port,
            "origin": f"http://127.0.0.1:{port}", "data": str(root),
            "python": sys.executable, "instanceId": uuid4().hex, "capabilities": {
                "nativeCore": True, "automaticDownloads": False,
                "ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe")),
                "modelsBundled": False, "engines": "operator-provisioned"}}


def run_service(options):
    """Serve until the owner sends quit or an OS signal, then release only owned resources."""
    root = options.data.expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ensure_temporary_root()
    stop = threading.Event()
    previous = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, lambda *_: stop.set())

    def control():
        """Accept one fixed command from the owner's private stdin pipe, never arbitrary code."""
        for line in sys.stdin:
            if line.strip() == "quit":
                stop.set()
                return
        # Native hosts retain a private stdin pipe; EOF means that owner has gone.
        stop.set()

    try:
        with workspace_lock(root):
            server = create_server(root, options.port, options.demo)
            identity = readiness(server, root)
            server.desktop_identity = identity
            metadata = root / "desktop-instance.json"
            metadata.write_text(json.dumps(identity) + "\n")
            worker = threading.Thread(target=server.serve_forever, name="native-studio-http", daemon=True)
            worker.start()
            threading.Thread(target=control, name="native-studio-control", daemon=True).start()
            print(json.dumps(identity), flush=True)
            try:
                while worker.is_alive() and not stop.wait(.1):
                    pass
            finally:
                server.shutdown()
                server.stop_storage.set()
                server.live_capture.close()
                server.studio.worker.shutdown(wait=False, cancel_futures=True)
                from voice_studio.speech.embedding_worker import close_worker
                close_worker()
                server.server_close()
                try:
                    current = json.loads(metadata.read_text())
                    if current.get("instanceId") == identity["instanceId"]:
                        metadata.unlink()
                except (OSError, ValueError):
                    pass
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    """Keep native app writes private and start its external, per-user workspace."""
    os.umask(0o077)
    if os.name == "posix":
        try:
            os.setsid()
        except PermissionError:
            pass
    run_service(parser().parse_args())


if __name__ == "__main__":
    main()
