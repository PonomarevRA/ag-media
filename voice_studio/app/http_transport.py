"""HTTP transport for the studio's loopback server."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
import os
import re
import shutil

from voice_studio.app.static_assets import CONTENT_SECURITY_POLICY
from voice_studio.app.transcript_editor import RevisionConflict
from voice_studio.mods import ModError


class LoopbackRequestHandler(BaseHTTPRequestHandler):
    """Validate local requests and send bounded HTTP responses."""

    def end_headers(self):
        """Restrict pages and media to this local origin."""
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        origin = getattr(self, "_live_cors_origin", None)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    @contextmanager
    def request_scope(self, write=False):
        """Reject cross-origin writes and translate input failures into JSON errors."""
        expected = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != expected or (
            write and (self.headers.get("Origin") != f"http://{expected}" or self.headers.get("X-Voice-Studio") != "1")
        ):
            self.reply(403, {"error": "Откройте студию по локальному адресу."})
            yield False
            return
        try:
            yield True
        except (BrokenPipeError, ConnectionResetError):
            return
        except RevisionConflict as exc:
            self.reply(409, {"error": str(exc)})
        except (ValueError, UnicodeError, ModError) as exc:
            self.reply(400, {"error": str(exc)[:300]})
        except Exception as exc:
            print(f"HTTP failure: {exc}", flush=True)
            self.reply(500, {"error": "Ошибка студии. Подробности в терминале."})

    def reply(self, status, data):
        """Return a UTF-8 JSON response."""
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def file(self, path, mime, attachment=None, metadata=None, seekable=False):
        """Stream a local file with a fixed media type."""
        if not path.is_file():
            self.reply(404, {"error": "Файл не найден."})
            return
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            interval = self.byte_interval(size) if seekable else (0, size - 1)
            if interval is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_file(source, mime, attachment, metadata, size, interval, seekable)

    def byte_interval(self, size):
        """Resolve a single optional HTTP range for browser audio seeking."""
        header = self.headers.get("Range")
        if not header:
            return 0, size - 1
        match = re.fullmatch(r"bytes=([0-9]{0,19})-([0-9]{0,19})", header)
        if not match or not any(match.groups()) or not size:
            return None
        first, last = match.groups()
        if not first:
            return (max(0, size - int(last)), size - 1) if int(last) else None
        start = int(first)
        end = min(size - 1, int(last)) if last else size - 1
        return (start, end) if start <= end else None

    def send_file(self, source, mime, attachment, metadata, size, interval, seekable):
        """Send a full file or a bounded byte range from an opened stream."""
        start, end = interval
        length = end - start + 1
        partial = seekable and bool(self.headers.get("Range"))
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        if seekable:
            self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if metadata is not None:
            self.send_header("X-Voice-Processing", json.dumps(metadata))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        if not partial:
            shutil.copyfileobj(source, self.wfile)
            return
        source.seek(start)
        while length:
            chunk = source.read(min(length, 64 * 1024))
            if not chunk:
                break
            self.wfile.write(chunk)
            length -= len(chunk)

    def body(self, limit):
        """Read a bounded upload or JSON request body."""
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= limit:
            raise ValueError("Пустое тело запроса или превышен лимит размера.")
        self.connection.settimeout(30)
        return self.rfile.read(size)
