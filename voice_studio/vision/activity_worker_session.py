"""Reuse the local Light-ASD model across bounded, synchronous requests."""

from voice_studio.core.optional_runtime import runtime_environment
from voice_studio.mod_runtime import worker_environment

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import subprocess
from time import monotonic

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class ActivityWorkerSession:
    """Restart the child after a protocol error or timeout without losing completed windows."""

    def __init__(self, command, imports):
        """Own one lazy child and one IO thread for sequential window requests."""
        self.command = [*command, "--window-server"]
        self.env = runtime_environment("vision", {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, imports))})
        self.process = None
        self.reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="light-asd-window")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        """Reap the owned child before joining IO, including a timed-out exchange."""
        try:
            self.close_child()
        finally:
            self.reader.shutdown()

    def close_child(self):
        """Terminate the owned worker to release blocked reads and writes before reuse."""
        if self.process is None:
            return
        process, self.process = self.process, None
        if process.poll() is None:
            process.kill()
        process.wait()
        # Closing buffered stdin may flush to a child we just killed. Still
        # release the sibling stream and preserve the original request failure.
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    @staticmethod
    def _exchange(process, request):
        """Write one bounded request and validate one complete dictionary response."""
        process.stdin.write(request)
        process.stdin.flush()
        raw = process.stdout.readline(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES or not raw.endswith(b"\n"):
            raise ValueError("Incomplete or oversized Light-ASD response")
        packet = json.loads(raw)
        if not isinstance(packet, dict) or not isinstance(packet.get("result"), dict):
            raise ValueError("Light-ASD window failed")
        return packet["result"]

    def request(self, arguments, timeout):
        """Exchange one window within timeout seconds, including blocked stdin/stdout.

        Requests are sequential, limited to 64 KiB of JSON; responses are at most
        16 MiB and contain a dictionary result. A timeout or protocol failure retires
        this session's child. The next explicit request starts a fresh worker.
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Light-ASD timeout must be finite and positive")
        started = monotonic()
        request = json.dumps(list(map(str, arguments))).encode() + b"\n"
        if len(request) > MAX_REQUEST_BYTES:
            raise ValueError("Light-ASD request exceeds limit")
        try:
            if self.process is None:
                self.process = subprocess.Popen(self.command, env=worker_environment(self.env), stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            # Both directions run off the deadline-owning caller. Killing only
            # our child releases its pipes, so timeout cleanup can join this IO.
            exchange = self.reader.submit(self._exchange, self.process, request)
            return exchange.result(timeout=max(0, timeout - (monotonic() - started)))
        except BaseException:
            self.close_child()
            raise
