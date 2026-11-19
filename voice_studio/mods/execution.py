"""Bounded subprocess execution of an already captured, explicitly trusted package."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading

from .errors import ModExecutionError
from .schema import json_bytes

MAX_REQUEST = 8 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 900


def _terminate(process):
    """Stop the owned group, falling back to the direct child if the OS denies group signals."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    except OSError:
        # Some app/sandbox hosts permit child termination but deny group signals.
        # Never let an exception escape a background pipe-draining thread.
        try:
            process.kill()
        except OSError:
            pass


def execute(package, capability, payload, settings, *, timeout=DEFAULT_TIMEOUT, output_limit=MAX_OUTPUT):
    """Run one batch using captured package bytes; bound input, combined output and elapsed time.

    The trusted local worker retains the user's filesystem and network permissions.
    Source capture prevents a later package edit from changing an in-flight invocation.
    ModExecutionError reports timeouts, protocol errors and worker exceptions explicitly.
    """
    if type(timeout) not in (int, float) or not 0 < timeout <= MAX_TIMEOUT:
        raise ModExecutionError("Worker timeout must be within (0, 900] seconds")
    if type(output_limit) is not int or not 1 <= output_limit <= MAX_OUTPUT:
        raise ModExecutionError("Worker output limit must be within 1 byte and 8 MiB")
    filename, function = capability["entrypoint"].split(":")
    with tempfile.TemporaryDirectory(prefix="voice-media-mod-") as directory:
        root = Path(directory)
        for name, content in package.assets.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        request_data = {"entrypoint": str(root / filename), "function": function,
                        "payload": payload, "settings": settings, "packageRoot": str(root)}
        for key in ("prepare_input", "convert_output", "rawOutputSchema", "contract"):
            if key in capability:
                request_data[key] = capability[key]
        request_data["capability"] = capability["id"]
        request = json_bytes(request_data)
        if len(request) > MAX_REQUEST:
            raise ModExecutionError("Worker input exceeds 8 MiB")
        worker = Path(__file__).with_name("worker.py")
        try:
            process = subprocess.Popen([sys.executable, "-I", "-B", str(worker)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root,
                                       start_new_session=os.name == "posix")
        except OSError as error:
            raise ModExecutionError(f"Cannot start mod worker: {error}") from error
        output = {"stdout": bytearray(), "stderr": bytearray()}
        total = [0]
        exceeded = threading.Event()
        guard = threading.Lock()

        def drain(pipe, name):
            """Consume each pipe concurrently, killing the worker before unbounded buffering."""
            try:
                while True:
                    chunk = pipe.read(65536)
                    if not chunk:
                        return
                    with guard:
                        total[0] += len(chunk)
                        if total[0] > output_limit:
                            exceeded.set()
                            _terminate(process)
                            return
                        output[name].extend(chunk)
            finally:
                pipe.close()

        def feed():
            """Write bounded stdin concurrently so a worker that never reads still times out."""
            try:
                process.stdin.write(request)
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        readers = [threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
                   threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True)]
        writer = threading.Thread(target=feed, daemon=True)
        for thread in [*readers, writer]:
            thread.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also close descendants retaining pipes after the main worker exits.
            _terminate(process)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as error:
                raise ModExecutionError("OS denied termination of the mod worker") from error
            for thread in [*readers, writer]:
                thread.join(timeout=2)
            if any(thread.is_alive() for thread in [*readers, writer]):
                raise ModExecutionError("Mod worker descendants retained protocol pipes; cleanup was denied")
        if timed_out:
            raise ModExecutionError(f"Mod worker exceeded {timeout:g} seconds")
        if exceeded.is_set():
            raise ModExecutionError("Mod worker exceeded its combined output limit")
        try:
            envelope = json.loads(output["stdout"])
        except (ValueError, UnicodeError) as error:
            raise ModExecutionError("Mod worker returned invalid JSON") from error
        if (not isinstance(envelope, dict) or envelope.get("ok") is not True
                or process.returncode != 0 or "result" not in envelope):
            detail = envelope.get("error", "worker exited without a valid success response") if isinstance(envelope, dict) else "invalid response envelope"
            raise ModExecutionError(f"Mod worker failed: {str(detail)[:2000]}")
        json_bytes(envelope["result"])
        return envelope["result"]
