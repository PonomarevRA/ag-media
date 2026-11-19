"""Exercise an explicitly enabled loopback live endpoint with fixed, standalone WebM chunks."""

import argparse
from collections import Counter
import ctypes
from hashlib import file_digest
import http.client
import json
from math import ceil
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from threading import Event, Thread
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.parse import urlencode
from uuid import uuid4


class ServerUsage:
    """Sample resident memory and cumulative CPU for an isolated macOS server tree."""

    def __init__(self, pid):
        """Prepare libproc and a sampler without running helper commands in the loop."""
        self.pid = pid
        self.lib = ctypes.CDLL("libproc.dylib")
        self.lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        timebase = (ctypes.c_uint32 * 2)()
        if ctypes.CDLL(None).mach_timebase_info(timebase) != 0 or not timebase[1]:
            raise RuntimeError("Cannot convert macOS process CPU counters")
        self.cpu_numerator, self.cpu_denominator = timebase
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)
        self.peak = 0
        self.last_cpu = {}

    def read(self):
        """Read own CPU and RSS from the root and its active descendants."""
        todo = [self.pid]
        rss = 0
        cpu = {}
        while todo:
            pid = todo.pop()
            children = (ctypes.c_int * 64)()
            count = self.lib.proc_listchildpids(pid, children, ctypes.sizeof(children))
            if count > 0:
                todo.extend(child for child in children[:min(count, len(children))] if child > 0)
            usage = (ctypes.c_uint64 * 20)()
            if self.lib.proc_pid_rusage(pid, 2, usage) == 0:
                # rusage_info_v2 stores CPU in Mach ticks and resident size in bytes.
                # PID plus process start distinguishes any reused identifiers.
                rss += usage[8]
                cpu[(pid, usage[10])] = usage[2] + usage[3]
        return rss, cpu

    def sample(self):
        """Poll aggregate RSS without creating another process per sample."""
        while not self.stop.wait(.025):
            rss, cpu = self.read()
            self.peak = max(self.peak, rss)
            self.last_cpu.update(cpu)

    def __enter__(self):
        """Establish the CPU baseline for processes already running."""
        self.idle, self.start = self.read()
        if not self.idle:
            raise RuntimeError("Cannot read server resource usage")
        self.peak = self.idle
        self.last_cpu.update(self.start)
        self.thread.start()
        return self

    def __exit__(self, *_):
        """Keep the last live counters for any children that have already exited."""
        self.stop.set()
        self.thread.join()
        self.end_rss, self.end_cpu = self.read()
        self.peak = max(self.peak, self.end_rss)
        self.last_cpu.update(self.end_cpu)

    def report(self):
        """Return sampled peaks and a CPU lower bound after subtracting the pre-run counters."""
        ticks = sum(last - self.start.get(process, 0) for process, last in self.last_cpu.items())
        cpu_ns = ticks * self.cpu_numerator / self.cpu_denominator
        return {"sampleIntervalMs": 25, "idleRssMiB": round(self.idle / 2**20, 1),
                "peakRssMiB": round(self.peak / 2**20, 1), "endRssMiB": round(self.end_rss / 2**20, 1),
                "sampledCpuLowerBoundMs": round(cpu_ns / 1_000_000)}


def duration_ms(source):
    """Read the fixture length before selecting exact fixed clock intervals."""
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(source)], timeout=15,
    )
    return round(float(raw) * 1000)


def prepare(source, folder, chunk_ms, height, fps):
    """Encode replay fixtures before measuring network or server turnaround."""
    total = duration_ms(source)
    chunks = []
    for start in range(0, total, chunk_ms):
        length = min(chunk_ms, total - start)
        if length < 500:
            break
        path = folder / f"{start}.webm"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start / 1000:.3f}",
             "-i", str(source), "-t", f"{length / 1000:.3f}",
             "-vf", f"scale=-2:{height},fps={fps}",
             "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "6", "-b:v", "400k",
             "-c:a", "libopus", "-ac", "1", "-b:a", "36k", str(path)],
            check=True, timeout=30,
        )
        chunks.append((start, length, path))
    return chunks


def request(port, session, token, language, start, body):
    """Send one chunk while keeping the token out of output and process arguments."""
    query = urlencode({"session": session, "lang": language, "start_ms": start})
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=45)
    clock = monotonic()
    try:
        connection.request("POST", "/api/live-chunk?" + query, body, {
            "Content-Type": "video/webm", "X-Voice-Live-Token": token,
        })
        response = connection.getresponse()
        payload = json.loads(response.read())
        elapsed = round((monotonic() - clock) * 1000)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {payload.get('error')}")
        return payload, elapsed
    finally:
        connection.close()


def latency(samples):
    """Return the median and the nearest-rank 95th percentile in milliseconds."""
    ordered = sorted(samples)
    return {"p50": median(ordered), "p95": ordered[ceil(.95 * len(ordered)) - 1]}


def analyze(port, chunks, token, language, paced=False):
    """Replay one session in time order, recording only metadata and recognized text."""
    session = uuid4().hex
    rows = []
    clock = monotonic()
    for start, expected_duration, path in chunks:
        if paced:
            sleep(max(0, clock + (start + expected_duration) / 1000 - monotonic()))
        result, elapsed = request(port, session, token, language, start, path.read_bytes())
        expected_fields = {"durationMs", "segments", "faceFrames", "warnings", "timingsMs"}
        expected_stages = {"decode", "faces", "speech", "total"}
        if not expected_fields <= result.keys() or not expected_stages <= result["timingsMs"].keys():
            raise ValueError(f"Unexpected response contract for chunk {start}")
        duration = result["durationMs"]
        caption_rows = result["segments"]
        face_rows = result["faceFrames"]
        if not 0 < duration <= expected_duration + 500:
            raise ValueError(f"Unexpected duration for chunk {start}: {duration}")
        if any(not start <= row["startMs"] < row["endMs"] <= start + duration + 200 for row in caption_rows):
            raise ValueError(f"Caption offsets outside request window {start}")
        if any(not start <= row["timeMs"] <= start + duration + 200 for row in face_rows):
            raise ValueError(f"Face offsets outside request window {start}")
        seen = Counter(face["id"] for row in face_rows for face in row["faces"])
        rows.append({
            "startMs": start, "durationMs": duration, "responseMs": elapsed,
            "rtf": round(elapsed / duration, 3), "text": " ".join(row["text"] for row in caption_rows),
            "faceIds": sorted(seen.items()), "faceSamples": len(face_rows),
            "facesAtMs": [[row["timeMs"], [face["id"] for face in row["faces"]]] for row in face_rows],
            "segmentTimesMs": [[row["startMs"], row["endMs"]] for row in caption_rows],
            "warnings": result.get("warnings", []), "stagesMs": result["timingsMs"],
        })
    return rows


def main():
    """Prepare the local fixture and benchmark the live request/response path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--language", choices=["ru", "en"], default="ru")
    parser.add_argument("--chunk-ms", type=int, default=3000)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--require-later-speech", action="store_true",
                        help="fail if the spoken fixture has no recognized text after its first chunk")
    args = parser.parse_args()
    token = os.environ.get("VOICE_LIVE_TOKEN")
    if not token:
        parser.error("set VOICE_LIVE_TOKEN in the environment")
    with args.source.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    with TemporaryDirectory(prefix="voice-live-chunks-") as directory:
        chunks = prepare(args.source, Path(directory), args.chunk_ms, args.height, args.fps)
        usage = None
        if args.server_pid:
            if sys.platform != "darwin":
                parser.error("--server-pid is supported on macOS")
            with ServerUsage(args.server_pid) as usage:
                rows = analyze(args.port, chunks, token, args.language, args.paced)
        else:
            rows = analyze(args.port, chunks, token, args.language, args.paced)
    if not rows:
        raise ValueError("No nonempty test chunks")
    output = {
        "scope": "loopback HTTP turnaround for pre-encoded independent WebM windows; capture/assembly excluded",
        "sourceSha256": digest, "configuredChunkMs": args.chunk_ms, "height": args.height, "fps": args.fps,
        "firstResponseMs": rows[0]["responseMs"], "responseMs": latency(row["responseMs"] for row in rows),
        "stageMs": {stage: latency(row["stagesMs"][stage] for row in rows) for stage in rows[0]["stagesMs"]},
        "durationMs": sum(row["durationMs"] for row in rows),
        "cumulativeResponseMs": sum(row["responseMs"] for row in rows),
        "paced": args.paced,
        "serverTreeUsage": usage.report() if usage else None,
        "rows": rows,
    }
    report = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")
    if args.require_later_speech and not any(row["text"].strip() for row in rows[1:]):
        raise RuntimeError("No delayed speech in the spoken fixture")


if __name__ == "__main__":
    main()
