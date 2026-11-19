"""Exercise a fixed local corpus through a fresh shipped MCP process and record bounded evidence."""

import argparse
import asyncio
from contextlib import suppress
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import which
from statistics import median
import sys
from time import perf_counter

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, fraction):
    """Use the nearest observed rank instead of interpolating a tiny timing sample."""
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(len(values) * fraction + .999) - 1))]


async def memory_samples(samples):
    """Sample the client and its complete MCP/FFmpeg child tree; RSS sums are not physical RAM."""
    if not which("ps"):
        return
    while True:
        process = await asyncio.create_subprocess_exec("ps", "-axo", "pid=,ppid=,rss=",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        stdout, _ = await process.communicate()
        if process.returncode:
            return
        rows = [tuple(map(int, row.split())) for row in stdout.decode().splitlines()]
        owned = {os.getpid()}
        while True:
            added = {pid for pid, parent, _ in rows if parent in owned and pid != process.pid}
            if added <= owned:
                break
            owned.update(added)
        total = sum(rss * 1024 for pid, _, rss in rows if pid in owned)
        samples.append(total)
        await asyncio.sleep(.05)


class Probe:
    """Keep response fingerprints and request timing without retaining transcript text or media blobs."""

    def __init__(self, session):
        self.session = session
        self.calls = []

    async def call(self, name, **arguments):
        """Record one actual JSON-RPC response, including expected errors and attachment sizes."""
        start = perf_counter()
        result = await self.session.call_tool(name, arguments, read_timeout_seconds=timedelta(seconds=120))
        elapsed = (perf_counter() - start) * 1000
        texts = [item.text for item in result.content if item.type == "text"]
        data = None
        if texts:
            with suppress(json.JSONDecodeError):
                data = json.loads(texts[0])
        raw = result.model_dump_json().encode()
        self.calls.append({"tool": name, "arguments": arguments, "elapsedMs": round(elapsed, 3),
                           "responseBytes": len(raw), "error": bool(result.isError),
                           "errorText": texts[0][:500] if result.isError and texts else None,
                           "responseSha256": sha256(raw).hexdigest()})
        return None if result.isError else data


async def inspect_graph(probe, item, repeats):
    """Walk complete speech pages and inspect scene tails, profiles and exact-source media access."""
    key = item["snapshotId"]
    args = {"snapshot_id": key}
    overview = await probe.call("media_graph_overview", **args)
    if overview is None:
        return {"snapshotId": key, "overviewReadable": False}
    duration = overview["durationMs"]
    findings = []
    ids, labels, cursor = [], [], 0
    while True:
        page = await probe.call("media_graph_timeline", **args, start_ms=0,
                                end_ms=min(duration, 300000), kind="utterance", limit=3, after=cursor)
        if page is None:
            findings.append("speech-page-error")
            break
        ids.extend(node["id"] for node in page["nodes"])
        labels.extend(node["label"] for node in page["nodes"])
        next_cursor = page.get("nextAfter")
        if next_cursor is None:
            break
        if next_cursor == cursor or len(ids) > 10000:
            findings.append("pagination-not-progressing")
            break
        cursor = next_cursor
    if len(ids) != len(set(ids)):
        findings.append("duplicate-speech-pages")
    if duration <= 300000 and len(ids) != overview["nodesByKind"].get("utterance", 0):
        findings.append("speech-count-mismatch")
    if ids:
        detail = await probe.call("media_graph_node", **args, node_id=ids[0], limit=3)
        if detail is None:
            findings.append("node-unreadable")
        elif detail.get("snapshotId") != key:
            findings.append("node-snapshot-not-identified")
        phrase = " ".join(labels[0].split()[:6])
        found = await probe.call("media_graph_search", **args, query=phrase, kind="utterance")
        if found is None or ids[0] not in [row["id"] for row in found["nodes"]]:
            findings.append("source-text-not-searchable")
    for _ in range(repeats):
        await probe.call("media_graph_timeline", **args, start_ms=max(0, duration - 5000), end_ms=duration, limit=5)
    at = min(duration - 1, 19000)
    if overview["kind"] == "video":
        frame = await probe.call("media_graph_frame", **args, at_ms=at, max_side=320, technical=True)
        if frame is None:
            findings.append("source-frame-unavailable")
    audio = await probe.call("media_graph_audio", **args, at_ms=0, duration_ms=min(500, duration))
    if audio is None:
        findings.append("source-audio-unavailable")
    return {"snapshotId": key, "title": overview["title"], "durationMs": duration,
            "sourceStatus": overview.get("source", {}).get("status"), "nodesByKind": overview["nodesByKind"],
            "dataHealth": overview.get("dataHealth"),
            "speechCount": len(ids), "speechIdsSha256": sha256(json.dumps(ids).encode()).hexdigest(),
            "speechTextSha256": sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest(), "findings": findings}


async def run(options):
    """Own one fresh MCP session and capture steady request timings separately from startup."""
    corpus = json.loads(options.corpus.read_text())
    samples = []
    monitor = asyncio.create_task(memory_samples(samples))
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "ag_media_studio_mcp.py"),
        "--graph-root", str(options.graph_root), "--studio-data", str(options.studio_data)], cwd=ROOT)
    start = perf_counter()
    try:
        async with stdio_client(params) as streams, ClientSession(*streams) as session:
            await session.initialize()
            startup = (perf_counter() - start) * 1000
            probe = Probe(session)
            catalog = await probe.call("media_catalog", category="graph")
            reports = []
            for item in corpus:
                reports.append(await inspect_graph(probe, item, options.repeats))
            latencies = [row["elapsedMs"] for row in probe.calls
                         if row["tool"] == "media_graph_timeline" and not row["error"]]
            timeline_latency = {"samples": len(latencies), "p50": None, "p95": None}
            if latencies:
                timeline_latency.update(p50=median(latencies), p95=percentile(latencies, .95))
            return {"label": options.label, "startupMs": round(startup, 2), "reports": reports, "calls": probe.calls,
                    "catalogCategories": list(catalog.get("categories", {})) if catalog else [],
                    "runtime": catalog.get("runtime") if catalog else None,
                    "timelineLatencyMs": timeline_latency,
                    "peakTreeRssBytes": max(samples) if samples else None,
                    "rssScope": "50ms ps sampled sum client+MCP+children RSS; not unique physical RAM; null if unavailable",
                    "responseBytes": sum(row["responseBytes"] for row in probe.calls)}
    finally:
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=ROOT / "data/studio/media-graphs")
    parser.add_argument("--studio-data", type=Path, default=ROOT / "data/studio",
                        help="Workspace for the fresh MCP process. Use an isolated workspace for tests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="current")
    parser.add_argument("--repeats", type=int, default=5)
    options = parser.parse_args()
    if not 1 <= options.repeats <= 30:
        parser.error("repeats must be between 1 and 30")
    report = asyncio.run(run(options))
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(options.output), "latency": report["timelineLatencyMs"],
                      "findings": [{"title": row.get("title"), "snapshotId": row["snapshotId"],
                                    "findings": row.get("findings", ["overview-unreadable"])}
                                   for row in report["reports"]]}, ensure_ascii=False))
