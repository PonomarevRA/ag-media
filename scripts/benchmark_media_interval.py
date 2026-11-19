"""Benchmark compact and full interval reads on a generated two-hour media graph."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import chain
from pathlib import Path
from statistics import median, quantiles
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_studio.graph import media_graph_store
from voice_studio.graph.media_graph_interval import SECTIONS, summarize_interval

DURATION_MS = 7_200_000
WINDOWS = {
    "early": (1_030, 16_420),
    "middle": (3_600_025, 3_615_028),
    "late": (DURATION_MS - 60_001, DURATION_MS - 1),
    "tail": (DURATION_MS - 15_000, DURATION_MS),
    "empty": (0, 100),
}


def fixture():
    """Generate a stable mix of timed facts and references without footage or model execution."""
    nodes, edges = [], []
    step_by_kind = {"shot": 9500, "camera_motion": 1234, "utterance": 1130,
                    "text_observation": 340, "screen_region": 650,
                    "ui_element": 175, "object_observation": 500}

    def add(identity, kind, start, end, **facts):
        """Append one generated graph row with its type-specific evidence."""
        nodes.append({"id": identity, "kind": kind, "startMs": start,
                      "endMs": end, "label": f"Synthetic {identity}", "data": facts})

    add("speaker", "speaker", None, None)
    for index in range(14):
        add(f"profile-{index}", "object_profile", None, None)
    for index, kind in enumerate(chain.from_iterable(SECTIONS.values())):
        step = step_by_kind.get(kind, 1700 + index * 170)
        for number, start in enumerate(range(index * 25 + 850, DURATION_MS, step)):
            identity = f"{kind}-{number}"
            facts = {"score": (number % 10) / 10, "role": "input-candidate" if number % 8 == 0 else "input"}
            if kind == "utterance":
                facts.update(overlapUnresolved=number % 7 == 0,
                             voiceProfileEvidence={"state": "abstained" if number % 9 == 0 else "assigned"})
                if number % 5:
                    edges.append((identity, "spoken_by", "speaker"))
            if kind == "object_observation":
                for profile in range(14 if number % 23 == 0 else 1):
                    edges.append((identity, "sample_of", f"profile-{profile}"))
            length = step if kind == "shot" else step // 2
            add(identity, kind, start, min(DURATION_MS, start + length), **facts)
    add("long-review", "dialogue_issue", 0, DURATION_MS, notConfirmedError=True)
    return {"nodes": nodes, "edges": edges}


def fingerprint(response):
    """Ignore timing and temporary database/source identities when comparing serialized facts."""
    evidence = {key: value for key, value in response.items() if key not in ("timingMs", "snapshotId")}
    evidence["source"] = {key: value for key, value in evidence["source"].items() if key != "path"}
    return sha256(media_graph_store.encode(evidence).encode()).hexdigest()


def interval_workload(store, key, repeats):
    """Measure repeated actual interval reads through the same immutable snapshot."""
    measurements = {}
    for label, (start, end) in WINDOWS.items():
        for detail in ("compact", "full"):
            call = summarize_interval(store, key, start, end, detail)
            expected = fingerprint(call)
            timings = []
            for _ in range(repeats):
                started = perf_counter()
                response = summarize_interval(store, key, start, end, detail)
                timings.append((perf_counter() - started) * 1000)
                if fingerprint(response) != expected:
                    raise RuntimeError(f"Interval evidence changed for {label}/{detail}.")
            measurements[f"{label}/{detail}"] = {
                "medianMs": round(median(timings), 3),
                "p95Ms": round(quantiles(timings, n=20, method="inclusive")[18], 3),
                "evidenceSha256": expected,
            }
    return measurements


def run(repeats, legacy_index):
    """Build a private temporary database and report creation, size and read measurements."""
    with TemporaryDirectory(prefix="ag-media-interval-benchmark-") as folder:
        root = Path(folder)
        source = root / "synthetic-input.bin"
        source.write_bytes(b"Generated interval benchmark fixture.")
        store = media_graph_store.MediaGraphStore(root / "graphs")
        graph = fixture()
        schema = media_graph_store.SCHEMA
        if legacy_index:
            current_index = "CREATE INDEX nodes_by_kind ON nodes(kind, start_ms, seq, end_ms);"
            previous_index = "CREATE INDEX nodes_by_kind ON nodes(kind, start_ms);"
            if current_index not in schema:
                raise RuntimeError("The current index changed. Update the benchmark setup.")
            schema = schema.replace(current_index, previous_index)
        source_identity = media_graph_store.identify(source)
        started = perf_counter()
        with patch.object(media_graph_store, "SCHEMA", schema):
            manifest = store.publish(source_identity, {"kind": "video", "durationMs": DURATION_MS},
                                     graph, "en", {})
        key = manifest["snapshotId"]
        build_ms = (perf_counter() - started) * 1000
        return {"durationMs": DURATION_MS, "nodeCount": len(graph["nodes"]), "edgeCount": len(graph["edges"]),
                "legacyIndex": legacy_index, "buildMs": round(build_ms, 3),
                "databaseBytes": (store.root / f"{key}.sqlite").stat().st_size, "readsPerCase": repeats,
                "intervals": interval_workload(store, key, repeats)}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--legacy-index", action="store_true", help="Use the previous index shape in the test database.")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2.")
    print(media_graph_store.encode(run(args.repeats, args.legacy_index)))
