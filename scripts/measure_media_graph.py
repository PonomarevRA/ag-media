"""Compare inlined, normalized and queryable media graphs on a fixed two-hour workload."""

from argparse import ArgumentParser
import gzip
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, encode, identify

DURATION_MS = 7_200_000


def fixture():
    """Create deterministic timed speech, shot, object, OCR and sparse face observations."""
    cues, scene, cuts = [], [], []
    frames = [()] * (DURATION_MS // 40)
    coverage = {"speech": [{"startMs": 0, "endMs": DURATION_MS, "status": "ok"}],
                "shots": [{"startMs": 0, "endMs": DURATION_MS, "status": "ok"}],
                "face": [], "scene": [{"startMs": 0, "endMs": DURATION_MS, "status": "sampled"}]}
    for stamp in range(0, DURATION_MS, 1500):
        cues.append({"startMs": stamp, "endMs": stamp + 900,
                     "speaker": stamp // 1500 % 8 + 1, "text": f"Проверочная реплика {stamp // 1500}"})
    for stamp in range(0, DURATION_MS, 10000):
        cuts.append({"startMs": stamp})
    for stamp in range(0, DURATION_MS, 1000):
        scene.append({"timeMs": stamp, "objects": [{"label": "laptop", "box": [.15, .5, .28, .35], "score": .82}],
                      "text": [{"text": "DemoSign", "box": [.2, .3, .4, .1]}]})
    for start in range(0, DURATION_MS, 600000):
        coverage["face"].append({"startMs": start, "endMs": start + 60000, "status": "sampled"})
        for stamp in range(start, start + 60000, 40):
            x = (10 + stamp // 40 % 21) / 100
            frames[stamp // 40] = [[1, x, .2, .3, .4, 0, 1]]
    return {"language": "ru", "segments": cues, "coverage": coverage,
            "videoEvidence": {"editorTimeline": {"shots": cuts, "durationMs": DURATION_MS},
                              "timeline": {"stepMs": 40, "frames": frames}},
            "sceneEvidence": {"frames": scene}}


def inlined_edges(graph):
    """Model one alternative that stores the complete target node beside each relation."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    return {**graph, "edges": [[source, relation, nodes[target]] for source, relation, target in graph["edges"]]}


def encoded_sizes(graph):
    """Report raw and gzip size without assuming that archive compression helps random reads."""
    raw = encode(graph).encode()
    return {"rawBytes": len(raw), "gzipBytes": len(gzip.compress(raw, mtime=0))}


def run():
    """Compare fixed-size alternatives and verify searches near the end of the timeline."""
    graph = build_graph(fixture(), kind="video", duration_ms=DURATION_MS)
    result = {"inlinedJson": encoded_sizes(inlined_edges(graph)), "referenceJson": encoded_sizes(graph)}
    with TemporaryDirectory() as folder:
        work = Path(folder)
        source = work / "synthetic-media"
        source.write_bytes(b"synthetic fixture")
        store = MediaGraphStore(work / "graphs")
        started = perf_counter()
        summary = store.publish(identify(source), {"kind": "video", "durationMs": DURATION_MS}, graph, "ru", {})
        key = summary["snapshotId"]
        indexed = {"bytes": summary["storageBytes"], "buildMs": round((perf_counter() - started) * 1000)}
        times = []
        for index in range(30):
            begin = DURATION_MS - 40000 + index * 200
            tick = perf_counter()
            window = store.timeline(key, begin, begin + 30000, "utterance")
            phrase = store.search(key, f"Проверочная реплика {4799-index}", "utterance")
            assert window["nodes"] and phrase["nodes"]
            times.append((perf_counter() - tick) * 1000)
        indexed["timelinePlusSearchMedianMs"] = round(median(times), 2)
    return {**result, "sqlite": indexed, "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
            "geometrySamples": len(graph["faceFrames"]), "durationMs": DURATION_MS}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = encode(run())
    if options.output:
        options.output.write_text(report + "\n", encoding="utf-8")
    print(report)
