#!/usr/bin/env python3
"""Compare a frozen legacy CLI with the public one-shot and resident APIs.

Run with the application's Python and an existing manifest; no model, package or
media downloads occur. Biometric vectors remain in memory, while artifacts keep
only shape, hashes, numerical differences and matching decisions. RSS is sampled
from the measured host and its descendants using macOS ps (KiB, not model size).
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import select
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("legacy", "oneshot", "resident")


def digest(path):
    """Hash complete input bytes without keeping a model or recording in RAM."""
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_baseline(path):
    """Import the preserved kernel for downstream comparisons, never rewrite it."""
    spec = importlib.util.spec_from_file_location("speaker_benchmark_legacy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def host(args):
    """Serve sequential measured API calls in one otherwise ordinary app host."""
    from voice_studio.mod_runtime import worker_environment
    from voice_studio.speech import speaker_review

    if args.host == "default":
        os.environ.pop("VOICE_SPEAKER_EMBEDDING_MODE", None)
    else:
        os.environ["VOICE_SPEAKER_EMBEDDING_MODE"] = args.host
    print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            if args.host in ("resident", "default"):
                from voice_studio.speech.embedding_worker import get_worker
                get_worker().close()
            print(json.dumps({"closed": True}), flush=True)
            return
        started = time.perf_counter()
        if args.host == "legacy":
            # This is the original separate-interpreter transport and preserved
            # original native kernel; only the launcher path names its copy.
            proc = subprocess.run(
                [sys.executable, str(args.baseline), request["wav"]],
                input=json.dumps(request["spans"]), env=worker_environment(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=35, check=True,
            )
            vectors = json.loads(proc.stdout)
        else:
            vectors = speaker_review.extract_embeddings(Path(request["wav"]), request["spans"])
        elapsed = time.perf_counter() - started
        stats = None
        if args.host in ("resident", "default"):
            from voice_studio.speech.embedding_worker import get_worker
            stats = get_worker().stats()
        print(json.dumps({"elapsedSeconds": elapsed, "vectors": vectors,
                          "worker": stats}, allow_nan=False), flush=True)


def process_table():
    """Read numeric process metadata only; never collect commands or environments."""
    proc = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,rss=,pcpu="],
                          capture_output=True, text=True, check=True)
    rows = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) == 4:
            pid, parent, rss, cpu = fields
            rows[int(pid)] = (int(parent), int(rss), float(cpu))
    return rows


def tree_sample(pid, table):
    """Sum the measured host and all descendants, preserving actual PID ownership."""
    owned = {pid}
    pending = True
    while pending:
        pending = False
        for child, (parent, _, _) in table.items():
            if parent in owned and child not in owned:
                owned.add(child)
                pending = True
    live = owned.intersection(table)
    host_rss = table.get(pid, (0, 0, 0))[1]
    total = sum(table[child][1] for child in live)
    return {"at": time.monotonic(), "hostRssKiB": host_rss,
            "childrenRssKiB": total - host_rss, "treeRssKiB": total,
            "pids": sorted(live), "treeCpuPercent": sum(table[child][2] for child in live)}


class Host:
    """Own one isolated benchmark host, its RPC deadline and memory sampler."""

    def __init__(self, mode, baseline, snapshot, idle_seconds=60):
        """Freeze the same provider snapshot and model settings for every host."""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        environment["VOICE_MOD_SNAPSHOT"] = snapshot
        environment["VOICE_SPEAKER_EMBEDDING_IDLE_SECONDS"] = str(idle_seconds)
        self.mode = mode
        self.samples = []
        self.child_pids = set()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--host", mode,
             "--baseline", str(baseline)], cwd=ROOT, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        try:
            self.read()
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)
            raise
        self.startup_seconds = time.perf_counter() - self.started
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()

    def sample(self):
        """Sample each 50ms; retain sampler failures instead of claiming RSS proof."""
        self.sampler_error = None
        while not self.stop.is_set():
            try:
                sample = tree_sample(self.proc.pid, process_table())
                with self.lock:
                    self.samples.append(sample)
                    self.child_pids.update(set(sample["pids"]) - {self.proc.pid})
            except Exception as error:
                self.sampler_error = repr(error)
                return
            self.stop.wait(.05)

    def read(self):
        """Read one protocol response with a deadline, surfacing native errors."""
        if not select.select([self.proc.stdout], [], [], 45)[0]:
            raise TimeoutError("Benchmark host did not respond within 45 seconds")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(self.proc.stderr.read()[-4000:])
        return json.loads(line)

    def request(self, payload):
        """Measure a complete public API call and its sampled process-tree peak."""
        before = time.monotonic()
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        response = self.read()
        after = time.monotonic()
        with self.lock:
            samples = [sample for sample in self.samples if before <= sample["at"] <= after]
        response["roundTripSeconds"] = after - before
        response["rssSamples"] = len(samples)
        for key in ("hostRssKiB", "childrenRssKiB", "treeRssKiB", "treeCpuPercent"):
            response["peak" + key[0].upper() + key[1:]] = max(
                (sample[key] for sample in samples), default=None)
        return response

    def memory(self):
        """Take a point-in-time host/child RSS observation outside inference."""
        return tree_sample(self.proc.pid, process_table())

    def close(self):
        """Close model transport then host, and verify every observed child exited."""
        try:
            if self.proc.poll() is None:
                self.request({"close": True})
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.stop.set()
            self.sampler.join(timeout=2)
        table = process_table()
        return {"hostReturnCode": self.proc.returncode,
                "observedChildren": sorted(self.child_pids),
                "survivingObservedChildren": sorted(self.child_pids.intersection(table)),
                "samplerError": self.sampler_error}


def vector_parity(expected, actual):
    """Compare finite vectors exactly and by cosine without retaining biometrics."""
    shapes_equal = expected.keys() == actual.keys()
    worst_difference, lowest_cosine, count = 0.0, 1.0, 0
    finite = True
    for key, rows in expected.items():
        candidates = actual.get(key, [])
        if len(rows) != len(candidates):
            shapes_equal = False
            continue
        for left, right in zip(rows, candidates):
            if len(left) != len(right):
                shapes_equal = False
                continue
            finite = finite and all(math.isfinite(value) for value in left + right)
            difference = max((abs(a - b) for a, b in zip(left, right)), default=0)
            norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
            similarity = sum(a * b for a, b in zip(left, right)) / norm if norm else 0
            worst_difference = max(worst_difference, difference)
            lowest_cosine = min(lowest_cosine, similarity)
            count += 1
    dimensions = sorted({len(vector) for rows in actual.values() for vector in rows})
    return {"shapesEqual": shapes_equal, "finite": finite, "vectorsCompared": count,
            "dimensions": dimensions, "maxAbsoluteDifference": worst_difference,
            "minimumCosine": lowest_cosine, "exact": expected == actual}


def matching_decisions(module, vectors):
    """Exercise unchanged production matching thresholds with explicit visual cases."""
    results = {}
    for scenario in ("voice-only", "shared-face", "distinct-faces", "overlap-exclusion"):
        profiles = {}
        for key, samples in vectors.items():
            number = int(key)
            face = None if scenario in ("voice-only", "overlap-exclusion") else 1
            if scenario == "distinct-faces":
                face = number
            profiles[number] = module.VoiceProfile(number, face, number * 1000, samples, False)
        overlaps = [{"speakers": list(profiles)}] if scenario == "overlap-exclusion" else []
        results[scenario] = module.validate_pairs(profiles, overlaps)
    # These are transport-parity controls, not labelled speaker-accuracy claims.
    # Reuse a real inferred reference to guarantee a positive pair, then apply
    # independent visual and concurrent-speech exclusions to that same pair.
    reference = next((row for rows in vectors.values() for row in rows), None)
    if reference is not None:
        for scenario in ("identical-reference-positive", "identical-reference-different-face", "identical-reference-overlap"):
            profiles = {
                1: module.VoiceProfile(1, 1, 1000, [reference, reference], False),
                2: module.VoiceProfile(2, 2 if scenario.endswith("different-face") else 1,
                                       2000, [reference, reference], False),
            }
            overlaps = [{"speakers": [1, 2]}] if scenario.endswith("overlap") else []
            results[scenario] = module.validate_pairs(profiles, overlaps)
    return results


def summarize(records):
    """Report count, median and min/max; five observations cannot support tail claims."""
    summary = {}
    for mode in MODES:
        workloads = sorted({row["workload"] for row in records})
        for workload in workloads:
            rows = [row for row in records if row["mode"] == mode and row["workload"] == workload]
            for phase in ("cold", "warm"):
                selected = [row for row in rows if row["phase"] == phase]
                if not selected:
                    continue
                values = [row["elapsedSeconds"] for row in selected]
                peaks = [row["peakTreeRssKiB"] for row in selected if row["peakTreeRssKiB"] is not None]
                summary[f"{mode}/{workload}/{phase}"] = {
                    "count": len(values), "medianSeconds": statistics.median(values),
                    "minSeconds": min(values), "maxSeconds": max(values),
                    "sampledPeakTreeRssKiB": max(peaks) if peaks else None,
                }
    return summary


def validate_evidence(results):
    """Fail the command on recorded errors, leaked children or idle parity loss."""
    problems = list(results["failures"])
    for mode, cleanup in results["cleanup"].items():
        if cleanup["hostReturnCode"] or cleanup["survivingObservedChildren"] or cleanup["samplerError"]:
            problems.append({"mode": mode, "reason": "Process cleanup or memory sampler failed"})
    idle_parity = results.get("idleRelease", {}).get("restartParity", {})
    if not idle_parity.get("exact") or not idle_parity.get("finite"):
        problems.append({"reason": "Idle restart vector parity failed or missing"})
    if problems:
        raise RuntimeError(json.dumps(problems))


def run(args):
    """Round-robin native workloads across isolated hosts and emit auditable evidence."""
    manifest = json.loads(args.manifest.read_text())
    if digest(args.baseline) != manifest["baselineSha256"]:
        raise ValueError("Frozen original source hash mismatch")
    for item in manifest["inputs"] + [manifest["model"]]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Fixed input/model bytes changed: " + item["path"])
    snapshot = args.snapshot.read_text()
    module = load_baseline(args.baseline)
    hosts = {}
    records, baselines = [], {}
    results = {"manifest": manifest, "platform": platform.platform(), "python": sys.version,
               "snapshotSha256": hashlib.sha256(snapshot.encode()).hexdigest(),
               "rssMethod": "ps numeric process tree sampled each50ms; measured host+descendants; supervisor excluded; sampled peaks are lower bounds",
               "failures": [], "records": records, "cleanup": {}}
    try:
        for mode in MODES:
            hosts[mode] = Host(mode, args.baseline, snapshot)
        # Cold first request has no cached model. Subsequent full batches exercise
        # retained model state; order rotates each round to limit ordering bias.
        for round_number in range(args.repeats + 1):
            for workload in manifest["workloads"]:
                for offset in range(len(MODES)):
                    mode = MODES[(round_number + offset) % len(MODES)]
                    response = hosts[mode].request(workload["request"])
                    vectors = response.pop("vectors")
                    key = workload["name"]
                    if mode == "legacy":
                        baselines.setdefault(key, vectors)
                    if key not in baselines:
                        raise RuntimeError("Legacy must establish each fixed workload first")
                    response.update(mode=mode, workload=key, round=round_number,
                                    phase="cold" if round_number == 0 and key == manifest["workloads"][0]["name"] else "warm")
                    response["parity"] = vector_parity(baselines[key], vectors)
                    expected_decisions = matching_decisions(module, baselines[key])
                    current_decisions = matching_decisions(module, vectors)
                    response["decisionsEqual"] = expected_decisions == current_decisions
                    response["decisionCounts"] = {name: len(rows) for name, rows in current_decisions.items()}
                    expected_count = sum(len(windows) for windows in workload["request"]["spans"].values())
                    response["expectedVectorCount"] = expected_count
                    complete = response["parity"]["vectorsCompared"] == expected_count
                    if not response["parity"]["exact"] or not response["parity"]["finite"] or not response["decisionsEqual"] or not complete:
                        results["failures"].append({"mode": mode, "workload": key, "round": round_number, "reason": "Numerical or decision parity mismatch"})
                    records.append(response)
                    print(json.dumps({"mode": mode, "workload": key, "round": round_number,
                                      "seconds": response["elapsedSeconds"], "exact": response["parity"]["exact"]}), flush=True)
        results["afterInferenceMemory"] = {mode: process.memory() for mode, process in hosts.items()}
    except Exception as error:
        results["failures"].append({"reason": repr(error)})
        raise
    finally:
        for mode, process in hosts.items():
            results["cleanup"][mode] = process.close()
        results["startupSeconds"] = {mode: process.startup_seconds for mode, process in hosts.items()}
        results["summary"] = summarize(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    # Use an explicit short test timeout in a fresh host. Main performance runs
    # retain the production 60s idle setting; this phase verifies the lifecycle
    # mechanism without imposing a minute of artificial inactivity.
    idle_host = Host("resident", args.baseline, snapshot, idle_seconds=2)
    try:
        first = idle_host.request(manifest["workloads"][0]["request"])
        before = idle_host.memory()
        time.sleep(3)
        after = idle_host.memory()
        second = idle_host.request(manifest["workloads"][0]["request"])
        results["idleRelease"] = {
            "configuredIdleSeconds": 2, "observedAfterSeconds": 3,
            "firstWorker": first["worker"], "secondWorker": second["worker"],
            "memoryBeforeIdle": before, "memoryAfterIdle": after,
            "restartParity": vector_parity(first["vectors"], second["vectors"]),
        }
        if len(after["pids"]) != 1 or first["worker"]["pid"] == second["worker"]["pid"]:
            results["failures"].append({"reason": "Idle child release or fresh PID restart failed"})
    finally:
        results["cleanup"]["idle-check"] = idle_host.close()
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    validate_evidence(results)


def main():
    """Route the benchmark supervisor or its private isolated API host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=(*MODES, "default"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.host:
        host(args)
    elif args.manifest and args.snapshot and args.output and args.repeats >= 1:
        run(args)
    else:
        parser.error("manifest, snapshot, output and positive repeats are required")


if __name__ == "__main__":
    main()
