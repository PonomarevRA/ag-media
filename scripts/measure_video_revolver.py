"""Compare local full video jobs with one fixed Light-ASD worker per variant."""

import argparse
from hashlib import file_digest, sha256
import json
import os
import signal
from pathlib import Path
import resource
import sqlite3
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice_studio.vision.activity_worker_session import ActivityWorkerSession

BUILTIN_WORKER = ROOT / "voice_studio/workers/light_asd_worker.py"


def millis(seconds):
    """Round elapsed and CPU seconds to the nearest millisecond."""
    return round(1000 * seconds)


def max_rss_mib(usage):
    """Normalize the OS-specific peak RSS of one measured process."""
    return round(usage.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10), 1)


class MeasuredActivitySession(ActivityWorkerSession):
    """Measure the exact warm worker process and its completed window requests."""

    def __init__(self, command, imports):
        """Initialize per-run counters for this worker and any restarted child."""
        super().__init__(command, imports)
        self.wall = self.cpu = self.peak_rss = 0
        self.stages = {}

    def request(self, arguments, timeout):
        """Attribute window work, including startup and IPC, to the activity worker."""
        start = monotonic()
        try:
            report = super().request(arguments, timeout)
        finally:
            self.wall += monotonic() - start
        for stage, seconds in report.get("seconds", {}).items():
            self.stages[stage] = self.stages.get(stage, 0) + seconds
        return report

    def close_child(self):
        """Collect usage when retiring the owned process, then close its pipes."""
        try:
            child = self.process
            if child is not None and child.returncode is None:
                # Popen.kill first polls and could reap an exited PID before wait4.
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                self.cpu += usage.ru_utime + usage.ru_stime
                self.peak_rss = max(self.peak_rss, max_rss_mib(usage))
        finally:
            super().close_child()

    def measurements(self):
        """Report worker-only resources separately from the full video job."""
        return {"wallMs": millis(self.wall), "cpuMs": millis(self.cpu),
                "peakRssMiB": self.peak_rss, "stages": self.stages}


def read_context(db_path, digest, language, report):
    """Apply the saved exact-source context in a read-only connection when requested."""
    from voice_studio.app import transcript_editor

    def connect():
        db = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    editor = object.__new__(transcript_editor.TranscriptEditor)
    editor.connect = connect
    editor.recover(language, digest, report)


def semantic_report(report):
    """Fingerprint speech, face IDs, geometry, activity, and cuts separately from runtime."""
    video = report["videoEvidence"]
    cues = report["segments"]
    test_cue = next((cue for cue in cues if cue["text"] == "Ну, это правда."), None)

    def digest(value):
        return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode()).hexdigest()

    return {
        "cueCount": len(cues),
        "targetCue": {key: test_cue.get(key) for key in (
            "startMs", "endMs", "speaker", "speakerNumber", "visualFace", "linkedSpeakerNumber",
        )} if test_cue else None,
        "srtSha256": digest(report["srt"]),
        "cuesSha256": digest(cues),
        "timelineSha256": digest(video["timeline"]),
        "faceMapSha256": digest(video.get("faceSpeakers", {})),
        "shotCutsMs": [shot["startMs"] for shot in video["editorTimeline"]["shots"]],
        "stages": video["stages"],
    }


def measure_once(args):
    """Run one isolated job, including native children, and validate its visual pass."""
    from voice_studio.vision import video_revolver
    from voice_studio.speech.media_transcript import transcribe_media

    session = None
    timings = {}

    def make_session(command, imports):
        """Keep production transport and switch only the variant entrypoint."""
        nonlocal session
        if args.worker.resolve() != BUILTIN_WORKER:
            command = [command[0], str(args.worker)]
            imports = [ROOT, *imports]
        session = MeasuredActivitySession(command, imports)
        return session

    def emit(event):
        if "elapsedMs" in event:
            timings[event["stage"]] = event["elapsedMs"]

    with args.source.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    start = monotonic()
    with patch.object(video_revolver, "ActivityWorkerSession", make_session):
        with TemporaryDirectory(prefix="voice-perf-run-") as work:
            report = transcribe_media(Path(work), args.language, args.source, speakers=True, quality=True,
                                      visual=True, scene=args.scene, emit=emit)
            if args.context_db:
                read_context(args.context_db, digest, args.language, report)

    duration = monotonic() - start
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if session is None or report.get("videoEvidence", {}).get("stages", {}).get("light-asd") != "ok":
        raise RuntimeError("Light-ASD did not run successfully")
    visual = session.measurements()
    return {
        "sourceSha256": digest,
        "durationMs": report["durationMs"],
        "workerSha256": sha256(args.worker.read_bytes()).hexdigest(),
        "wallMs": millis(duration),
        "cpuMs": millis(self_usage.ru_utime + self_usage.ru_stime +
                           child_usage.ru_utime + child_usage.ru_stime),
        "largestProcessPeakRssMiB": max(max_rss_mib(self_usage), max_rss_mib(child_usage), visual["peakRssMiB"]),
        "stageWallMs": timings,
        "lightAsdWorker": visual,
        "semantic": semantic_report(report),
    }


def summarize(samples, variant_names):
    """Summarize variant medians and check equality of every semantic fingerprint."""
    reference = samples[0]["semantic"]
    variants = {}
    for name in variant_names:
        rows = [sample for sample in samples if sample["variant"] == name]
        variants[name] = {
            **{metric: median([row[metric] for row in rows])
               for metric in ("wallMs", "cpuMs", "largestProcessPeakRssMiB")},
            "lightAsdWorker": {
                metric: median([row["lightAsdWorker"][metric] for row in rows])
                for metric in ("wallMs", "cpuMs", "peakRssMiB")
            },
            "stageMedianWallMs": {
                stage: median([row["stageWallMs"][stage] for row in rows])
                for stage in rows[0]["stageWallMs"]
            },
            "matchesFirst": all(row["semantic"] == reference for row in rows),
        }
    return variants


def main():
    """Run isolated child jobs in reversed order every other round."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--language", choices=["ru", "en"], default="ru")
    parser.add_argument("--context-db", type=Path)
    parser.add_argument("--scene", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--variant", action="append", metavar="NAME=PATH", help="use a worker supporting --window-server")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure_once(args), ensure_ascii=False))
        return
    if not 1 <= args.samples <= 10:
        parser.error("--samples must be between 1 and 10")
    variants = dict(item.split("=", 1) for item in args.variant or [f"current={BUILTIN_WORKER}"])
    command = [sys.executable, str(Path(__file__).resolve()), str(args.source.resolve()), "--language", args.language]
    if args.context_db:
        command += ["--context-db", str(args.context_db.resolve())]
    if args.scene:
        command += ["--scene"]
    samples = []
    for index in range(args.samples):
        names = list(variants)
        for name in names[::(-1 if index % 2 else 1)]:
            result = subprocess.run([*command, "--worker", str(Path(variants[name]).resolve())],
                                    cwd=ROOT, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stderr[-3000:])
            sample = {"variant": name, "round": index + 1, **json.loads(result.stdout)}
            samples.append(sample)
            print(f"{name} #{index + 1} wall={sample['wallMs']} ms cpu={sample['cpuMs']} ms", file=sys.stderr)
    summary = {"scope": "offline WAV/ASR/visual pipeline without HTTP, browser or WebM playback preview",
               "memoryScope": "largest single process peak RSS, not simultaneous tree RSS",
               "medians": summarize(samples, variants), "samples": samples}
    output = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
