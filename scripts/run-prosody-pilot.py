"""Run controlled local F5 accent-tune pairs in a new isolated workspace."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
from uuid import uuid4

from voice_studio.synthesis.engines import F5_ACCENT_WEIGHTS
from voice_studio.prosody.revolver import mark_lapshi
from voice_studio.app.studio import Studio

REFERENCE_TEXT = "Два друга назначили встречу и договорились ждать друг друга."
CASES = (
    ("У лапши хороший вкус. Хочешь попробовать?", 17027),
    ("У лапши хороший вкус. Хочешь попробовать?", 88219),
    ("Мы остановились у лапши. Потом пошли домой.", 451007),
)


def digest(path):
    """Hash the exact WAV or weights used in the pilot without loading it into memory."""
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def main():
    """Queue three matched pairs and run a second-generation retry through the studio API path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    args.workspace.mkdir(mode=0o700, parents=True)
    studio = Studio(args.workspace)
    ids = []
    try:
        ref = studio.record("ru", REFERENCE_TEXT, args.reference.read_bytes())
        rows = studio.references("ru", [ref])
        for text, seed in CASES:
            takes = [
                ("f5", studio.accent_checkpoint("plain"), seed, "pilot-plain", [(text, 0)]),
                ("f5", studio.accent_checkpoint("lapshi-plus"), seed, "pilot-plus", [(mark_lapshi(text), 0)]),
            ]
            ids.extend(studio.enqueue_synthesis("ru", text, rows, 0, takes, None, batch_id=uuid4().hex))
        studio.worker.submit(lambda: None).result()
        accent = ids[1]
        retries = studio.retry_words("ru", accent, 1, 1, "short")
        studio.worker.submit(lambda: None).result()
        ids.extend(retries)
        ids.extend(studio.retry_words("ru", retries[0], 1, 1, "short"))
        studio.worker.submit(lambda: None).result()
        weight_hash = digest(F5_ACCENT_WEIGHTS)
        report = {"checkpointSha256": weight_hash,
                  "referenceSha256": digest(studio.media / (ref + ".wav")), "runs": []}
        baselines = {}
        for key in ids:
            job = studio.synthesis_result("ru", key)
            graph = studio.inspect_words("ru", key)
            metrics = studio.tracking.client
            run_id = job["run_id"]
            metrics.set_tag(run_id, "pilot_checkpoint_sha256", weight_hash)
            metrics.set_tag(run_id, "pilot_reference_sha256", report["referenceSha256"])
            if job["variant"] == "pilot-plain":
                baselines[key] = run_id
                metrics.set_tag(run_id, "pilot_pair_run", run_id)
            elif job["variant"] == "pilot-plus":
                metrics.set_tag(run_id, "pilot_pair_run", baselines[job["baseline_id"]])
            metrics.log_dict(run_id, graph, "word-graph.json")
            pauses = graph["pauses"]
            metrics.log_metric(run_id, "pilot_pause_count", len(pauses))
            if pauses:
                metrics.log_metric(run_id, "pilot_max_pause_ms", max(pause["durationMs"] for pause in pauses))
            if graph["wordErrorRate"] is not None:
                metrics.log_metric(run_id, "pilot_graph_word_error_rate", graph["wordErrorRate"])
            report["runs"].append({
                "id": key, "run": run_id, "variant": job["variant"], "seed": job["seed"],
                "batch": job["batch_id"], "baseline": job["baseline_id"], "accentMode": job["accent_mode"],
                "wavSha256": digest(studio.media / (key + ".wav")), "graph": graph,
                "metrics": metrics.get_run(run_id).data.metrics,
            })
        (args.workspace / "pilot-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        studio.worker.shutdown()


if __name__ == "__main__":
    main()
