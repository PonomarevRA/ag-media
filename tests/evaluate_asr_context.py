"""Benchmark a fixed dictionary and its context links against the same ASR hypotheses."""

import argparse
import json
import os
import re
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

from voice_studio.speech.asr_context_graph import VERSION, apply_corrections, corrections
from voice_studio.speech.recognition import transcribe
from voice_studio.synthesis.speech_metrics import word_error_rate

FIXTURES = (
    ("В коробке лежат пять шариков.", "В коробке лежат пять шериков."),
    ("Для игры надули шесть шариков.", "Для игры надули шесть шаликов."),
    ("Маленьких шариков больше нет.", "Маленьких шеликов больше нет."),
    ("Шесть шариков покатились вниз.", "Шесть шеликов покатились вниз."),
    ("Подпись Шеликов на письме.", "Подпись Шеликов на письме."),
    ("Инженер Шериков принёс пять чертежей.", "Инженер Шериков принёс пять чертежей."),
    ("Врач Шаликов приехал в шесть.", "Врач Шаликов приехал в шесть."),
    ("Шеликов позвонил. Пять минут мы ждали.", "Шеликов позвонил. Пять минут мы ждали."),
    ("Отдел Шериков закончил отчёт.", "Отдел Шериков закончил отчёт."),
    ("На столе лежат четыре кубика.", "На столе лежат четыре кубика."),
    ("Надули пять шариков.", "Надули пять шариков."),
    ("Раздался громкий звонок.", "Раздался громкий звонок."),
)


def measure(pairs):
    """Return aggregate WER and false edits on lines with an exact baseline."""
    report = {}
    for mode in ("whisper", "flat", "graph"):
        words = errors = changed = false_clean = improved = regressed = 0
        started = perf_counter()
        for gold, raw in pairs:
            edits = [] if mode == "whisper" else corrections(raw, "ru", contextual=mode == "graph")
            revised = apply_corrections(raw, edits)
            before = word_error_rate(gold, raw)
            after = word_error_rate(gold, revised)
            n = len(re.findall(r"\w+", gold))
            words += n
            errors += round(after * n)
            changed += len(edits)
            false_clean += before == 0 and bool(edits) and after != 0
            improved += after < before
            regressed += after > before
        report[mode] = dict(lines=len(pairs), words=words, errors=errors, wer=errors / words,
                            edits=changed, false_clean=false_clean, improved=improved, regressed=regressed,
                            milliseconds=round((perf_counter() - started) * 1000, 3))
    return report


def recording_pairs(studio, scratch):
    """Read local inputs without printing, copying or changing the corpus."""
    db_path = studio / "studio.db"
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT id, text FROM recordings WHERE language='ru' ORDER BY rowid").fetchall()
    return [(text, transcribe(scratch, "ru", (studio / "media" / f"{key}.wav").read_bytes()))
            for key, text in rows]


def synthesis_pairs(studio, keys, scratch):
    """Read selected completed WAVs whose saved scripts supply evaluation targets."""
    with sqlite3.connect(f"{(studio / 'studio.db').as_uri()}?mode=ro", uri=True) as db:
        rows = []
        for key in dict.fromkeys(keys):
            row = db.execute("SELECT id, text FROM jobs WHERE id=? AND language='ru' "
                             "AND kind='synthesis' AND status='done' AND demo=0", (key,)).fetchone()
            if row is None:
                raise ValueError("A selected synthesis result is unavailable.")
            rows.append(row)
    return [(text, transcribe(scratch, "ru", (studio / "media" / f"{key}.wav").read_bytes()))
            for key, text in rows]


def main():
    """Run the fixed comparison and optionally log aggregates only to local MLflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", type=Path)
    parser.add_argument("--mlflow-db", type=Path)
    parser.add_argument("--synthesis", nargs="+", help="Completed studio result IDs")
    args = parser.parse_args()
    if args.synthesis and not args.studio:
        parser.error("--synthesis requires --studio")
    reports = {"controlled": measure(FIXTURES)}
    if args.studio:
        with TemporaryDirectory() as folder:
            pairs = recording_pairs(args.studio.resolve(), Path(folder))
            if args.synthesis:
                generated = synthesis_pairs(args.studio.resolve(), args.synthesis, Path(folder))
                reports["synthesis"] = measure(generated)
        if pairs:
            reports["recordings"] = measure(pairs)
    if args.mlflow_db:
        from mlflow.tracking import MlflowClient

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        client = MlflowClient(tracking_uri=f"sqlite:///{args.mlflow_db.resolve()}")
        name = "asr-context-graph"
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        for cohort, inputs in reports.items():
            for mode, metrics in inputs.items():
                run = client.create_run(experiment_id, tags={"mlflow.runName": f"{cohort}-{mode}"})
                key = run.info.run_id
                client.log_param(key, "context_version", VERSION)
                for field, value in metrics.items():
                    client.log_metric(key, field, value)
                client.set_terminated(key)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
