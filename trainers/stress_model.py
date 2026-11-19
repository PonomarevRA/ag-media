"""Train an averaged perceptron to select the stressed vowel in dictionary words."""

from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sys
from time import monotonic

from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_studio.prosody.stress import ROOT, VOWELS, WORD, marked_index, stress_scores, vowel_features


def examples(dictionary, ambiguous):
    """Split related spelling families together and remove ambiguous forms."""
    groups = [[], [], []]
    for word, marked in sorted(dictionary.items()):
        if word in ambiguous or not WORD.fullmatch(word) or "ё" in word:
            continue
        index = marked_index(marked)
        if index is None or sum(char in VOWELS for char in word) < 2:
            continue
        # A prefix proxy groups close inflections in the absence of lemma IDs.
        family = word[:min(4, max(2, len(word) - 2))]
        bucket = int.from_bytes(sha256(family.encode()).digest()[:4], "big") % 100
        groups[0 if bucket < 80 else 1 if bucket < 90 else 2].append((word, index))
    return groups


def train(rows, checkpoints):
    """Continue one averaged-perceptron run and yield cumulative snapshots."""
    random = Random(77)
    weights, totals, stamp = defaultdict(float), defaultdict(float), defaultdict(int)
    step = 0
    order = list(rows)
    for epoch in range(1, max(checkpoints) + 1):
        random.shuffle(order)
        for word, correct in order:
            step += 1
            predicted = stress_scores(word, weights)[0][1]
            if predicted == correct:
                continue
            for position, delta in [(correct, 1), (predicted, -1)]:
                for feature in vowel_features(word, position):
                    totals[feature] += (step - stamp[feature]) * weights[feature]
                    stamp[feature] = step
                    weights[feature] += delta
        if epoch in checkpoints:
            averaged = {}
            for feature, weight in weights.items():
                value = round((totals[feature] + (step - stamp[feature]) * weight) / step, 4)
                if value:
                    averaged[feature] = value
            yield epoch, averaged


def evaluate(rows, weights, margin):
    """Report coverage, conditional accuracy, and single-rule baseline."""
    accepted = correct = baseline = 0
    for word, gold in rows:
        best, second, *_ = stress_scores(word, weights)
        baseline += [i for i, char in enumerate(word) if char in VOWELS][-2] == gold
        if best[0] - second[0] >= margin:
            accepted += 1
            correct += best[1] == gold
    return {"size": len(rows), "coverage": accepted / len(rows),
            "accuracy": correct / max(1, accepted), "penultimate_baseline": baseline / len(rows)}


def write_json(path, data):
    """Publish an intact JSON file, even if training is interrupted."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def log_iteration(client, experiment, epoch, model, row, dictionary_hash):
    """Store one checkpoint, calibration metrics, and training provenance."""
    run = client.create_run(experiment, tags={"mlflow.runName": f"stress-epoch-{epoch}"})
    run_id = run.info.run_id
    try:
        for key, value in {"epochs": epoch, "seed": 77, "dictionary_sha256": dictionary_hash,
                           "split": "prefix-family-v1", "margin": model["margin"]}.items():
            client.log_param(run_id, key, str(value))
        metrics = {"validation_accuracy": row["validation"]["accuracy"],
                   "validation_coverage": row["validation"]["coverage"],
                   "eligible_accuracy": row["accepted"]["accuracy"] if row["accepted"] else 0,
                   "eligible_coverage": row["accepted"]["coverage"] if row["accepted"] else 0,
                   "elapsed_seconds": row["elapsed_seconds"], "features": len(model["weights"])}
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
        path = ROOT / f"stress-epoch-{epoch}.json"
        write_json(path, model)
        client.log_artifact(run_id, str(path))
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    client.set_terminated(run_id, "FINISHED")
    return run_id


def main():
    """Continue on a fixed split and select the active model on calibration only."""
    source = ROOT / "accents_nn.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        dictionary = json.load(handle)
    with gzip.open(ROOT / "omographs.json.gz", "rt", encoding="utf-8") as handle:
        ambiguous = json.load(handle)
    training, calibration, test = examples(dictionary, ambiguous)
    dictionary_hash = sha256(source.read_bytes()).hexdigest()
    client = MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
    name = "Voice Studio - Russian stress"
    existing = client.get_experiment_by_name(name)
    experiment = existing.experiment_id if existing else client.create_experiment(
        name, artifact_location=(ROOT / "mlflow-artifacts").as_uri())
    started = monotonic()
    history = []
    best = None
    for epoch, weights in train(training, (3, 7, 14)):
        margins = [(value, evaluate(calibration, weights, value))
                   for value in (0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30)]
        eligible = [(value, result) for value, result in margins
                    if result["accuracy"] >= .90 and result["coverage"] >= .05]
        margin, accepted = eligible[0] if eligible else (None, None)
        model = {"version": f"perceptron-{epoch}", "margin": margin, "weights": weights}
        row = {"epoch": epoch, "validation": margins[0][1], "accepted": accepted,
               "elapsed_seconds": round(monotonic() - started, 3)}
        row["run_id"] = log_iteration(client, experiment, epoch, model, row, dictionary_hash)
        history.append(row)
        rank = (bool(accepted), (accepted or {}).get("coverage", 0), row["validation"]["accuracy"])
        if best is None or rank > best[0]:
            best = (rank, model, row["run_id"], margins)
    _, model, winner, calibration_results = best
    final = evaluate(test, model["weights"], 0)
    accepted = evaluate(test, model["weights"], model["margin"]) if model["margin"] is not None else None
    client.log_metric(winner, "holdout_accuracy", final["accuracy"])
    client.log_metric(winner, "holdout_penultimate", final["penultimate_baseline"])
    if accepted:
        client.log_metric(winner, "holdout_eligible_accuracy", accepted["accuracy"])
        client.log_metric(winner, "holdout_eligible_coverage", accepted["coverage"])
    client.set_tag(winner, "selected", "true")
    report = {"train": len(training), "calibration": calibration_results, "iterations": history,
              "selected": model["version"], "selected_run": winner, "margin": model["margin"],
              "test_all": final, "test_after_abstention": accepted, "dictionary_sha256": dictionary_hash}
    write_json(ROOT / "stress-metrics.json", report)
    client.log_artifact(winner, str(ROOT / "stress-metrics.json"))
    write_json(ROOT / "stress-model.json", model)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
