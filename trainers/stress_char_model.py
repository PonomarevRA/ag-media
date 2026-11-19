"""Train and review a local, dictionary-free character stress predictor."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_studio.prosody.stress import ROOT, marked_index, stress_scores
from voice_studio.prosody.stress_char_runtime import ALPHABET, MODEL_PATH, candidates, predict

OUTPUT = MODEL_PATH.parent
POINTS = (2, 5, 10, 16)
TARGET_ACCURACY = 0.90
MIN_COVERAGE = 0.05
SEED = 77


def read_gzip(path):
    """Read a dictionary or serialized checkpoint."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def publish_model(output, selected):
    """Activate an already-written model without exposing partial reads."""
    temporary = output / ".selected-new.json.gz"
    temporary.write_bytes((output / selected).read_bytes())
    temporary.replace(output / MODEL_PATH.name)


def split_words(dictionary, ambiguous):
    """Reuse the previous trainer's prefix-grouped partition for paired comparison."""
    groups = [[], [], []]
    for word, marked in sorted(dictionary.items()):
        if word in ambiguous or "ё" in word:
            continue
        index = marked_index(marked)
        entries = candidates(word)
        positions = [row[0] for row in entries]
        if len(entries) < 2 or index not in positions:
            continue
        family = word[:min(4, max(2, len(word) - 2))]
        bucket = int.from_bytes(sha256(family.encode()).digest()[:4], "big") % 100
        groups[0 if bucket < 80 else 1 if bucket < 90 else 2].append((word, entries, positions.index(index)))
    return groups


def loader(rows, shuffle=False):
    """Batch packed candidate inputs, padding only the vowel dimension."""
    import torch
    from torch.utils.data import DataLoader

    def collate(batch):
        """Shape embedded-character inputs and validity mask for one minibatch."""
        longest = max(len(entries) for _, entries, _ in batch)
        char_inputs, numeric_inputs, masks, labels = [], [], [], []
        for _, entries, label in batch:
            missing = longest - len(entries)
            char_inputs.append([row[1] for row in entries] + [[0] * 17] * missing)
            numeric_inputs.append([row[2] for row in entries] + [[0] * 6] * missing)
            masks.append([True] * len(entries) + [False] * missing)
            labels.append(label)
        return (torch.tensor(char_inputs), torch.tensor(numeric_inputs), torch.tensor(masks), torch.tensor(labels))

    return DataLoader(rows, batch_size=384, shuffle=shuffle, collate_fn=collate)


def build_network():
    """Create a per-vowel scorer with a shared character embedding."""
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            """Build a small scorer shared across positions and word lengths."""
            super().__init__()
            self.embedding = nn.Embedding(len(ALPHABET), 8)
            self.layers = nn.ModuleList((nn.Linear(17 * 8 + 6, 80), nn.Linear(80, 32)))
            self.output = nn.Linear(32, 1)

        def forward(self, chars, features, mask):
            """Score all vowel candidates and mask padded positions."""
            vector = torch.cat((self.embedding(chars).flatten(-2), features), dim=-1)
            for layer in self.layers:
                vector = layer(vector).relu()
            return self.output(vector).squeeze(-1).masked_fill(~mask, float("-inf"))

    return Network()


def serialize(network, epoch):
    """Export tensors as portable finite arrays without pickled code."""
    def rounded(tensor):
        """Round weight arrays to six decimal places for transport."""
        return tensor.detach().numpy().round(6).tolist()

    def layer_row(layer):
        """Export one affine layer in the runtime's weight layout."""
        return {"weights": rounded(layer.weight), "bias": rounded(layer.bias)}

    return {"version": f"char-mlp-{epoch}", "embedding": rounded(network.embedding.weight),
            "layers": [layer_row(layer) for layer in network.layers],
            "output_weights": rounded(network.output.weight[0]),
            "output_bias": float(network.output.bias[0].detach())}


def predictions(network, data):
    """Collect one correctness flag and one confidence margin per word."""
    import torch

    network.eval()
    pairs = []
    with torch.no_grad():
        for chars, features, mask, gold in data:
            scores, indices = network(chars, features, mask).topk(2, dim=1)
            gaps = (scores[:, 0] - scores[:, 1]).tolist()
            pairs.extend(zip((indices[:, 0] == gold).tolist(), gaps))
    return pairs


def metrics(rows, margin=0):
    """Calculate coverage and conditional and total accuracy at a threshold."""
    accepted = [correct for correct, confidence in rows if confidence >= margin]
    return {"size": len(rows), "accepted": len(accepted), "correct": sum(accepted),
            "accuracy": sum(accepted) / max(1, len(accepted)), "coverage": len(accepted) / len(rows)}


def calibrate(rows):
    """Pick the highest-coverage margin meeting the validation criteria."""
    options = [step / 4 for step in range(33)]
    for margin in options:
        result = metrics(rows, margin)
        if result["accuracy"] >= TARGET_ACCURACY and result["coverage"] >= MIN_COVERAGE:
            return margin, result
    return None, None


def legacy_scores(rows):
    """Measure the previous perceptron on each candidate of the exact same holdout."""
    model = json.loads((ROOT / "stress-model.json").read_text(encoding="utf-8"))
    scored = []
    for word, entries, label in rows:
        best, second, *_ = stress_scores(word, model["weights"])
        scored.append((best[1] == entries[label][0], best[0] - second[0]))
    return metrics(scored)


def run_training(output):
    """Train checkpoints, select exclusively on validation, and evaluate holdout once."""
    import torch
    from torch.nn.functional import cross_entropy

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    source = ROOT / "accents_nn.json.gz"
    rows = split_words(read_gzip(source), read_gzip(ROOT / "omographs.json.gz"))
    validation, holdout = map(loader, rows[1:])
    train = loader(rows[0], shuffle=True)
    net = build_network()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.002)
    report = {"seed": SEED, "split": "prefix-family-v1", "dictionary_sha256": sha256(source.read_bytes()).hexdigest(),
              "train_size": len(rows[0]), "validation_size": len(rows[1]), "holdout_size": len(rows[2]), "iterations": []}
    output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    best = None
    for epoch in range(1, max(POINTS) + 1):
        net.train()
        for chars, features, mask, gold in train:
            optimizer.zero_grad()
            loss = cross_entropy(net(chars, features, mask), gold)
            loss.backward()
            optimizer.step()
        if epoch not in POINTS:
            continue
        calculated = predictions(net, validation)
        margin, eligible = calibrate(calculated)
        model = serialize(net, epoch)
        model["margin"] = margin
        checkpoint = output / f"{model['version']}.json.gz"
        with gzip.open(checkpoint, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(model, handle, separators=(",", ":"))
        row = {"epoch": epoch, "version": model["version"], "validation": metrics(calculated), "eligible": eligible,
               "margin": margin, "file": checkpoint.name, "bytes": checkpoint.stat().st_size,
               "seconds": round(monotonic() - started, 2)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        report["iterations"].append(row)
        rank = (eligible["coverage"] if eligible else 0, row["validation"]["accuracy"])
        if best is None or rank > best[0]:
            best = rank, model, row, {key: value.clone() for key, value in net.state_dict().items()}
    _, model, selected, state = best
    net.load_state_dict(state)
    scored = predictions(net, holdout)
    report["legacy"] = {"validation": legacy_scores(rows[1]), "holdout": legacy_scores(rows[2])}
    report["selected"] = selected["version"]
    report["holdout"] = metrics(scored)
    margin = model["margin"]
    report["holdout_eligible"] = metrics(scored, margin) if margin is not None else None
    # Compare the portable runtime with the training network on an untouched subset.
    parity_count = 0
    chars, features, mask, _ = next(iter(loader(rows[2][:32])))
    with torch.no_grad():
        logits = net(chars, features, mask).argmax(dim=1).tolist()
    for (word, entries, _), (_, expected_margin), expected in zip(rows[2][:32], scored[:32], logits, strict=True):
        index, runtime_margin = predict(word, model)
        if index == entries[expected][0] and abs(runtime_margin - expected_margin) < .0001:
            parity_count += 1
    report["portable_margin_parity"] = {"count": parity_count, "total": 32}
    publish_model(output, selected["file"])
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "iterations"}, ensure_ascii=False), flush=True)


def log_training(output):
    """Log the actual checkpoints and evaluation metrics to the local MLflow store."""
    from mlflow import MlflowClient

    report_path = output / "metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
    name = "Voice Studio - Stress revolver"
    existing = client.get_experiment_by_name(name)
    experiment = existing.experiment_id if existing else client.create_experiment(
        name, artifact_location=(ROOT / "mlflow-artifacts").as_uri())

    def log_scores(run_id, split, values):
        """Record all measured metrics for one evaluation partition."""
        for field, value in values.items():
            client.log_metric(run_id, f"{split}_{field}", value)

    run_ids = {}
    for row in report["iterations"]:
        selected = row["version"] == report["selected"]
        run = client.create_run(experiment, tags={"mlflow.runName": row["version"], "stage": "lexical",
                                                   "selected_on_validation": str(selected).lower()})
        run_id = run.info.run_id
        try:
            for key in ("seed", "split", "dictionary_sha256", "train_size", "validation_size", "holdout_size"):
                client.log_param(run_id, key, report[key])
            for key in ("epoch", "margin", "bytes", "width", "alpha"):
                if key in row:
                    client.log_param(run_id, key, str(row[key]))
            log_scores(run_id, "validation", row["validation"])
            if row["eligible"]:
                log_scores(run_id, "validation_calibrated", row["eligible"])
            client.log_metric(run_id, "training_seconds", row["seconds"])
            client.log_artifact(run_id, str(output / row["file"]))
            if selected:
                log_scores(run_id, "holdout", report["holdout"])
                log_scores(run_id, "holdout_legacy", report["legacy"]["holdout"])
                if report["holdout_eligible"]:
                    log_scores(run_id, "holdout_calibrated", report["holdout_eligible"])
                client.log_artifact(run_id, str(report_path))
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise
        client.set_terminated(run_id, "FINISHED")
        run_ids[row["version"]] = run_id
    report["mlflow_experiment_id"] = experiment
    report["mlflow_run_ids"] = run_ids
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_ids, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "neural")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    (log_training if args.log else run_training)(args.output)
