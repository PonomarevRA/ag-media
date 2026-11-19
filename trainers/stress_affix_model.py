"""Learn a dictionary-free stress ranker from character-ending patterns."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_studio.prosody.stress import ROOT
from voice_studio.prosody.stress_char_runtime import PACKED_MAGIC, affix_features, feature_id
from trainers.stress_char_model import (
    OUTPUT, SEED, calibrate, legacy_scores, metrics, publish_model, read_gzip, split_words,
)

MIN_OCCURRENCES = 3


def pack_weights(checkpoint, destination):
    """Export collision-checked feature IDs and float weights for local inference."""
    model = read_gzip(checkpoint)
    items = sorted((feature_id(key), weight) for key, weight in model.pop("weights").items())
    if len({key for key, _ in items}) != len(items):
        raise ValueError("Stress weight feature hash collision.")
    header = json.dumps(model, separators=(",", ":")).encode()
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(PACKED_MAGIC)
        handle.write(struct.pack("<II", len(header), len(items)))
        handle.write(header)
        for key, _ in items:
            handle.write(struct.pack("<Q", key))
        for _, weight in items:
            handle.write(struct.pack("<f", weight))
    temporary.replace(destination)


def encode(rows, width, known=None):
    """Make sparse binary features for each vowel, preserving each word's group."""
    import numpy as np
    from scipy.sparse import csr_matrix

    pairs = []
    offsets = [0]
    labels = []
    for word, entries, gold in rows:
        for index, _, _ in entries:
            pairs.append(tuple(affix_features(word, index, width)))
        offsets.append(len(pairs))
        labels.append(gold)
    if known is None:
        counts = Counter(feature for pair in pairs for feature in pair)
        known = {feature: index for index, feature in enumerate(
            feature for feature, count in counts.items() if count >= MIN_OCCURRENCES)}
    indices = []
    starts = [0]
    for pair in pairs:
        indices.extend(known[feature] for feature in pair if feature in known)
        starts.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    mat = csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(starts, dtype=np.int32)),
                     shape=(len(pairs), len(known)))
    return mat, np.asarray(offsets), np.asarray(labels), known


def predict_groups(scored, offsets, labels):
    """Collect correctness and runner-up margins from flat candidate scores."""
    import numpy as np
    groups = []
    for offset, end, gold in zip(offsets[:-1], offsets[1:], labels, strict=True):
        group = scored[offset:end]
        top = int(np.argmax(group))
        runner_up = np.partition(group, len(group) - 2)[-2]
        groups.append((bool(top == gold), float(group[top] - runner_up)))
    return groups


def train(output):
    """Train three suffix spans and select by independent calibration."""
    import numpy as np
    from scipy.sparse import vstack
    from sklearn.linear_model import SGDClassifier

    source = ROOT / "accents_nn.json.gz"
    train_rows, validation, holdout = split_words(read_gzip(source), read_gzip(ROOT / "omographs.json.gz"))
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed": SEED, "split": "prefix-family-v1", "dictionary_sha256": sha256(source.read_bytes()).hexdigest(),
              "train_size": len(train_rows), "validation_size": len(validation), "holdout_size": len(holdout),
              "iterations": []}
    best = None
    started = monotonic()
    for width in (4, 7, 10):
        matrix, offsets, gold, vocab = encode(train_rows, width)
        true_rows = np.repeat(offsets[:-1] + gold, np.diff(offsets))
        choices = np.arange(matrix.shape[0])
        wrong = choices != true_rows
        # Learn relative likelihood rather than fitting absolute stress labels independently.
        positive = matrix[true_rows[wrong]] - matrix[wrong]
        paired = vstack((positive, -positive), format="csr")
        labels = np.repeat([1, 0], positive.shape[0])
        evaluate, indices, expected, _ = encode(validation, width, vocab)
        for alpha in (0.00003, 0.0003):
            classifier = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=25, random_state=SEED,
                                       fit_intercept=False, average=True, tol=0.0001)
            classifier.fit(paired, labels)
            weights = classifier.coef_[0]
            scored = predict_groups(evaluate @ weights, indices, expected)
            margin, eligible = calibrate(scored)
            version = f"affix-{width}-reg-{alpha:g}"
            checkpoint = output / f"{version}.json.gz"
            exported = {feature: round(float(weights[i]), 6) for feature, i in vocab.items() if abs(weights[i]) >= 1e-5}
            model = {"version": version, "kind": "affix", "width": width, "weights": exported, "margin": margin}
            with gzip.open(checkpoint, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(model, handle, ensure_ascii=False, separators=(",", ":"))
            row = {"version": version, "epoch": classifier.n_iter_, "width": width, "alpha": alpha,
                   "validation": metrics(scored), "eligible": eligible, "margin": margin, "file": checkpoint.name,
                   "bytes": checkpoint.stat().st_size, "seconds": round(monotonic() - started, 2)}
            report["iterations"].append(row)
            print(json.dumps(row), flush=True)
            rank = (eligible["coverage"] if eligible else 0, row["validation"]["accuracy"])
            if best is None or rank > best[0]:
                best = rank, model, row, weights, vocab
    _, model, selected, weights, vocab = best
    evaluate, indices, expected, _ = encode(holdout, model["width"], vocab)
    scored = predict_groups(evaluate @ weights, indices, expected)
    report["legacy"] = {"validation": legacy_scores(validation), "holdout": legacy_scores(holdout)}
    report["selected"] = selected["version"]
    report["holdout"] = metrics(scored)
    report["holdout_eligible"] = metrics(scored, model["margin"]) if model["margin"] is not None else None
    publish_model(output, selected["file"])
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "iterations"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "affix")
    parser.add_argument("--pack", nargs=2, type=Path, metavar=("CHECKPOINT", "DESTINATION"))
    args = parser.parse_args()
    if args.pack:
        pack_weights(*args.pack)
    else:
        train(args.output)
