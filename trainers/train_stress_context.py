"""Fit and compare successive sparse homograph models in local MLflow."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

from mlflow import MlflowClient

from voice_studio.prosody.prosody import analyze
from voice_studio.prosody.stress import ROOT, marked_index
from voice_studio.prosody.stress_context import WORDS, context_features, predict
from voice_paths import MODELS_DIR, publish_file, temporary_directory

HERE = Path(__file__).resolve().parent
TARGET = re.compile(r"\[([а-яА-ЯёЁ+]+)\]")
STAGES = ("neighbors", "previous-word", "previous-and-prefix")


def parse_line(sentence, implied=None):
    """Return the target, its expected vowel (or abstain), and its two contexts."""
    match, = TARGET.finditer(sentence)
    marked = implied or match[1].casefold()
    word = marked.replace("+", "")
    if word != match[1].casefold().replace("+", ""):
        raise ValueError(f"Несовпадение целевого слова: {sentence}")
    return word, marked_index(marked), sentence[:match.start()], sentence[match.end():]


def training_sets():
    """Reserve the last independently written phrase for every accent for calibration."""
    train, calibration = [], []
    for line in (HERE / "stress_context_train.tsv").read_text(encoding="utf-8").splitlines():
        marked, *sentences = line.split("\t")
        train.extend(parse_line(sentence, marked) for sentence in sentences[:-1])
        calibration.append(parse_line(sentences[-1], marked))
    neutral = ["Уже нашли [атлас]?", "Это не белые [гвоздики].", "Они увидели [орган].",
               "Слово [мука] написано правильно.", "Приведи пример со словом [пропасть]."]
    calibration.extend(map(parse_line, neutral))
    # Separately calibrate direct-word cues against new continuations with identical suffixes.
    previous_pairs = (
        ("географический [+атлас]", "шелковый [атл+ас]"),
        ("пушистые [б+елки]", "молочные [белк+и]"),
        ("минутные [стр+елки]", "военные [стрелк+и]"),
        ("внутренний [+орган]", "церковный [орг+ан]"),
        ("керамические [кр+ужки]", "танцевальные [кружк+и]"),
        ("книжные [п+олки]", "пехотные [полк+и]"),
    )
    distractors = ("для платья", "в твороге", "на турнире", "обследовали врачи", "в буфете", "в шкафу")
    for (first, second), distractor in zip(previous_pairs, distractors, strict=True):
        for phrase in (first, second):
            calibration.append(parse_line(phrase + " упоминались в записи."))
            calibration.append(parse_line(phrase + f" ({distractor} — подпись в соседней колонке)."))
    return train, calibration


def fit(rows, stage):
    """Estimate a sparse discriminative weight for each sense from labeled phrases."""
    counts = defaultdict(lambda: defaultdict(Counter))
    for word, accent, left, right in rows:
        counts[word][accent].update(context_features(left, right, stage))
    weights = {}
    for word, senses in counts.items():
        totals = Counter()
        for features in senses.values():
            totals.update(features.keys())
        # Features present in both senses carry no evidence for either pronunciation.
        weights[word] = {str(accent): {key: round(value, 3) for key, value in features.items() if totals[key] == 1}
                         for accent, features in senses.items()}
    return {"stage": stage, "weights": weights}


def answer(row, model, *, include_previous=True):
    """Return the production baseline mark or a shadow model's proposed index."""
    word, _, left, right = row
    text = left + word + right
    marks = analyze(text, "ru", experimental=False)["stress"]
    baseline = next((item["index"] for item in marks if item["start"] == len(left)), None)
    candidate = predict(word, left, right, model, include_previous=include_previous) if model else None
    return candidate[0] if candidate else baseline


def measure(rows, predictor):
    """Count correct marks, wrong marks and appropriately unanswered targets."""
    count = Counter()
    predictions = []
    for word, gold, left, right in rows:
        actual = predictor((word, gold, left, right))
        outcome = ("correct_abstention" if gold is None else "missed") if actual is None else (
            "correct" if actual == gold else "wrong")
        count[outcome] += 1
        predictions.append({"word": word, "expected": gold, "actual": actual, "context": left + word + right})
    offered = count["correct"] + count["wrong"]
    return dict(size=len(rows), correct=count["correct"], wrong=count["wrong"], missed=count["missed"],
                correct_abstention=count["correct_abstention"], precision=count["correct"] / max(1, offered),
                recall=count["correct"] / max(1, sum(row[1] is not None for row in rows))), predictions


def read_holdouts():
    """Read independent sentence evaluations without making them available to fit or selection."""
    groups = defaultdict(list)
    for name in ("context", "previous"):
        for line in (HERE / f"stress_{name}_holdout.tsv").read_text(encoding="utf-8").splitlines():
            category, sentence = line.split("\t")
            groups[f"{name}_{category}"].append(parse_line(sentence))
    return groups


def without_previous(row):
    """Mask the immediate left token while preserving all other words and punctuation."""
    word, gold, left, right = row
    cut = re.sub(r"[а-яА-ЯёЁ]+([^а-яА-ЯёЁ]*)$", r"\1", left)
    return word, gold, cut, right


def previous_cue(row):
    """Find the word and direct-left token used to audit cue overlap."""
    word, _, left, _ = row
    tokens = WORDS.findall(left.casefold())
    return word, tokens[-1] if tokens else ""


def log_run(client, experiment, name, model, scores, details, train_hash, selected):
    """Persist per-slice metrics and inspectable predictions for one model."""
    run_id = client.create_run(experiment, tags={"mlflow.runName": name, "selected": str(selected).lower()}).info.run_id
    try:
        client.log_param(run_id, "training_sha256", train_hash)
        client.log_param(run_id, "stage", model.get("stage", 0))
        client.log_param(run_id, "threshold", model.get("threshold", "none"))
        for group, metrics in scores.items():
            for key, value in metrics.items():
                client.log_metric(run_id, f"{group}_{key}", value)
        client.log_dict(run_id, details, "predictions.json")
        if model:
            client.log_dict(run_id, model, "context-model.json")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    client.set_terminated(run_id)
    return run_id


def main():
    """Train three feature configurations, choose by calibration and report the frozen slices."""
    training, calibration = training_sets()
    models = [{}]
    for stage in range(1, 4):
        model = fit(training, stage)
        candidates = []
        for threshold in (0.5, 1.0, 1.5, 2.0, 3.0):
            model["threshold"] = threshold
            metrics, _ = measure(calibration, lambda row: answer(row, model))
            candidates.append((metrics["correct"] - 3 * metrics["wrong"], metrics["correct"], threshold))
        model["threshold"] = max(candidates)[2]
        models.append(model)
    ranks = []
    for model in models:
        values, _ = measure(calibration, lambda row: answer(row, model))
        ranks.append((values["correct"] - 3 * values["wrong"], -values["wrong"]))
    winner = max(range(len(ranks)), key=ranks.__getitem__)
    groups = {"calibration": calibration, **read_holdouts()}
    seen_cues = {previous_cue(row) for row in training}
    previous_rows = groups["previous_previous"]
    groups["previous_seen_cue"] = [row for row in previous_rows if previous_cue(row) in seen_cues]
    groups["previous_unseen_cue"] = [row for row in previous_rows if previous_cue(row) not in seen_cues]
    selected_model = MODELS_DIR / "prosody" / "context.json"
    selected_model.parent.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}")
    name = "Voice Studio - stress shadow context"
    existing = client.get_experiment_by_name(name)
    experiment = existing.experiment_id if existing else client.create_experiment(
        name, artifact_location=(ROOT / "mlflow-artifacts").as_uri())
    train_hash = sha256((HERE / "stress_context_train.tsv").read_bytes()).hexdigest()
    rows = []
    for index, (run_name, model) in enumerate(zip(("baseline", *STAGES), models, strict=True)):
        started = perf_counter()
        pairs = {key: measure(items, lambda row: answer(row, model)) for key, items in groups.items()}
        if model.get("stage", 0) >= 2:
            pairs["previous_without_position"] = measure(
                groups["previous_previous"], lambda row: answer(row, model, include_previous=False))
            pairs["previous_masked_word"] = measure(
                groups["previous_previous"], lambda row: answer(without_previous(row), model))
        scores = {key: pair[0] for key, pair in pairs.items()}
        scores["runtime"] = {"evaluation_seconds": perf_counter() - started}
        details = {key: pair[1] for key, pair in pairs.items()}
        rows.append({"name": run_name, "metrics": scores, "run_id": log_run(
            client, experiment, run_name, model, scores, details, train_hash, index == winner)})
    report = {"training_size": len(training), "selected": rows[winner]["name"], "iterations": rows}
    metrics = ROOT / "stress-revolver" / "context-metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with temporary_directory(prefix="stress-context-") as folder:
        for path, data in ((selected_model, models[winner]), (metrics, report)):
            temporary = Path(folder) / path.name
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            publish_file(temporary, path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
