"""Summarize sampled face appearance without using demographics to match identities."""

from voice_studio.mod_runtime import model_resource

from collections import defaultdict
from math import dist, isfinite
from statistics import median


def appearance_model(path):
    """Load the optional OpenVINO CPU model once per sampling process."""
    path = model_resource('vision.appearance', 'builtin.age-gender', path)
    if not path or not path.is_file():
        return None
    try:
        from openvino import Core
        return Core().compile_model(str(path), "CPU", {"INFERENCE_NUM_THREADS": 1})
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def apparent_age_and_sex(model, crop):
    """Return an adult age estimate and the model's binary appearance score."""
    import cv2
    if model is None or min(crop.shape[:2]) < 62:
        return {}
    try:
        input_data = cv2.dnn.blobFromImage(crop, size=(62, 62))
        outputs = model(input_data)
        scores = outputs["prob"].ravel()
        age = float(outputs[model.output(1)].ravel()[0]) * 100
    except (cv2.error, KeyError, RuntimeError, ValueError):
        return {}
    if len(scores) != 2 or not isfinite(age) or not 18 <= age <= 75:
        return {}
    label = int(scores.argmax())
    return {"age": age, "appearanceClass": ("female", "male")[label],
            "appearanceScore": float(scores[label])}


def age_consensus(ages):
    """Allow an outlier only when a clear majority shares one unambiguous age estimate."""
    ordered = sorted(ages)
    windows = [tuple(value for value in ordered if start <= value <= start + 12) for start in ordered]
    if not windows:
        return []
    largest = max(map(len, windows))
    if largest < 2 or largest * 2 <= len(ordered):
        return []
    best = {window for window in windows if len(window) == largest}
    return next(iter(best)) if len(best) == 1 else []


def describe_appearance(samples, aliases, *, allow_age_consensus=False):
    """Publish only attributes that agree across at least two separated sample frames."""
    grouped = defaultdict(dict)
    for row in samples:
        grouped[aliases.get(row["face"], row["face"])][row["timeMs"]] = row
    profiles = {}
    for face, frames in grouped.items():
        rows = list(frames.values())
        if len(rows) < 2 or max(frames) - min(frames) < 120:
            continue
        record = {}
        ages = [row["age"] for row in rows if "age" in row]
        if allow_age_consensus:
            ages = age_consensus(ages)
        if len(ages) >= 2 and max(ages) - min(ages) <= 12:
            record.update(age=round(median(ages)), ageSource="openvino-age-gender", ageSamples=len(ages))
        labels = [row["appearanceClass"] for row in rows if row.get("appearanceScore", 0) >= .85]
        if len(labels) >= 2 and len(set(labels)) == 1:
            record.update(appearanceClass=labels[0], appearanceSource="openvino-age-gender",
                          appearanceSamples=len(labels))
        hair = [row["hairRgb"] for row in rows if "hairRgb" in row]
        if len(hair) >= 2 and max(dist(rgb, hair[0]) for rgb in hair) <= 48:
            rgb = [round(median(channel)) for channel in zip(*hair)]
            record.update(hairSwatch="#" + "".join(f"{value:02x}" for value in rgb),
                          hairSource="mediapipe-hair-mask", hairSamples=len(hair))
        if record:
            profiles[face] = {**record, "sampleTimesMs": sorted(frames)}
    return profiles
