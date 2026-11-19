"""Project measured expression samples onto source time without inferring mental states."""

from collections import defaultdict
from bisect import bisect_left, bisect_right
from math import isfinite
import re

from voice_studio.vision.static_face_evidence import annotate_static_faces, valid_signature

LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
MODEL_ID = "opencv-zoo-mobilefacenet-2022july"
MODEL_SHA256 = "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"
SAMPLE_STEP_MS = 480
THRESHOLD = .6
INTERVIEW_PROMPT = re.compile(
    r"^(?:я(?:\s+так)?\s+правильно\s+понимаю\b|расскажи(?:те)?(?:\s*,\s*|\s+)пожалуйста\b|"
    r"could\s+you\s+tell\b|tell\s+us\s+about\b|do\s+you\s+mean\b)",
    re.IGNORECASE,
)


def merged_duration(spans):
    """Count a union of half-open millisecond spans once, including simultaneous faces."""
    total = 0
    previous_end = 0
    for start, end in sorted(spans):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def valid_samples(rows, duration):
    """Reject malformed imported samples and deduplicate the same track measurement."""
    result = {}
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, dict):
            continue
        time, track, face = row.get("timeMs"), row.get("track"), row.get("faceId")
        confidence = row.get("confidence")
        if (type(time) is not int or not 0 <= time < duration or type(track) is not int or track < 1
                or type(face) is not int or face < 1 or type(confidence) not in (int, float)
                or not isfinite(confidence) or not 0 <= confidence <= 1):
            continue
        label = row.get("label")
        known = label in LABELS and confidence >= THRESHOLD and row.get("status") == "ok"
        probabilities = {}
        raw = row.get("probabilities")
        for name in LABELS:
            value = raw.get(name) if isinstance(raw, dict) else None
            if type(value) in (int, float) and isfinite(value) and 0 <= value <= 1:
                probabilities[name] = value
        clean = {"timeMs": time, "track": track, "faceId": face, "label": label if known else "unknown",
                 "confidence": confidence, "status": "ok" if known else "unknown", "probabilities": probabilities}
        signature = row.get("faceCropSignature")
        if valid_signature(row) is not None:
            clean["faceCropSignature"] = {key: signature[key] for key in (
                "version", "sha256", "basis", "normalizedWidth", "normalizedHeight", "channels", "comparison")}
        raw_label = row.get("rawLabel", label)
        clean.update(rawLabel=raw_label if raw_label in LABELS else "unknown", rawConfidence=confidence,
                     rawProbabilities=probabilities, evidenceStatus="model-candidate",
                     verificationStatus="unverified", internalStateInference=False)
        if not known:
            clean["reason"] = str(row.get("reason") or "low-confidence")[:80]
        result[time, track] = clean
    return sorted(result.values(), key=lambda row: (row["timeMs"], row["track"]))


def visibility(evidence, duration):
    """Return exact tracked face spans and lookup cells from the existing geometry timeline."""
    timeline = evidence.get("timeline") or {}
    frames = timeline.get("frames") or []
    step = timeline.get("stepMs", 40)
    if type(step) is not int or step < 1:
        return {}, {}, 40
    by_face = defaultdict(list)
    lookup = {}
    for index, hits in enumerate(frames):
        start, end = index * step, min((index + 1) * step, duration)
        if start >= end:
            break
        seen = set()
        for hit in hits:
            if not isinstance(hit, (list, tuple)) or len(hit) < 7:
                continue
            track, face = hit[0], hit[6]
            if type(track) is int and type(face) is int and track > 0 and face > 0:
                lookup[index, track] = face
                if face not in seen:
                    spans = by_face[face]
                    if spans and spans[-1][1] == start:
                        spans[-1] = (spans[-1][0], end)
                    else:
                        spans.append((start, end))
                    seen.add(face)
    return dict(by_face), lookup, step


def expression_intervals(samples, evidence, duration, lookup, step):
    """Extend a sample only through continuous same-track geometry, up to the next cut/sample."""
    cuts = sorted({row["startMs"] for row in (evidence.get("editorTimeline") or {}).get("shots", [])
                   if isinstance(row, dict) and type(row.get("startMs")) is int})
    cuts.extend((evidence.get("faceTracking") or {}).get("identityCutTimesMs", []))
    cuts = sorted({time for time in cuts if type(time) is int and 0 <= time <= duration})
    next_sample = {}
    intervals = []
    for sample in reversed(samples):
        start, track, face = sample["timeMs"], sample["track"], sample["faceId"]
        end = min(duration, start + SAMPLE_STEP_MS, next_sample.get(track, duration))
        next_sample[track] = start
        if sample["status"] != "ok" or start % step:
            continue
        cut_index = bisect_right(cuts, start)
        if cut_index < len(cuts):
            end = min(end, cuts[cut_index])
        position = start
        while position < end and lookup.get((position // step, track)) == face:
            position = min(end, position + step)
        if position > start:
            intervals.append({"startMs": start, "endMs": position, "track": track, "faceId": face,
                              "label": sample["label"], "confidence": sample["confidence"],
                              "evidenceStatus": "model-candidate", "verificationStatus": "unverified",
                              "internalStateInference": False, "reactionEligible": sample.get("reactionEligible", True),
                              "staticFaceStatus": sample.get("staticFaceStatus", "unknown")})
    return sorted(intervals, key=lambda row: (row["startMs"], row["track"]))


def label_statistics(intervals):
    """Weight label shares by classified face-time; never count an unknown as neutral."""
    totals = defaultdict(int)
    for row in intervals:
        totals[row["label"]] += row["endMs"] - row["startMs"]
    denominator = sum(totals.values())
    return {label: {"durationMs": totals[label], "share": totals[label] / denominator if denominator else 0}
            for label in LABELS}


def summarize(intervals, by_face, duration):
    """Keep source-time coverage separate from label shares across simultaneous faces."""
    known_spans = [(row["startMs"], row["endMs"]) for row in intervals]
    classified = merged_duration(known_spans)
    all_visible = [span for spans in by_face.values() for span in spans]
    faces = []
    classified_by_face = defaultdict(list)
    for row in intervals:
        classified_by_face[row["faceId"]].append(row)
    for face, spans in sorted(by_face.items()):
        visible = merged_duration(spans)
        rows = classified_by_face.get(face, [])
        known = merged_duration([(row["startMs"], row["endMs"]) for row in rows])
        faces.append({"faceId": face, "observedMs": visible, "classifiedMs": known,
                      "unknownMs": visible - known, "coverage": known / visible if visible else 0,
                      "byLabel": label_statistics(rows)})
    return {"durationMs": duration, "observedMs": merged_duration(all_visible), "classifiedMs": classified,
            "unknownMs": duration - classified, "coverage": classified / duration if duration else 0,
            "observedFaceMs": sum(row["observedMs"] for row in faces),
            "classifiedFaceMs": sum(end - start for start, end in known_spans),
            "byLabel": label_statistics(intervals), "faces": faces}


def question_basis(cue):
    """Recognize explicit question evidence and a small anchored interview-prompt vocabulary.

    ASR can render interview questions with periods. Literal prompt prefixes supplement
    punctuation; they do not search inside an answer or infer what the speaker intended.
    An explicit boolean annotation overrides automatic detection, including exclusion.
    """
    annotation = cue.get("isQuestion")
    if type(annotation) is bool:
        return "explicit-annotation" if annotation else None
    text = cue.get("text")
    if not isinstance(text, str):
        return None
    if "?" in text:
        return "question-mark"
    if INTERVIEW_PROMPT.match(text.strip()):
        return "interview-prompt-pattern"
    return None


def question_reactions(phrases, intervals, by_face, duration):
    """Describe visible expressions during evidenced questions/prompts and the following 3 seconds."""
    reactions = []
    interval_starts = [row["startMs"] for row in intervals]
    for index, cue in enumerate(phrases, 1):
        text = cue.get("text")
        start, stop = cue.get("startMs"), cue.get("endMs")
        detected_basis = question_basis(cue)
        if (not detected_basis or not isinstance(text, str) or type(start) is not int or type(stop) is not int
                or not 0 <= start < min(stop, duration)):
            continue
        end = min(duration, stop + 3000)
        first = bisect_left(interval_starts, start - SAMPLE_STEP_MS)
        stop_index = bisect_left(interval_starts, end)
        candidates = intervals[first:stop_index]
        rows = [{**row, "startMs": max(start, row["startMs"]), "endMs": min(end, row["endMs"])}
                for row in candidates if row["endMs"] > start]
        visible = {face: [(max(start, left), min(end, right)) for left, right in spans
                          if left < end and right > start] for face, spans in by_face.items()}
        visible = {face: spans for face, spans in visible.items() if spans}
        faces = summarize(rows, visible, end - start)["faces"]
        for face in faces:
            label = max(LABELS, key=lambda name: face["byLabel"][name]["durationMs"])
            face["dominantLabel"] = label if face["classifiedMs"] else "unknown"
            face["dominantShare"] = face["byLabel"][label]["share"] if face["classifiedMs"] else 0
            face.update(evidenceStatus="model-candidate", verificationStatus="unverified", internalStateInference=False)
        reactions.append({"questionCue": index, "questionStartMs": start, "questionEndMs": stop,
                          "text": text[:4096], "windowStartMs": start, "windowEndMs": end,
                          "evidenceStatus": "model-candidate", "verificationStatus": "unverified",
                          "internalStateInference": False, "basis": "explicit-question-temporal-window", "questionBasis": detected_basis, "faces": faces})
    return reactions


def finalize_expressions(evidence, phrases=(), duration_ms=None):
    """Derive durable source-clock layers from raw worker measurements without altering speaker data."""
    raw = evidence.get("facialExpressions") or {}
    duration = duration_ms or (evidence.get("editorTimeline") or {}).get("durationMs") or 0
    samples = valid_samples(raw.get("samples", []), duration)
    cuts = [row.get("startMs") for row in (evidence.get("editorTimeline") or {}).get("shots", [])
            if isinstance(row, dict)]
    cuts.extend((evidence.get("faceTracking") or {}).get("identityCutTimesMs", []))
    samples, static_evidence = annotate_static_faces(samples, duration, cuts)
    by_face, lookup, step = visibility(evidence, duration)
    intervals = expression_intervals(samples, evidence, duration, lookup, step)
    eligible_intervals = [row for row in intervals if row["reactionEligible"]]
    statuses = raw.get("windowStatuses") or [raw.get("status", "unavailable")]
    windows = (evidence.get("activityCoverage") or {}).get("windows") or []
    incomplete = any(row.get("status") != "ok" for row in windows)
    if statuses and all(item == "ok" for item in statuses) and not incomplete:
        status = "ok"
    elif samples:
        status = "partial"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "unavailable"
    evidence["facialExpressions"] = {"version": "facial-expressions-1", "status": status,
        "basis": "visible-expression-classifier", "model": MODEL_ID, "modelSha256": MODEL_SHA256,
        "sampleStepMs": SAMPLE_STEP_MS, "confidenceThreshold": THRESHOLD, "labels": list(LABELS),
        "probabilityBasis": "softmax-logits-not-calibrated",
        "evidenceStatus": "model-candidate", "verificationStatus": "unverified",
        "internalStateInference": False, "staticFaceEvidence": static_evidence,
        "interpretationRisk": "Classifier confidence is not calibrated accuracy, a verified visible reaction, or a probability of internal emotion.",
        "temporalSupport": "forward-sample-cell-clipped-to-continuous-visibility-and-cuts",
        "samples": samples, "intervals": intervals, "statistics": summarize(intervals, by_face, duration),
        "reactionStatistics": summarize(eligible_intervals, by_face, duration),
        "reactions": question_reactions(phrases, eligible_intervals, by_face, duration)}
    return evidence["facialExpressions"]
