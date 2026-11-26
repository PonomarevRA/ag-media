"""Use confirmed speaking intervals to avoid sampling unrelated audio in long cues."""

from collections import Counter, defaultdict
from itertools import combinations, islice

from voice_studio.vision.face_speakers import co_visible_pairs
from voice_studio.vision.static_face_evidence import strong_av_anchor_eligible

MIN_LOCAL_MS = 480
MAX_NATIVE_REVIEW_PAIRS = 3
LOCAL_BASIS = "light-asd-local"


def local_anchors(evidence, turns):
    """Select one face per cue and intersect its strongest speech run with a native turn."""
    if not isinstance(evidence, dict):
        return {}
    expressions = evidence.get("facialExpressions") or {}
    exclusions = (expressions.get("staticFaceEvidence") or {}).get("exclusions", ())
    candidates = defaultdict(list)
    turns = [row for row in turns if isinstance(row, dict)
             and all(type(row.get(key)) is int for key in ("startMs", "endMs", "speaker"))
             and 0 <= row["startMs"] < row["endMs"] and row["speaker"] > 0]
    for row in evidence.get("visibleSpeechIntervals") or ():
        if not isinstance(row, dict) or row.get("basis") != LOCAL_BASIS:
            continue
        if any(type(row.get(key)) is not int for key in ("startMs", "endMs", "cueStartMs", "track", "faceId")):
            continue
        left, right, face = row["startMs"], row["endMs"], row["faceId"]
        if not 0 <= row["cueStartMs"] <= left < right or min(face, row["track"]) < 1:
            continue
        activity = (evidence.get("phrases") or {}).get(str(row["cueStartMs"]), {}).get("activity", {})
        if activity.get("state") == "visible-active":
            continue
        hits = [turn for turn in turns if turn["startMs"] < right and left < turn["endMs"]]
        if len({turn["speaker"] for turn in hits}) != 1:
            continue
        for turn in hits:
            start, end = max(left, turn["startMs"]), min(right, turn["endMs"])
            if end - start < MIN_LOCAL_MS:
                continue
            times = [start, *(max(start, span["startMs"]) for span in exclusions
                             if span["faceId"] == face and span["startMs"] < end and start < span["endMs"])]
            if all(strong_av_anchor_eligible(expressions, face, stamp) for stamp in times):
                candidates[row["cueStartMs"]].append({**row, "startMs": start, "endMs": end,
                                                      "nativeSpeaker": turn["speaker"]})
    return {str(stamp): max(rows, key=anchor_duration) for stamp, rows in candidates.items()
            if len({(row["faceId"], row["nativeSpeaker"]) for row in rows}) == 1}


def anchor_duration(row):
    """Measure the interval with both visual and audio activity."""
    return row["endMs"] - row["startMs"]


def review_pairs(anchors, evidence):
    """Review repeated native turns with different face IDs, subject to hard visual vetoes."""
    timeline = evidence.get("timeline") or {}
    frames, step = timeline.get("frames"), timeline.get("stepMs")
    if not frames or step != 40:
        return []
    by_voice = defaultdict(dict)
    for row in sorted(anchors.values(), key=anchor_duration):
        by_voice[row["nativeSpeaker"]][row["faceId"]] = row
    faces = {face for observations in by_voice.values() for face in observations}
    covisible = co_visible_pairs(timeline, faces)
    distinct = {tuple(sorted(pair)) for pair in (evidence.get("faceTracking") or {}).get("distinctFaces", ())}
    pairs = []
    for observations in by_voice.values():
        for a, b in islice(combinations(sorted(observations), 2), MAX_NATIVE_REVIEW_PAIRS * 4):
            if (a, b) in covisible or (a, b) in distinct:
                continue
            samples = [anchor_samples(observations[face], frames, step) for face in (a, b)]
            if all(samples):
                pairs.append({"cueMs": observations[b]["cueStartMs"], "face": a, "otherFace": b, "samples": samples})
            if len(pairs) >= MAX_NATIVE_REVIEW_PAIRS:
                return pairs
    return pairs


def anchor_samples(anchor, frames, step):
    """Take two separated original-frame crops inside a confirmed speech interval."""
    left, right = anchor["startMs"], anchor["endMs"]
    samples = []
    for fraction in (.25, .55):
        index = round((left + (right - left) * fraction) / step)
        if index >= len(frames):
            return []
        hit = next((hit for hit in frames[index]
                    if hit[0] == anchor["track"] and hit[6] == anchor["faceId"] and hit[5] == 1), None)
        if hit is None:
            return []
        samples.append({"timeMs": index * step, "box": hit[1:5]})
    return samples


def reconcile_anchors(anchors, pairs, matches):
    """Allow disjoint verified pairs only; never infer identities transitively."""
    measured = {(row["cueMs"], row["face"]): row["similarity"] for row in matches}
    verified = [pair for pair in pairs if (pair["cueMs"], pair["face"]) in measured]
    competition = Counter(face for pair in verified for face in (pair["face"], pair["otherFace"]))
    for pair in verified:
        first, second = pair["face"], pair["otherFace"]
        if competition[first] != 1 or competition[second] != 1:
            continue
        similarity = measured[pair["cueMs"], first]
        for row in anchors.values():
            if row["faceId"] in (first, second):
                row.update(profileFace=first, identitySimilarity=similarity)
    return anchors
