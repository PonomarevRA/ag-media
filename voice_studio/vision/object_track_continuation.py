"""Continue confirmed tracks with a single nearby weak detector measurement."""

from collections import Counter, defaultdict
from time import perf_counter

import cv2

from voice_studio.vision.object_tracking_camera import (
    dense_tracking_transform, sparse_tracking_transform, tracking_background,
)
from voice_studio.vision.object_tracking_contract import (
    MAX_TRACKING_GAP_MS, TRACKING_EXCLUDED_LABELS, TRACKING_PROPOSAL_SCORE,
    STRONG_TRACKING_SCORE, WEAK_CONTINUATION,
)
from voice_studio.vision.scene_objects import iou
from voice_studio.vision.vehicle_evidence import valid_box

MAX_OBSERVATIONS = 12
MAX_CAMERA_PAIRS = 36
MAX_DENSE_PAIRS = 12
MIN_DIRECT_IOU = .60
MIN_MOVED_IOU = .58
AMBIGUOUS_IOU = .34


def move_box(box, motion):
    """Project a source rectangle using the camera transform around the frame center."""
    if not motion:
        return box
    x, y, w, h = box
    scale = motion["scale"]
    return [.5 + (x - .5) * scale + motion["shiftX"], .5 + (y - .5) * scale + motion["shiftY"],
            w * scale, h * scale]


def usable(item):
    """Allow native physical object detections with normalized box geometry."""
    label, score = item.get("label"), item.get("score")
    return (isinstance(label, str) and label.casefold() not in TRACKING_EXCLUDED_LABELS
            and type(score) in (int, float) and TRACKING_PROPOSAL_SCORE <= score <= 1
            and valid_box(item.get("box")))


def compatible(anchor, candidate):
    """Require the original classes and plausible box size and aspect ratio."""
    if anchor["label"].casefold() != candidate["label"].casefold():
        return False
    w, h = anchor["box"][2:]
    u, v = candidate["box"][2:]
    return .62 <= u / w <= 1.62 and .62 <= v / h <= 1.62


def near_other(box, detections, threshold):
    """Treat competing measurements in the same pixels as occupied."""
    return any(iou(box, row["box"]) >= threshold for row in detections)


def weak_candidates(raw, confirmed):
    """Retain only originally rejected weak measurements outside occupied boxes."""
    return [item for item in raw[:20] if usable(item) and item["score"] < STRONG_TRACKING_SCORE
            and not near_other(item["box"], confirmed, .60)][:MAX_OBSERVATIONS]


def plausible_pair(anchor, item):
    """Skip costly motion fitting for unrelated labels or distant geometry."""
    if not compatible(anchor, item):
        return False
    x, y, w, h = anchor["box"]
    u, v, s, t = item["box"]
    return abs(x + w / 2 - u - s / 2) <= .21 and abs(y + h / 2 - v - t / 2) <= .21


def match_weak(anchors, candidates, occupied, stats, motion=None):
    """Choose reciprocal pairs while vetoing competition from any eligible anchor."""
    weights = defaultdict(list)
    reverse = Counter()
    minimum = MIN_MOVED_IOU if motion else MIN_DIRECT_IOU
    for i, anchor in enumerate(anchors):
        expected = move_box(anchor["box"], motion)
        anchor_occupied = near_other(expected, occupied, AMBIGUOUS_IOU)
        for j, item in enumerate(candidates):
            if not compatible(anchor, item):
                continue
            score = iou(expected, item["box"])
            if score >= AMBIGUOUS_IOU:
                # An occupied track can veto a weak candidate but cannot consume it.
                reverse[j] += 1
                if not anchor_occupied:
                    weights[i].append((j, score))
    matches = []
    for i, choices in weights.items():
        if len(choices) != 1 or reverse[choices[0][0]] != 1:
            stats["ambiguousCandidates"] += len(choices)
            continue
        j, score = choices[0]
        if score >= minimum:
            matches.append((i, j))
    return matches


def read_background(path, items):
    """Decode one reduced source sample for correspondence outside detector boxes."""
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    return tracking_background(image, [item["box"] for item in items if valid_box(item.get("box"))])


def camera_for_pair(previous, current, stats):
    """Bound expensive work and use DIS only when sparse background fitting abstains."""
    if stats["cameraPairs"] >= MAX_CAMERA_PAIRS:
        return None
    stats["cameraPairs"] += 1
    try:
        first, second = (read_background(path, raw) for path, raw in (previous, current))
    except cv2.error:
        return None
    if first is None or second is None:
        return None
    if motion := sparse_tracking_transform(first, second):
        return motion
    if stats["densePairs"] >= MAX_DENSE_PAIRS:
        return None
    stats["densePairs"] += 1
    return dense_tracking_transform(first, second)


def add_candidates(frame, anchor_frame, anchors, candidates, matches, stats, basis):
    """Append weak samples while retaining their exact source evidence and raw label."""
    for i, j in matches:
        anchor, item = anchors[i], candidates[j]
        frame["objects"].append({
            "label": item["label"], "box": [round(value, 4) for value in item["box"]], "score": item["score"],
            "basis": "weak-detector-proposal", "detailBasis": WEAK_CONTINUATION,
            "trackingWitness": {"timeMs": anchor_frame["timeMs"], "box": list(anchor["box"]), "basis": basis},
        })
        stats["continued"] += 1
        if basis != "box-iou":
            stats["cameraContinued"] += 1


def continue_object_tracks(frames, raw_samples):
    """Append only same-shot continuations of already confirmed detector observations.

    Raw samples contain the existing extracted image path and pre-confirmation
    detector candidates at that same timestamp. This pass never starts a track
    or changes any confirmed observation. Weak outputs are excluded as anchors.
    """
    started = perf_counter()
    stats = {"basis": WEAK_CONTINUATION, "continued": 0, "cameraContinued": 0, "weakCandidates": 0,
             "cameraPairs": 0, "densePairs": 0, "ambiguousCandidates": 0}
    previous = None
    for frame in frames:
        stamp = frame["timeMs"]
        if stamp not in raw_samples:
            continue
        occupied = [row for row in frame["objects"] if usable(row) and "detailBasis" not in row]
        if previous:
            anchor_frame, anchors = previous
            age = stamp - anchor_frame["timeMs"]
            if frame["shot"] == anchor_frame["shot"] and 0 < age <= MAX_TRACKING_GAP_MS and anchors:
                cont = weak_candidates(raw_samples[stamp][1], occupied)
                stats["weakCandidates"] += len(cont)
                matches = match_weak(anchors, cont, occupied, stats)
                add_candidates(frame, anchor_frame, anchors, cont, matches, stats, "box-iou")
                if not matches and any(plausible_pair(anchor, item) for anchor in anchors for item in cont):
                    motion = camera_for_pair(raw_samples[anchor_frame["timeMs"]], raw_samples[stamp], stats)
                    if motion:
                        matches = match_weak(anchors, cont, occupied, stats, motion)
                        add_candidates(frame, anchor_frame, anchors, cont, matches, stats, motion["basis"])
        previous = (frame, occupied[:MAX_OBSERVATIONS])
    stats["elapsedMs"] = round((perf_counter() - started) * 1000)
    return stats
