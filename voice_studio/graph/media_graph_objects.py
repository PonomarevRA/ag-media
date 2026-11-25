"""Track sampled object boxes within cuts and describe their relation to stable speakers."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import hypot, isfinite

from voice_studio.graph.object_reconciliation import color_conflict
from voice_studio.vision.frame_changes import BASIS as FRAME_CHANGE_DETAIL
from voice_studio.vision.scene_sampling import SMALL_OBJECT_DETAIL
from voice_studio.vision.vehicle_evidence import VEHICLE_LABELS

MAX_GAP_MS = 1800
MIN_IOU = .44
MAX_APPEARANCE_GAP_MS = 1400
DISPLAY_LABELS = frozenset({"tv", "tv screen", "monitor", "screen"})


def overlap(a, b):
    """Measure normalized intersection over union for two source-frame boxes."""
    x, y, w, h = a
    u, v, p, q = b
    intersection = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    union = w * h + p * q - intersection
    return intersection / union if union > 0 else 0.0



def box_coverage(inner, outer):
    """Return the fraction of one normalized box covered by another."""
    x, y, w, h = inner
    u, v, p, q = outer
    common = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    return common / (w * h) if w > 0 and h > 0 else 0


def centered_display_box(first, second):
    """Match a cropped detection of the same display by containment and center alignment."""
    inner, outer = sorted((first, second), key=lambda box: box[2] * box[3])
    x, y = center(inner)
    u, v = center(outer)
    return box_coverage(inner, outer) >= .9 and abs(x - u) <= .22 * outer[2] and abs(y - v) <= .25 * outer[3]


def canonical_displays(observations):
    """Keep one outer display for aligned nested estimates with any detector scores."""
    ordered = sorted(observations, key=lambda node: -(node["data"]["box"][2] * node["data"]["box"][3]))
    canonical, fragments = [], defaultdict(list)
    for item in ordered:
        owners = [other for other in canonical if centered_display_box(item["data"]["box"], other["data"]["box"])]
        if len(owners) == 1:
            fragments[owners[0]["id"]].append(item)
        else:
            canonical.append(item)
    return canonical, fragments


def nested_fragment(inner, outer):
    """Identify a weaker centered proposal for a part of a same-class object."""
    a, b = inner["data"], outer["data"]
    if type(a.get("score")) not in (int, float) or type(b.get("score")) not in (int, float):
        return False
    if b["score"] - a["score"] < .10:
        return False
    first, second = a["box"], b["box"]
    w, h = first[2:]
    wide, tall = second[2:]
    if not (.35 <= w / wide <= .90 and .35 <= h / tall <= .90):
        return False
    ratio = w * h / (wide * tall)
    iou = overlap(first, second)
    coverage = iou * (1 + ratio) / ((1 + iou) * ratio)
    x, y = center(first)
    u, v = center(second)
    return coverage >= .90 and abs(x - u) <= .22 * wide and abs(y - v) <= .22 * tall


def canonical_observations(observations):
    """Keep separate objects when a proposal has several plausible containing owners."""
    owners = {}
    for node in observations:
        matches = [other["id"] for other in observations if other is not node and nested_fragment(node, other)]
        if len(matches) == 1:
            owners[node["id"]] = matches[0]
    fragments = defaultdict(list)
    canonical = []
    for node in observations:
        owner = owners.get(node["id"])
        if owner is not None and owner not in owners:
            fragments[owner].append(node)
        else:
            canonical.append(node)
    return canonical, fragments


def crosses_panel(checks, first, second, *boxes):
    """Break display continuity at a verified panel covering the same source pixels."""
    if not checks:
        return False
    start, end = sorted((first, second))
    left = bisect_right(checks, start, key=lambda check: check[0])
    right = bisect_right(checks, end, key=lambda check: check[0])
    return any(box_coverage(box, panel) >= .9 for _, panel in checks[left:right] for box in boxes)


def measured_motion_match(previous, current, overlap_score, *, with_anchor=False):
    """Recover a modest box shift using an exact cleaned foreground comparison.

    Source times, label and box must identify the preceding track observation.
    The worker measures these normalized crops at gaps up to 1400 ms. Background
    similarity alone cannot extend tracking, and reciprocal uniqueness is still
    decided by split_tracks over all geometric and appearance-backed candidates.
    A short pair of change-crop observations can tolerate no overlap with stronger
    foreground and edge evidence. Cross-pass matching needs an explicit anchor comparison.
    """
    gap = current["startMs"] - previous["startMs"]
    changes = sum(item["data"].get("detailBasis") == FRAME_CHANGE_DETAIL for item in (previous, current))
    rapid = changes >= (1 if with_anchor else 2) and 0 < gap <= 160
    if not 0 < gap <= MAX_APPEARANCE_GAP_MS or (overlap_score < .35 and not rapid):
        return False
    label = previous["label"].casefold()
    if label != current["label"].casefold():
        return False
    box = tuple(previous["data"]["box"])
    for match in current["data"].get("objectMatches") or ():
        if with_anchor and (not rapid or match.get("comparisonBasis") != "frame-change-anchor"):
            continue
        if (match["timeMs"] != previous["startMs"] or match["label"].casefold() != label
                or tuple(match["box"]) != box):
            continue
        inside, edges, scale = (match.get(key) for key in
                               ("objectSimilarity", "edgeSimilarity", "scaleRatio"))
        if any(type(value) not in (int, float) or not isfinite(value) for value in (inside, edges, scale)):
            continue
        if rapid and overlap_score < .35:
            if .97 <= inside <= 1 and .9 <= edges <= 1 and .8 <= scale <= 1.25:
                return True
        elif .94 <= inside <= 1 and .75 <= edges <= 1 and .65 <= scale <= 1.54:
            return True
    return False


def predicted_overlap(track, node):
    """Bridge a short constant-velocity continuation through a gap in box overlap."""
    if len(track) < 2:
        return 0
    first, last = track[-2:]
    span = last["startMs"] - first["startMs"]
    gap = node["startMs"] - last["startMs"]
    if not 0 < span <= 700 or not 0 < gap <= min(700, 2 * span):
        return 0
    before, current = first["data"]["box"], last["data"]["box"]
    x, y = center(current)
    u, v = center(before)
    vx, vy = (x - u) * gap / span, (y - v) * gap / span
    if hypot(vx, vy) > .16:
        return 0
    shifted = [current[0] + vx, current[1] + vy, *current[2:]]
    return overlap(shifted, node["data"]["box"])


def merged_vehicle_boxes(frames):
    """Find one box temporarily covering two independently visible vehicles."""
    pairs = sorted(stamp for stamp, rows in frames.items() if len(rows) == 2
                   and overlap(rows[0]["data"]["box"], rows[1]["data"]["box"]) < .35)
    merged = set()
    for stamp, rows in frames.items():
        if len(rows) != 1:
            continue
        next_pair = bisect_left(pairs, stamp)
        if not 0 < next_pair < len(pairs):
            continue
        before, after = pairs[next_pair - 1], pairs[next_pair]
        if max(stamp - before, after - stamp) > 800:
            continue
        box = rows[0]["data"]["box"]
        if all(box_coverage(node["data"]["box"], box) >= .68
               for pair in (before, after) for node in frames[pair]):
            merged.add(stamp)
    return merged


def split_tracks(frames, *, displays=False, panel_checks=(), vehicle_review=False):
    """Join mutually unique same-class geometry or measured-appearance candidates."""
    active, completed = [], []
    merged = merged_vehicle_boxes(frames) if vehicle_review else set()
    for stamp, observations in sorted(frames.items()):
        remaining = []
        for track in active:
            last = track[-1]
            if (stamp - last["startMs"] <= MAX_GAP_MS
                    and not crosses_panel(panel_checks, last["startMs"], stamp, last["data"]["box"])):
                remaining.append(track)
            else:
                completed.append(track)
        active = remaining
        if stamp in merged:
            observations[0]["data"]["vehicleIdentityAmbiguous"] = True
            completed.append(observations)
            continue
        incoming, outgoing = defaultdict(list), defaultdict(list)
        for i, track in enumerate(active):
            last = track[-1]
            previous = last["data"]["box"]
            for j, node in enumerate(observations):
                bounds = node["data"]["box"]
                overlap_score = overlap(previous, bounds)
                matches = overlap_score >= MIN_IOU
                if not matches and displays:
                    matches = centered_display_box(previous, bounds)
                if not matches:
                    matches = measured_motion_match(last, node, overlap_score)
                if not matches and not displays:
                    matches = predicted_overlap(track, node) >= .65 and not color_conflict([last], [node])
                if matches and not crosses_panel(panel_checks, last["startMs"], stamp, bounds):
                    incoming[j].append(i)
                    outgoing[i].append(j)
        for j, node in enumerate(observations):
            choices = incoming[j]
            if len(choices) == 1 and len(outgoing[choices[0]]) == 1:
                active[choices[0]].append(node)
            else:
                active.append([node])
    return completed + active


def append_detail_tracks(baseline, detail, *, panel_checks=()):
    """Attach a detail track only when one measured baseline track supports it unambiguously."""
    if not detail:
        return [(track[0], track) for track in baseline]
    if not baseline:
        return [(track[0], track) for track in detail]
    groups = [(track[0], track.copy()) for track in baseline]
    stamps_by_track = [{item["startMs"] for item in track} for track in baseline]
    nodes_by_stamp = defaultdict(list)
    for index, track in enumerate(baseline):
        for item in track:
            nodes_by_stamp[item["startMs"]].append((index, item))
    stamps = sorted(nodes_by_stamp)
    for track in detail:
        candidates = set()
        for sample in track:
            stamp = sample["startMs"]
            start = bisect_left(stamps, stamp - MAX_GAP_MS)
            stop = bisect_right(stamps, stamp + MAX_GAP_MS)
            for nearby_stamp in stamps[start:stop]:
                for owner, node in nodes_by_stamp[nearby_stamp]:
                    box = node["data"]["box"]
                    score = overlap(box, sample["data"]["box"])
                    before, after = (node, sample) if nearby_stamp <= stamp else (sample, node)
                    if ((score >= MIN_IOU or measured_motion_match(before, after, score, with_anchor=True))
                            and not crosses_panel(panel_checks, nearby_stamp, stamp, box, sample["data"]["box"])):
                        candidates.add(owner)
            if len(candidates) > 1:
                break
        if len(candidates) == 1:
            owner = next(iter(candidates))
            detail_stamps = {item["startMs"] for item in track}
            if stamps_by_track[owner].isdisjoint(detail_stamps):
                groups[owner][1].extend(track)
                stamps_by_track[owner].update(detail_stamps)
                continue
        groups.append((track[0], track))
    for _, track in groups:
        track.sort(key=lambda item: item["startMs"])
    return groups


def confirmed_tracks(frames, fragments, *, displays=False, panel_checks=(), vehicle_review=False):
    """Preserve existing fragment and association rules within each detection pass."""
    canonicalize = canonical_displays if displays else canonical_observations
    canonical_frames = {}
    for stamp, observations in frames.items():
        canonical_frames[stamp], aliases = canonicalize(observations)
        fragments.update(aliases)
    return split_tracks(canonical_frames, displays=displays, panel_checks=panel_checks,
                        vehicle_review=vehicle_review)


def add_object_tracks(graph, *, object_reconciliation=False):
    """Link object samples to a concrete track, preserving separations at cuts."""
    grouped = defaultdict(lambda: (defaultdict(list), defaultdict(list)))
    panels = defaultdict(list)
    for node in graph.nodes.values():
        if node["kind"] != "object_observation":
            continue
        stamp = node["startMs"]
        shot = next(graph.intersecting_shots(stamp, stamp + 1), None)
        if shot:
            if node["data"].get("source") == "interface-layout" and node["label"] in ("mobile-ui", "ui"):
                panels[shot].append((stamp, node["data"]["box"]))
            baseline, detail = grouped[shot, node["label"].casefold()]
            frames = detail if node["data"].get("detailBasis") in (SMALL_OBJECT_DETAIL, FRAME_CHANGE_DETAIL) else baseline
            frames[stamp].append(node)
    for checks in panels.values():
        checks.sort(key=lambda check: check[0])
    tracks = []
    for (shot, label), (base_frames, detail_frames) in grouped.items():
        fragments = {}
        displays = label in DISPLAY_LABELS
        checks = panels.get(shot, ()) if displays else ()
        baseline = confirmed_tracks(base_frames, fragments, displays=displays, panel_checks=checks,
                                    vehicle_review=object_reconciliation and label in VEHICLE_LABELS)
        detail = confirmed_tracks(detail_frames, fragments, displays=displays, panel_checks=checks)
        for anchor, observations in append_detail_tracks(baseline, detail, panel_checks=checks):
            first, last = observations[0], observations[-1]
            # Earlier detail extends coverage without changing the baseline track ID.
            key = "object_track:" + anchor["id"].split(":", 1)[1]
            label = anchor["label"]
            graph.node(key, "object_track", label, (first["startMs"], last["endMs"]),
                       {"sampleCount": len(observations), "basis": "box-iou-within-shot",
                        "coordinateSpace": "normalized-oriented-source", "sampledGeometry": True,
                        "representative": representative_sample(observations)})
            graph.connect(key, "in_shot", shot)
            for node in observations:
                graph.connect(node["id"], "sample_of", key)
                for fragment in fragments.get(node["id"], ()):
                    graph.connect(fragment["id"], "part_of_observation", node["id"])
            tracks.append((key, shot, observations))
    return tracks


def representative_sample(observations):
    """Pair the strongest valid detector score with its exact frame and box."""
    best, best_score = observations[0], None
    for observation in observations:
        score = observation["data"].get("score")
        if type(score) in (int, float) and 0 <= score <= 1 and (best_score is None or score > best_score):
            best, best_score = observation, score
    data = best["data"]
    sample = {"observationId": best["id"], "timeMs": best["startMs"], "box": data["box"], "score": best_score}
    if palette := data.get("objectPalette"):
        sample["palette"] = palette
    if screen := data.get("interfaceEvidence"):
        sample["interfaceEvidence"] = screen
    return sample


def center(box):
    """Return the center of a normalized frame rectangle."""
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def nearby(box, faces):
    """Identify a nearby face only when geometry separates it from other faces."""
    x, y = center(box)
    ranked = []
    for face in faces:
        u, v = center(face["box"])
        ranked.append((hypot(x - u, y - v), face["track"]))
    ranked.sort()
    if not ranked or ranked[0][0] > .32:
        return None
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .10:
        return None
    return ranked[0][1]


def add_speaker_context(graph, tracks, samples, step):
    """Add bounded, shot-local context after face-to-speaker bindings are established."""
    if not step:
        return
    frames = dict(samples)
    profiles = defaultdict(list)
    for key, shot, observations in tracks:
        if len(observations) < 2:
            continue
        first_box = observations[0]["data"]["box"]
        paired = defaultdict(list)
        for node in observations:
            box = node["data"]["box"]
            if overlap(first_box, box) < .55:
                continue
            stamp = node["startMs"]
            faces = frames.get(stamp - stamp % step, [])
            nearest = nearby(box, faces)
            for face in faces:
                if face.get("speakerId"):
                    paired[face["speakerId"], face["track"]].append((face["box"], face["track"] == nearest))
        for (speaker, _), rows in paired.items():
            reference = rows[0][0]
            stable = [is_near for box, is_near in rows if overlap(reference, box) >= .52]
            if len(stable) >= 2:
                near = sum(stable) >= 2
                profiles[speaker].append((.08 if near else .02, key, shot, len(stable)))
    for speaker, entries in profiles.items():
        context = []
        seen = set()
        for weight, key, shot, count in sorted(entries, reverse=True):
            if key in seen:
                continue
            seen.add(key)
            context.append({"objectTrackId": key, "shotId": shot, "samples": count,
                            "relation": "near" if weight == .08 else "background", "weight": weight})
            graph.connect(speaker, "shot_context", key)
            if len(context) >= 8:
                break
        graph.nodes[speaker]["data"]["shotContext"] = context
