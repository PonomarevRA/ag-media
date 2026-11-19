"""Link non-overlapping object track groups using bounded comparisons of measured views."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from colorsys import rgb_to_hsv

from voice_studio.graph.media_graph_objects import overlap

MAX_REAPPEARANCE_GAP_MS = 30_000
MAX_CANDIDATES = 16
MAX_VIEWS = 5
NEARBY_CANDIDATE_MS = 2200


def appearance_signature(node):
    """Read a distinctive dominant color from a measured crop, excluding neutral backgrounds."""
    palette = node["data"].get("objectPalette")
    if not isinstance(palette, list) or not palette or not isinstance(palette[0], dict):
        return None
    color = palette[0].get("hex")
    share = palette[0].get("share")
    if not isinstance(color, str) or len(color) != 7 or not isinstance(share, (int, float)) or share < .25:
        return None
    try:
        rgb = [int(color[pos:pos + 2], 16) / 255 for pos in (1, 3, 5)]
    except ValueError:
        return None
    hue, saturation, value = rgb_to_hsv(*rgb)
    if saturation < .32 or value < .22:
        return None
    return node["startMs"], node["data"]["box"], hue, saturation, value


def group_views(group, samples):
    """Keep time-spread views per detector label, one view per measured timestamp."""
    by_label = defaultdict(dict)
    for member in group[1]:
        for node in samples[member["id"]]:
            if hit := appearance_signature(node):
                by_label[node["label"].casefold()].setdefault(hit[0], hit)
    result = {}
    for label, views in by_label.items():
        ordered = sorted(views.values())
        count = min(MAX_VIEWS, len(ordered))
        result[label] = [ordered[round(index * (len(ordered) - 1) / max(1, count - 1))]
                         for index in range(count)]
    return result


def view_similarity(left, right):
    """Require compatible foreground colors, same screen region and similar scale."""
    _, first, a_hue, a_sat, a_value = left
    _, second, b_hue, b_sat, b_value = right
    iou = overlap(first, second)
    hue = abs(a_hue - b_hue)
    hue = min(hue, 1 - hue)
    saturation = abs(a_sat - b_sat)
    value = abs(a_value - b_value)
    if iou < .32 or hue > .04 or saturation > .14 or value > .12:
        return None
    return iou * .45 + (1 - hue / .04) * .15 + (1 - saturation / .14) * .2 + (1 - value / .12) * .2


def witness_score(left, right, *, across_cut):
    """Require repeated support across cuts and reject weak single-frame matches."""
    strength = 0
    for label in left.keys() & right.keys():
        paired = []
        for a in left[label]:
            for b in right[label]:
                score = view_similarity(a, b)
                if score is not None and score >= .46:
                    paired.append((score, a[0], b[0]))
        if not paired:
            continue
        paired.sort(reverse=True)
        best, first_time, second_time = paired[0]
        repeated = next((score for score, a, b in paired if a != first_time and b != second_time), None)
        if repeated is not None:
            strength = max(strength, (best + repeated) / 2)
        elif not across_cut and best >= .67:
            strength = max(strength, best - .05)
    return strength


def preferred(choices):
    """Resolve nearby alternatives before jumping over a plausible appearance."""
    if not choices:
        return None
    nearest = min(choices)[0]
    ranked = sorted(((score, index) for gap, score, index in choices
                     if gap <= nearest + NEARBY_CANDIDATE_MS), reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < .055):
        return None
    return ranked[0][1]


def reappearance_chains(groups, samples):
    """Return bounded, mutually preferred links between disjoint time intervals."""
    intervals = [(min(node["startMs"] for node in members), max(node["endMs"] for node in members))
                 for _, members, _ in groups]
    views = [group_views(group, samples) for group in groups]
    index_by_label = defaultdict(list)
    for index, (start, _) in enumerate(intervals):
        for label in views[index]:
            index_by_label[label].append((start, index))
    for rows in index_by_label.values():
        rows.sort()

    outgoing, incoming = defaultdict(list), defaultdict(list)
    for index, (_, end) in enumerate(intervals):
        candidates = set()
        for label in views[index]:
            rows = index_by_label[label]
            start = bisect_left(rows, (end, -1))
            stop = bisect_right(rows, (end + MAX_REAPPEARANCE_GAP_MS, len(groups)))
            candidates.update(other for _, other in rows[start:min(stop, start + MAX_CANDIDATES)])
        for other in candidates:
            gap = intervals[other][0] - end
            score = witness_score(views[index], views[other], across_cut=groups[index][0] != groups[other][0])
            if score:
                preference = score - min(.32, gap / 22_000)
                outgoing[index].append((gap, preference, other))
                incoming[other].append((gap, preference, index))
    links = {}
    for index, choices in outgoing.items():
        other = preferred(choices)
        if other is not None and preferred(incoming[other]) == index:
            links[index] = other
    return links
