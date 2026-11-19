"""Optional conservative identity reconciliation from measured source-crop parts.

This is a 2D gallery of observations, not an inferred object surface or 3D model.
Raw observations and track identifiers are never changed by this pure module.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from colorsys import rgb_to_hsv
from math import isfinite

MAX_GAP_MS = 30_000
MAX_CANDIDATES = 16
MAX_GALLERY = 8
MAX_CLUSTER = 16


def observation_key(node):
    """Identify an exact measured source crop independently of detector ordering."""
    return node['startMs'], node['label'], tuple(node['data']['box'])


def gallery(group, samples, preferred_ids=()):
    """Select bounded, time-spread real observations, prioritizing measured witnesses."""
    rows = [node for member in group[1] for node in samples[member['id']]]
    rows.sort(key=lambda node: (node['startMs'], node['id']))
    if len(rows) <= MAX_GALLERY:
        return rows
    preferred = set(preferred_ids)
    selected = [node for node in rows if node["id"] in preferred][:MAX_GALLERY]
    for node in rows:
        if len(selected) >= MAX_GALLERY // 2:
            break
        if node['data'].get('objectPartMatches') and node not in selected:
            selected.append(node)
    for index in range(MAX_GALLERY):
        node = rows[round(index * (len(rows) - 1) / (MAX_GALLERY - 1))]
        if node not in selected and len(selected) < MAX_GALLERY:
            selected.append(node)
    return sorted(selected, key=lambda node: (node['startMs'], node['id']))


def dominant_color(node):
    """Read reliable dominant crop color; missing or mixed palettes stay unknown."""
    palette = node['data'].get('objectPalette') or []
    if not isinstance(palette, list) or not palette or not isinstance(palette[0], dict):
        return None
    color, share = palette[0].get('hex'), palette[0].get('share')
    if not isinstance(color, str) or len(color) != 7 or not color.startswith('#') or not valid_number(share, .5, 1):
        return None
    try:
        rgb = [int(color[pos:pos + 2], 16) / 255 for pos in (1, 3, 5)]
    except ValueError:
        return None
    return rgb_to_hsv(*rgb)


def color_conflict(left, right):
    """Veto clear chromatic/white or different-hue identities, never merge by color."""
    for a in left:
        first = dominant_color(a)
        if first is None:
            continue
        for b in right:
            second = dominant_color(b)
            if second is None:
                continue
            hue, saturation, value = first
            other_hue, other_sat, other_value = second
            if ((saturation >= .45 and value >= .25 and other_sat <= .16 and other_value >= .70)
                    or (other_sat >= .45 and other_value >= .25 and saturation <= .16 and value >= .70)):
                return True
            distance = min(abs(hue - other_hue), 1 - abs(hue - other_hue))
            if min(saturation, other_sat) >= .45 and min(value, other_value) >= .25 and distance > .12:
                return True
    return False


def valid_number(value, lower, upper):
    """Reject booleans, non-finite measurements and out-of-domain evidence."""
    return type(value) in (int, float) and isfinite(value) and lower <= value <= upper


def part_score(match):
    """Accept distributed, geometrically verified local-feature correspondences only."""
    if match.get('basis') != 'orb-local-parts':
        return 0
    limits = {'inliers': (8, 512), 'inlierFraction': (.55, 1),
              'sourceCoverage': (.12, 1), 'targetCoverage': (.12, 1),
              'sourceCells': (3, 9), 'targetCells': (3, 9)}
    if any(not valid_number(match.get(key), *bounds) for key, bounds in limits.items()):
        return 0
    return min(1, match['inliers'] / 24) * .4 + match['inlierFraction'] * .6


def source_match_key(match):
    """Validate exact-source references before hashing optional imported witness data."""
    if not isinstance(match, dict):
        return None
    stamp, label, box = match.get('timeMs'), match.get('label'), match.get('box')
    if type(stamp) is not int or stamp < 0 or not isinstance(label, str):
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(valid_number(v, 0, 1) for v in box):
        return None
    return stamp, label, tuple(box)


def pair_evidence(left, right):
    """Resolve exact observed witnesses; class names and palettes cannot create a link."""
    if color_conflict(left, right):
        return None
    known = {observation_key(node): node for node in left}
    witnesses = []
    for node in right:
        raw = node['data'].get('objectPartMatches') or []
        for match in raw[:3] if isinstance(raw, (list, tuple)) else []:
            key = source_match_key(match)
            if key is None:
                continue
            previous = known.get(key)
            score = part_score(match)
            if previous is not None and previous['startMs'] < node['startMs'] and score:
                witnesses.append((score, previous['id'], node['id']))
    if not witnesses:
        return None
    witnesses.sort(reverse=True)
    score, source, target = witnesses[0]
    return {'score': score, 'fromObservationId': source, 'toObservationId': target,
            'basis': 'measured-local-parts'}


def preferred(choices):
    """Abstain when the best candidate lacks a clear measured-score margin."""
    ranked = sorted(choices, reverse=True)
    if not ranked or len(ranked) > 1 and ranked[0][0] - ranked[1][0] < .08:
        return None
    return ranked[0][1]


def reconciliation_links(groups, samples, baseline_links=None):
    """Return disjoint complete-evidence chains and bounded diagnostic counts.

    At most 16 future same-class groups within 30 seconds are scored per group.
    Overflow abstains rather than concealing an unexamined competing identity.
    Every pair in a new extension needs its own positive evidence; A-B and B-C
    cannot invent A-C identity. Simultaneous intervals are always distinct.
    """
    intervals = [(min(n['startMs'] for n in group[1]), max(n['endMs'] for n in group[1])) for group in groups]
    views = [gallery(group, samples) for group in groups]
    labels = [{n['label'].casefold() for n in group[1]} for group in groups]
    index = defaultdict(list)
    for key, (start, _) in enumerate(intervals):
        for label in labels[key]:
            index[label].append((start, key))
    for rows in index.values():
        rows.sort()
    baseline = {}
    rejected_baseline = []
    baseline_members = {key: [key] for key in range(len(groups))}
    ordered_baseline = sorted((baseline_links or {}).items(), key=lambda pair: (intervals[pair[0]], pair))
    for key, other in ordered_baseline:
        members = baseline_members[key]
        compatible = all(intervals[member][1] <= intervals[other][0]
                         and not color_conflict(views[member], views[other]) for member in members)
        if compatible and intervals[key][0] < intervals[other][0] and len(members) < MAX_CLUSTER:
            baseline[key] = other
            baseline_members[other] = members + [other]
        else:
            rejected_baseline.append({'fromGroup': key, 'toGroup': other, 'reason': 'time-or-color-conflict'})
    followers = set(baseline.values())
    chains = {}
    for head in range(len(groups)):
        if head in followers:
            continue
        members = [head]
        while members[-1] in baseline:
            members.append(baseline[members[-1]])
        chains[head] = members
    tails = {members[-1]: members for members in chains.values()}
    outgoing, incoming, evidence = defaultdict(list), defaultdict(list), {}
    diagnostics = {'basis': 'measured-local-parts-pilot', 'comparisons': 0, 'overflowGroups': 0,
                   'maxCandidates': MAX_CANDIDATES, 'maxGallery': MAX_GALLERY, 'acceptedLinks': [],
                   'preservedBaselineLinks': len(baseline), 'rejectedBaselineLinks': rejected_baseline}
    for key, (_, end) in enumerate(intervals):
        candidates = set()
        for label in labels[key]:
            rows = index[label]
            lo = bisect_left(rows, (end, -1))
            hi = bisect_right(rows, (end + MAX_GAP_MS, len(groups)))
            candidates.update(other for _, other in rows[lo:min(hi, lo + MAX_CANDIDATES + 1)] if other != key)
        if len(candidates) > MAX_CANDIDATES:
            diagnostics['overflowGroups'] += 1
            continue
        for other in sorted(candidates):
            if intervals[other][0] < end or intervals[other][0] <= intervals[key][0]:
                continue
            diagnostics['comparisons'] += 1
            witness = pair_evidence(views[key], views[other])
            if witness:
                evidence[key, other] = witness
    # Existing default chains are retained unless an explicit time/color conflict was measured.
    # A new extension needs direct part evidence for every original group on both sides.
    for tail, head in evidence:
        if tail not in tails or head not in chains:
            continue
        members, following = tails[tail], chains[head]
        pairs = [(left, right) for left in members for right in following]
        if all(pair in evidence for pair in pairs):
            score = min(evidence[pair]['score'] for pair in pairs)
            outgoing[tail].append((score, head))
            incoming[head].append((score, tail))
    links = dict(baseline)
    cluster = {key: list(tails.get(key, [key])) for key in range(len(groups))}
    ordered = sorted(range(len(groups)), key=lambda key: (intervals[key], key))
    for key in ordered:
        other = preferred(outgoing[key])
        if other is None or preferred(incoming[other]) != key:
            continue
        members = cluster[key]
        following = chains[other]
        pairs = [(left, right) for left in members for right in following]
        if len(members) + len(following) > MAX_CLUSTER or any(pair not in evidence for pair in pairs):
            continue
        links[key] = other
        cluster[following[-1]] = members + following
        diagnostics['acceptedLinks'].append({'fromGroup': key, 'toGroup': other, **evidence[key, other]})
    return links, diagnostics


def reconcile_object_groups(groups, samples, *, baseline_links=None):
    """Pure callable retaining original groups plus explicit bounded reconciliation links.

    Original group/track boundaries are retained for graph appearance intervals.
    Return (links, diagnostics); callers publish one profile per chain head.
    """
    return reconciliation_links(groups, samples, baseline_links)


def source_gallery(samples, preferred_ids=()):
    """Publish real source crop locators without synthesizing unobserved object views."""
    group = ('', [{'id': 'gallery'}], '')
    rows = gallery(group, {'gallery': samples}, preferred_ids)
    return {'basis': 'measured-source-crops-not-3d', 'views': [
        {'observationId': row['id'], 'timeMs': row['startMs'], 'label': row['label'],
         'box': list(row['data']['box']), 'palette': row['data'].get('objectPalette', [])} for row in rows]}
