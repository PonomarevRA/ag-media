"""Use segmented clothing as corroboration for borderline within-video face matches."""

from collections import defaultdict
from itertools import combinations
from math import sqrt

from voice_studio.vision.face_reid import MIN_MARGIN, MIN_MATCH, MIN_SAMPLES, cosine, unit_vector

FACE_FLOOR = .48
CLOTHING_FLOOR = .88
SAMPLE_AGREEMENT = .82
MAX_GAP_FRAMES = 25 * 18
MAX_GROUPS = 12
MIN_TRACK_FRAMES = 9


def face_refs(tracks, signatures, aliases):
    """Build a stable face reference and its lifetime from established track groups."""
    grouped = defaultdict(list)
    for track_id, track in enumerate(tracks, 1):
        samples = signatures.get(track_id, ())
        if len(samples) >= MIN_SAMPLES:
            vector = unit_vector([sum(items) for items in zip(*samples)])
            if vector:
                grouped[aliases[track_id]].append((track_id, vector))
    windows = {}
    for track_id, track in enumerate(tracks, 1):
        face = aliases[track_id]
        start, end = track[0][0], track[-1][0]
        low, high = windows.get(face, (start, end))
        windows[face] = min(low, start), max(high, end)
    return grouped, windows


def borderline_pairs(tracks, signatures, aliases):
    """Propose mutually unique, nearby returns whose face score nearly passes on its own."""
    refs, windows = face_refs(tracks, signatures, aliases)
    ranked = defaultdict(list)
    for left, right in combinations(refs, 2):
        first, second = sorted((left, right), key=lambda face: windows[face][0])
        gap = windows[second][0] - windows[first][1]
        if not 0 < gap <= MAX_GAP_FRAMES:
            continue
        score = min(cosine(refs[first][0][1], refs[second][0][1]),
                    cosine(refs[first][-1][1], refs[second][0][1]))
        if score >= FACE_FLOOR - MIN_MARGIN:
            ranked[first].append((score, second))
            ranked[second].append((score, first))
    best = {}
    for face, matches in ranked.items():
        matches.sort(reverse=True)
        score, other = matches[0]
        if FACE_FLOOR <= score < MIN_MATCH and (len(matches) == 1 or score - matches[1][0] >= MIN_MARGIN):
            best[face] = score, other
    return [(face, other, score) for face, (score, other) in best.items()
            if windows[face][0] < windows[other][0] and best.get(other, (0, None))[1] == face]


def choose_sample_tracks(tracks, aliases, pairs, active_faces=()):
    """Prioritize borderline pairs, visible talkers, then long remaining tracks."""
    grouped = {}
    for track_id, track in enumerate(tracks, 1):
        face = aliases[track_id]
        previous = grouped.get(face)
        if len(track) >= MIN_TRACK_FRAMES and (previous is None or len(track) > len(tracks[previous - 1])):
            grouped[face] = track_id
    priority = []
    for left, right, _ in sorted(pairs, key=lambda pair: -pair[2]):
        if len(priority) >= MAX_GROUPS:
            break
        if left in grouped and right in grouped:
            priority.extend((left, right))
    for face in [*active_faces, *sorted(grouped, key=lambda value: -len(tracks[grouped[value] - 1]))]:
        if face in grouped and face not in priority and len(priority) < MAX_GROUPS:
            priority.append(face)
    return {face: grouped[face] for face in priority}


def similarity(left, right):
    """Bhattacharyya similarity across normalized clothing histograms."""
    return sum(sqrt(a * b) for a, b in zip(left, right))


def corroborate(pairs, samples, aliases):
    """Merge disjoint borderline groups only when both outfits have repeatable evidence."""
    refs = {}
    outfits = defaultdict(list)
    for sample in samples:
        outfits[sample["face"]].append(sample)
    for face, rows in outfits.items():
        if len(rows) == 2 and similarity(rows[0]["hist"], rows[1]["hist"]) >= SAMPLE_AGREEMENT:
            refs[face] = rows
    merges = {}
    notes = []
    for first, second, face_score in pairs:
        if first not in refs or second not in refs:
            continue
        measured = min(similarity(left["hist"], right["hist"]) for left in refs[first] for right in refs[second])
        if measured >= CLOTHING_FLOOR:
            merges[second] = first
            notes.append({"face": first, "joinedFace": second, "faceCosine": round(face_score, 3),
                          "clothingSimilarity": round(measured, 3), "basis": "face-and-clothing"})
    revised = {track: merges.get(face, face) for track, face in aliases.items()}
    descriptions, combined = {}, defaultdict(list)
    for face, rows in refs.items():
        combined[merges.get(face, face)].extend(rows)
    for face, rows in combined.items():
        description = {"swatch": rows[0]["swatch"], "source": "mediapipe-clothing",
                       "sampleTimesMs": [row["timeMs"] for row in rows]}
        pattern = rows[0].get("pattern")
        if pattern and all(row.get("pattern") == pattern for row in rows):
            description["pattern"] = pattern
        descriptions[face] = description
    return revised, notes, descriptions
