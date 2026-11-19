"""Bind the numbered face boxes to transcript speakers using aligned evidence."""

from collections import Counter, defaultdict
from itertools import combinations


def unique_bindings(proposals):
    """Accept only one-to-one bindings with no competing speaker or face."""
    owners = Counter(number for numbers in proposals.values() for number in numbers)
    return {face: next(iter(numbers)) for face, numbers in proposals.items()
            if len(numbers) == 1 and owners[next(iter(numbers))] == 1}


def dominant_bindings(proposals):
    """Select mutually dominant assignments from weighted, non-overlapping cue evidence."""
    by_number = defaultdict(Counter)
    for face, scores in proposals.items():
        for number, weight in scores.items():
            by_number[number][face] = weight

    def leader(scores):
        """Require both an absolute margin and a substantial lead over alternatives."""
        candidates = scores.most_common(2)
        candidate, score = candidates[0]
        other = candidates[1][1] if len(candidates) > 1 else 0
        return candidate if score >= 480 and score >= 1.6 * other and score - other >= 380 else None

    assigned = {}
    for face, scores in proposals.items():
        number = leader(scores)
        if number is not None and leader(by_number[number]) == face:
            assigned[face] = number
    return assigned


def face_at(row):
    """Return a stable face ID when available; older timelines carry only a track ID."""
    return row[6] if len(row) > 6 else row[0]


def co_visible_pairs(timeline, candidates=None):
    """Find faces occupying separate visible positions on repeated frames."""
    counts = Counter()
    for hits in timeline["frames"]:
        if len(hits) < 2:
            continue
        observed = set()
        for left, right in combinations(hits, 2):
            first, second = face_at(left), face_at(right)
            if first == second or (candidates is not None and not {first, second} <= candidates):
                continue
            x, y, w, h = left[1:5]
            a, b, c, d = right[1:5]
            if x + w <= a or a + c <= x or y + h <= b or b + d <= y:
                observed.add(tuple(sorted((first, second))))
        counts.update(observed)
    return {pair for pair, count in counts.items() if count >= 2}


def active_face(cue, timeline):
    """Find one dominant visible talker in a cue with no diarization overlap."""
    if cue.get("overlap"):
        return None
    if cue.get("visualFace") is not None:
        return cue["visualFace"]
    step = timeline["stepMs"]
    start = (cue["startMs"] + step - 1) // step
    end = (cue["endMs"] + step - 1) // step
    votes = Counter()
    for hits in timeline["frames"][start:end]:
        active = [face_at(hit) for hit in hits if hit[5] == 1]
        if len(active) == 1:
            votes[active[0]] += 1
    best = votes.most_common(2)
    if not best:
        return None
    face, frames = best[0]
    runner_up = best[1][1] if len(best) > 1 else 0
    if frames >= max(10, .40 * (end - start)) and frames >= 2 * runner_up:
        return face
    return None


def apply_editorial_evidence(cue, phrase):
    """Use unique local activity as corroboration when a human has already labeled the cue."""
    if cue.get("speakerNumber") is None or cue.get("visualFace") is not None or cue.get("overlap"):
        return
    candidate = phrase.get("activity", {}).get("referenceCandidate")
    if not candidate:
        return
    cue["visualSpeaker"] = candidate["track"]
    cue["visualFace"] = candidate["face"]
    cue["visualSpeakerBasis"] = "editorial-with-activity"
    phrase["activity"] = {"state": "visible-active", "basis": "editorial-with-activity", **candidate}
    photo = phrase.pop("referencePhoto", None)
    if photo:
        phrase["activePhoto"] = photo


def separate_conflicting_clusters(assigned, owners, verified, distinct, used):
    """Split confirmed visible talkers that one acoustic cluster grouped together."""
    separated = {}
    for number, scores in verified.items():
        faces = [face for face, _ in scores.most_common()]
        owner = owners.get(number)
        cohort = set(faces)
        if owner is not None:
            cohort.add(owner)
        if not any(pair in distinct for pair in combinations(sorted(cohort), 2)):
            continue
        if owner is None:
            owner = next((face for face in faces if face not in assigned), None)
        if owner is not None:
            assigned.setdefault(owner, {"speaker": number, "basis": "audio-visual"})
        # Once the acoustic cluster is shown to contain multiple people, rely on each
        # confirmed face's current label and leave unconfirmed audio cues on the raw cluster.
        for face in faces:
            if face == owner:
                continue
            if face not in assigned:
                label = max(used, default=0) + 1
                used.add(label)
                assigned[face] = {"speaker": label, "basis": "audio-visual", "splitFrom": number}
            separated[face, number] = assigned[face]["speaker"]
    return separated


def bind_faces(result):
    """Allocate a visible speaker label to every face and update only supported speech."""
    evidence = result.get("videoEvidence") or {}
    timeline = evidence.get("timeline")
    if not timeline:
        return
    cues = result["segments"]
    by_editor = defaultdict(set)
    for cue in cues:
        cue.pop("linkedSpeakerNumber", None)
        phrase = evidence.get("phrases", {}).get(str(cue["startMs"]), {})
        apply_editorial_evidence(cue, phrase)
        if cue.get("speakerNumber") is not None and cue.get("visualFace") is not None:
            by_editor[cue["visualFace"]].add(cue["speakerNumber"])
    explicit = unique_bindings(by_editor)
    assigned = {face: {"speaker": number, "basis": "editorial"} for face, number in explicit.items()}
    mentioned = set().union(*by_editor.values()) if by_editor else set()
    by_audio = defaultdict(Counter)
    verified = defaultdict(Counter)
    for cue in cues:
        number = cue.get("profileSpeakerNumber") or cue.get("speaker")
        if number is None or number in mentioned or cue.get("speakerNumber") is not None:
            continue
        face = active_face(cue, timeline)
        if face is not None and face not in by_editor:
            duration = min(2000, cue["endMs"] - cue["startMs"])
            by_audio[face][number] += duration if cue.get("visualFace") is not None else duration // 2
            if cue.get("visualFace") is not None and not cue.get("overlap"):
                verified[number][face] += duration
    for face, number in dominant_bindings(by_audio).items():
        assigned[face] = {"speaker": number, "basis": "audio-visual"}

    used = {number for cue in cues for number in (cue.get("speaker"), cue.get("profileSpeakerNumber"))
            if type(number) is int} | mentioned
    distinct = {tuple(sorted(pair)) for pair in (evidence.get("faceTracking") or {}).get("distinctFaces", [])
                if isinstance(pair, list) and len(pair) == 2 and all(type(face) is int for face in pair)}
    owners = {binding["speaker"]: face for face, binding in assigned.items()}
    candidates = set()
    for number, scores in verified.items():
        potential = set(scores)
        if number in owners:
            potential.add(owners[number])
        if len(potential) > 1:
            candidates.update(potential)
    if candidates:
        distinct.update(co_visible_pairs(timeline, candidates))
    separated = separate_conflicting_clusters(assigned, owners, verified, distinct, used)
    used.update(binding["speaker"] for binding in assigned.values())
    faces = sorted({face_at(hit) for hits in timeline["frames"] for hit in hits})
    number = 1
    for face in faces:
        if face in assigned:
            continue
        while number in used:
            number += 1
        assigned[face] = {"speaker": number, "basis": "visible-only"}
        used.add(number)
    evidence["faceSpeakers"] = assigned

    for cue in cues:
        binding = assigned.get(cue.get("visualFace"))
        raw = cue.get("profileSpeakerNumber") or cue.get("speaker")
        if label := separated.get((cue.get("visualFace"), raw)):
            cue["linkedSpeakerNumber"] = label
            continue
        if binding and (binding["basis"] != "visible-only" or raw is None):
            # Use an established face only on its confirmed cue. Keep conflicting clusters with
            # independent visual support instead of inheriting the face's label.
            alternatives = {face for face, scores in by_audio.items()
                            if face != cue["visualFace"] and scores[raw] >= 480}
            if (raw is not None and raw != binding["speaker"] and alternatives
                    and binding["basis"] != "editorial"):
                continue
            cue["linkedSpeakerNumber"] = binding["speaker"]

    if "speakerCount" in result:
        result["audioSpeakerCount"] = len({cue["speaker"] for cue in cues if cue.get("speaker") is not None})
        result["speakerCount"] = len({
            number for cue in cues
            if (number := cue.get("speakerNumber") or cue.get("linkedSpeakerNumber")
                or cue.get("profileSpeakerNumber") or cue.get("speaker") or cue.get("visualSpeakerNumber")) is not None
        })
