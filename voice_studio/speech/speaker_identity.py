"""Replay confirmed same-person groups on an exact source using timed evidence."""

from collections import Counter
from math import isfinite
from uuid import uuid4

from voice_studio.vision.face_speakers import co_visible_pairs, face_at
from voice_studio.speech.speaker_profiles import cue_number
from voice_studio.speech.speech_overlap import mixed_label

IDENTITY_REVIEW_WARNING = (
    "Сохранённое объединение спикеров требует проверки: текущие лица противоречат подтверждению."
)


def face_anchors(timeline, identities):
    """Sample at most six separated boxes per confirmed face on the original clock."""
    collected = {face: [] for face in identities}
    for index, hits in enumerate(timeline["frames"]):
        stamp = index * timeline["stepMs"]
        for hit in hits:
            face = face_at(hit)
            if face in collected:
                rows = collected[face]
                if len(rows) < 6 and (not rows or stamp - rows[-1][0] >= 160):
                    rows.append((stamp, hit[1:5]))
    return {str(face): rows[:6] for face, rows in collected.items() if len(rows) >= 2}


def make_confirmation(report, preferred, duplicates, additional_faces):
    """Capture acknowledged profile numbers, utterances and geometry from a full analysis."""
    numbers = {preferred, *duplicates}
    evidence = report["videoEvidence"]
    bindings = evidence["faceSpeakers"]
    faces = {int(face) for face, row in bindings.items() if row["speaker"] in numbers}
    anchors = face_anchors(evidence["timeline"], faces | set(additional_faces))
    if set(map(int, anchors)) != faces | set(additional_faces):
        raise ValueError("Подтверждённое лицо должно иметь два измерения в исходнике.")
    primary = next(int(face) for face, row in bindings.items() if row["speaker"] == preferred)
    profile = next((row["id"] for row in report.get("speakerProfileIndex", []) if row["number"] == preferred), None)
    cues = [cue_reference(cue) for cue in report["segments"]
            if cue_number(cue) in numbers and not mixed_label(cue) and not cue.get("overlap")]
    return {"id": profile or uuid4().hex, "preferred": preferred, "numbers": sorted(numbers),
            "primaryFaceId": primary, "faces": anchors, "cues": cues}


def cue_reference(cue):
    """Normalize harmless whitespace and case while preserving the original clock."""
    return (cue["startMs"], cue["endMs"], " ".join(cue["text"].casefold().split()))


def intersection_score(left, right):
    """Return normalized intersection over union for source-frame rectangles."""
    x, y, w, h = left
    a, b, c, d = right
    intersection = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
    union = w * h + c * d - intersection
    return intersection / union if union > 0 and isfinite(union) else 0


def current_faces(timeline, confirmation):
    """Re-identify tracks by unique same-time geometry, independent of detector numbering."""
    step = timeline["stepMs"]
    frames = timeline["frames"]
    resolved = {}
    for original, anchors in confirmation["faces"].items():
        votes = Counter()
        for stamp, box in anchors:
            index = stamp // step
            if index >= len(frames):
                continue
            scores = {}
            for hit in frames[index]:
                face = face_at(hit)
                scores[face] = max(scores.get(face, 0), intersection_score(box, hit[1:5]))
            leaders = Counter(scores).most_common(2)
            if leaders and leaders[0][1] >= .65 and (len(leaders) == 1 or leaders[0][1] - leaders[1][1] >= .15):
                votes[leaders[0][0]] += 1
        ranking = votes.most_common(2)
        if ranking and ranking[0][1] >= 2 and (len(ranking) == 1 or ranking[0][1] > ranking[1][1]):
            resolved[int(original)] = ranking[0][0]
    return resolved


def matches_cue(cue, confirmation, faces):
    """Retain explicit contradictory edits and avoid assigning overlapping speech."""
    explicit = cue.get("speakerNumber")
    if cue.get("overlap") or mixed_label(cue) or (explicit is not None and explicit not in confirmation["numbers"]):
        return False
    if cue.get("visualFace") is not None:
        return cue["visualFace"] in faces
    start, end, text = cue_reference(cue)
    return any(text == value and abs(start - a) <= 240 and abs(end - b) <= 240
               for a, b, value in confirmation["cues"])


def distinct_confirmation_pairs(evidence, faces):
    """Read only established current face non-matches, never missing or failed model facts."""
    tracking = evidence.get("faceTracking") or {}
    if not isinstance(tracking, dict) or tracking.get("status") != "ok":
        return []
    rows = tracking.get("distinctFaces")
    if not isinstance(rows, (list, tuple)):
        return []
    pairs = set()
    for pair in rows:
        if (isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(type(face) is int and face > 0 for face in pair)
                and pair[0] != pair[1] and set(pair).issubset(faces)):
            pairs.add(tuple(sorted(pair)))
    return [list(pair) for pair in sorted(pairs)]


def clear_derived_identity(result, identity):
    """Withdraw this replay's previous outputs while preserving explicit cue edits and history."""
    groups = result.get("speakerIdentityGroups") or []
    previous = [group for group in groups if group.get("id") == identity]
    faces = {face for group in previous for face in group.get("faceIds", [])}
    face_keys = {str(face) for face in faces}
    numbers = {group.get("number") for group in previous}
    backups = {}
    for group in previous:
        backups.update(group.get("replayFaceBindings") or {})
    for cue in result["segments"]:
        if cue.get("speakerIdentityId") != identity:
            continue
        restore_cue_overlay(cue, identity)
    bindings = (result.get("videoEvidence") or {}).get("faceSpeakers") or {}
    for face, binding in list(bindings.items()):
        owned = binding.get("speakerIdentityId") == identity
        legacy_owned = (str(face) in face_keys
                        and binding.get("speaker") in numbers and binding.get("confirmedIdentity"))
        if binding.get("basis") == "editorial" and (owned or legacy_owned):
            bindings[face] = restored_face_binding(result, face, backups.get(str(face)))
    retained = [group for group in groups if group.get("id") != identity]
    if retained:
        result["speakerIdentityGroups"] = retained
    else:
        result.pop("speakerIdentityGroups", None)
    if "speakerProfileIndex" in result:
        result["speakerProfileIndex"] = [row for row in result["speakerProfileIndex"]
                                         if row.get("confirmedIdentityId") != identity]


def restore_cue_overlay(cue, identity):
    """Restore backed-up same-clock fields, retaining later explicit labels and speakerNumber edits."""
    cue.pop("speakerIdentityId", None)
    replay = cue.pop("speakerIdentityReplay", None)
    backed_up = isinstance(replay, dict) and replay.get("id") == identity
    same_clock = (backed_up and replay.get("startMs") == cue.get("startMs")
                  and replay.get("endMs") == cue.get("endMs"))
    if backed_up:
        # A later explicit linked-label change belongs to its caller, not this overlay.
        if replay.get("changedLinked") and cue.get("linkedSpeakerNumber") == replay.get("assignedLinked"):
            if same_clock and replay.get("hadLinked"):
                cue["linkedSpeakerNumber"] = replay.get("originalLinked")
            else:
                cue.pop("linkedSpeakerNumber", None)
    else:
        cue.pop("linkedSpeakerNumber", None)
    if cue.get("speakerProfileId") == identity:
        if same_clock and replay.get("hadProfile") and replay.get("originalIdentity") in (None, identity):
            cue["speakerProfileId"] = replay.get("originalProfile")
        else:
            cue.pop("speakerProfileId", None)


def restored_face_binding(result, face, backup):
    """Restore the pre-replay binding, prioritizing current edits and measured source voice proof."""
    cues = [cue for cue in result["segments"] if str(cue.get("visualFace")) == str(face)]
    edited = {cue["speakerNumber"] for cue in cues if type(cue.get("speakerNumber")) is int}
    if len(edited) == 1:
        return {"speaker": next(iter(edited)), "basis": "editorial"}
    if edited:
        return {"speaker": None, "basis": "visible-only"}
    if isinstance(backup, dict):
        return dict(backup)
    # Legacy replays did not save a binding backup. A source voice with current
    # explicit AV proof can restore it; raw integer coincidence cannot do so.
    measured = set()
    for cue in cues:
        number = cue.get("profileSpeakerNumber")
        proof = cue.get("voiceProfileEvidence") or {}
        if (type(number) is int and number > 0 and cue.get("voiceProfileId")
                and proof.get("basis") == "explicit-video"):
            measured.add(number)
    if len(measured) == 1:
        return {"speaker": next(iter(measured)), "basis": "audio-visual", "voiceProfileMapped": True}
    return {"speaker": None, "basis": "visible-only"}


def identity_conflict_warning(result, conflicts):
    """Publish bounded current diagnostics and combine our warning without duplicating other text."""
    warning = result.get("speakerProfileWarning")
    lines = warning.splitlines() if isinstance(warning, str) else []
    lines = [line for line in lines if line != IDENTITY_REVIEW_WARNING]
    if conflicts:
        result["speakerIdentityConflicts"] = conflicts[:64]
        lines.append(IDENTITY_REVIEW_WARNING)
    else:
        result.pop("speakerIdentityConflicts", None)
    if lines:
        result["speakerProfileWarning"] = "\n".join(lines)
    elif isinstance(warning, str):
        result.pop("speakerProfileWarning", None)


def apply_confirmations(result, confirmations, *, reaffirmed_ids=()):
    """Replay timed identities, abstaining on current contradictory face measurements.

    Existing automatic recovery never supplies reaffirmed_ids. Only an explicit
    current-user action may pass exact group IDs to override a model non-match;
    this ephemeral bypass is never written into a stored confirmation. The existing
    simultaneous-face veto still applies. Explicit speakerNumber edits are retained.
    """
    evidence = result.get("videoEvidence") or {}
    timeline = evidence.get("timeline")
    if not timeline or not timeline.get("stepMs") or not timeline.get("frames"):
        return []
    cues = result["segments"]
    bindings = evidence.setdefault("faceSpeakers", {})
    applied = []
    requested = reaffirmed_ids if isinstance(reaffirmed_ids, (list, tuple, set, frozenset)) else ()
    reaffirmed = {value for value in requested if isinstance(value, str)}
    processed = {confirmation["id"] for confirmation in confirmations}
    conflicts = [row for row in result.get("speakerIdentityConflicts", [])
                 if isinstance(row, dict) and row.get("id") not in processed]
    had_derived = any(cue.get("speakerIdentityId") in processed for cue in cues)
    # Re-evaluate derived overlays from retained source evidence on every recovery.
    # A veto must not leave a previous successful replay active in the same object.
    for identity in processed:
        clear_derived_identity(result, identity)
    for confirmation in confirmations:
        matched = current_faces(timeline, confirmation)
        # A conflicting explicit cue blocks geometry-only reuse of that face.
        contested = {cue.get("visualFace") for cue in cues if cue.get("speakerNumber") is not None
                     and cue["speakerNumber"] not in confirmation["numbers"]}
        faces = set(matched.values()) - contested
        co_visible = co_visible_pairs(timeline, faces) if len(faces) > 1 else set()
        distinct = distinct_confirmation_pairs(evidence, faces)
        if co_visible or (distinct and confirmation["id"] not in reaffirmed):
            pairs = [list(pair) for pair in sorted(co_visible)] if co_visible else distinct
            conflicts.append({"id": confirmation["id"], "state": "needs-review", "faceIds": sorted(faces),
                              "basis": "co-visible-faces" if co_visible else "strong-face-nonmatch", "pairs": pairs})
            continue
        selected = {id(cue) for cue in cues if matches_cue(cue, confirmation, faces)}
        if not faces or not selected:
            continue
        occupied = {cue_number(cue) for cue in cues if id(cue) not in selected and not mixed_label(cue)}
        occupied.update(row["speaker"] for face, row in bindings.items()
                        if int(face) not in faces and row["basis"] != "visible-only")
        number = confirmation["preferred"]
        while number in occupied:
            number += 1
        primary = matched.get(confirmation["primaryFaceId"])
        primary = primary if primary in faces else min(faces)
        previous_bindings = {}
        for face in faces:
            prior = bindings.get(str(face), bindings.get(face))
            if isinstance(prior, dict):
                previous_bindings[str(face)] = dict(prior)
            bindings.pop(str(face), None)
            bindings[face] = {"speaker": number, "basis": "editorial", "confirmedIdentity": True,
                              "speakerIdentityId": confirmation["id"]}
        for cue in cues:
            if id(cue) in selected:
                cue["speakerIdentityReplay"] = {
                    "id": confirmation["id"], "hadLinked": "linkedSpeakerNumber" in cue,
                    "originalLinked": cue.get("linkedSpeakerNumber"),
                    "changedLinked": cue.get("speakerNumber") is None, "assignedLinked": number,
                    "hadProfile": "speakerProfileId" in cue, "originalProfile": cue.get("speakerProfileId"),
                    "originalIdentity": cue.get("speakerIdentityId"),
                    "startMs": cue.get("startMs"), "endMs": cue.get("endMs"),
                }
                if cue.get("speakerNumber") is None:
                    cue["linkedSpeakerNumber"] = number
                cue["speakerIdentityId"] = confirmation["id"]
        applied.append({"id": confirmation["id"], "number": number, "faceId": primary,
                        "faceIds": sorted(faces), "replayFaceBindings": previous_bindings})
    if applied:
        other = [row for row in result.get("speakerIdentityGroups", []) if row.get("id") not in processed]
        result["speakerIdentityGroups"] = other + applied
    if applied or conflicts or had_derived:
        result["speakerCount"] = len({cue_number(cue) for cue in cues if cue_number(cue) is not None})
    identity_conflict_warning(result, conflicts)
    return applied
