"""Recover a missed short speaker turn from independent face and voice evidence."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from statistics import median

from voice_studio.vision.face_reid import cosine
from voice_studio.vision.face_speakers import active_face

MIN_REPLY_MS = 600
MAX_REPLY_MS = 1500
MIN_ANCHOR_MS = 1500
MAX_ANCHOR_MS = 3600
MAX_CANDIDATES = 6
BASIS = "short-reply-face-and-voice"


@dataclass
class Reply:
    """Hold one provisional reply and the surrounding speaker's independent voice anchors."""

    cue: dict
    phrase: dict
    candidate: dict
    references: list


def source_span(cue, audio_start_ms):
    """Convert a whole cue into a bounded window on the decoded audio clock."""
    start = cue["startMs"] - audio_start_ms
    return [start, min(cue["endMs"] - audio_start_ms, start + MAX_ANCHOR_MS)]


def has_overlap(cue):
    """Honor both the provisional diarization and the explicit overlap ranges."""
    return cue.get("overlap") or bool(cue.get("overlaps"))


def voice_references(cues, timeline, audio_start_ms):
    """Index clean windows and independently established direct face anchors per audio label."""
    references, direct = defaultdict(list), defaultdict(set)
    for cue in cues:
        number = cue.get("speaker")
        start, end = source_span(cue, audio_start_ms)
        if number is None or start < 0 or has_overlap(cue):
            continue
        if (face := cue.get("visualFace")) is not None:
            direct[number].add(face)
        if end - start >= MIN_ANCHOR_MS and (face := active_face(cue, timeline)) is not None:
            references[number, face].append([start, end])
    return references, direct


def find_replies(cues, evidence, audio_start_ms):
    """Require a held raw label, a unique active face, and a distinct established face."""
    timeline = evidence.get("timeline")
    if not timeline or len(cues) < 3:
        return []
    tracking = evidence.get("faceTracking") or {}
    distinct = {tuple(sorted(pair)) for pair in tracking.get("distinctFaces", [])}
    cuts = {row["startMs"] for row in (evidence.get("editorTimeline") or {}).get("shots", [])}
    cuts.update(tracking.get("identityCutTimesMs", []))
    references, direct = voice_references(cues, timeline, audio_start_ms)
    established_faces = set().union(*direct.values())
    replies = []
    for previous, cue, following in zip(cues, cues[1:], cues[2:]):
        raw = cue.get("speaker")
        start, end = cue["startMs"], cue["endMs"]
        if (raw is None or raw != previous.get("speaker") or raw != following.get("speaker")
                or has_overlap(cue) or cue.get("speakerNumber") is not None or cue.get("visualFace") is not None
                or start < audio_start_ms or not MIN_REPLY_MS <= end - start <= MAX_REPLY_MS):
            continue
        if max(start - previous["endMs"], following["startMs"] - end) > 450:
            continue
        if any(start < cut < end for cut in cuts):
            continue
        phrase = evidence.get("phrases", {}).get(str(start), {})
        activity = phrase.get("activity", {})
        candidate = activity.get("referenceCandidate")
        if activity.get("state") != "partial-or-ambiguous" or not candidate or len(direct[raw]) != 1:
            continue
        face = candidate["face"]
        established = next(iter(direct[raw]))
        if face in established_faces or tuple(sorted((face, established))) not in distinct:
            continue
        if candidate["phraseFrames"] <= 0 or candidate["activeFrames"] * 10 < candidate["phraseFrames"] * 7:
            continue
        windows = [window for window in references[raw, established]
                   if window[1] <= start - audio_start_ms or window[0] >= end - audio_start_ms]
        # Prefer longer independent anchors and avoid multiple crops from one reference.
        anchors = sorted(windows, key=lambda row: (row[0] - row[1], row[0]))[:3]
        if len(anchors) >= 2:
            replies.append(Reply(cue, phrase, candidate, anchors))
            if len(replies) == MAX_CANDIDATES:
                break
    return replies


def apply_reply(reply, number, score):
    """Record a local correction while retaining the original acoustic classification."""
    cue, candidate, phrase = reply.cue, reply.candidate, reply.phrase
    cue.update(profileSpeakerNumber=number, visualSpeaker=candidate["track"], visualFace=candidate["face"],
               visualSpeakerBasis=BASIS)
    phrase["activity"] = {"state": "visible-active", "basis": BASIS, **candidate}
    if photo := phrase.pop("referencePhoto", None):
        phrase["activePhoto"] = photo
    return {"startMs": cue["startMs"], "from": cue["speaker"], "to": number,
            "face": candidate["face"], "voiceCosine": round(score, 3), "basis": BASIS}


def review_short_replies(cues, replies, wav, audio_start_ms, extract_embeddings):
    """Use one local voice pass and abstain when the established voice still matches."""
    if not replies:
        return []
    windows = {}
    for index, reply in enumerate(replies):
        windows[index * 2] = [source_span(reply.cue, audio_start_ms)]
        windows[index * 2 + 1] = reply.references
    embeddings = extract_embeddings(wav, windows)
    used = {cue.get(field) for cue in cues for field in ("speaker", "speakerNumber", "profileSpeakerNumber")}
    next_number = max((number for number in used if type(number) is int), default=0) + 1
    decisions, promoted = [], set()
    for index, reply in enumerate(replies):
        sample = embeddings.get(index * 2, [])
        anchors = embeddings.get(index * 2 + 1, [])
        face = reply.candidate["face"]
        if not sample or len(anchors) < 2 or face in promoted or "profileSpeakerNumber" in reply.cue:
            continue
        if any(cosine(first, second) < .45 for first, second in combinations(anchors, 2)):
            continue
        similarities = [cosine(sample[0], anchor) for anchor in anchors]
        score = median(similarities)
        if score <= .30 and max(similarities) <= .40:
            decisions.append(apply_reply(reply, next_number, score))
            promoted.add(face)
            next_number += 1
    return decisions
