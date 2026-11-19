"""Reconcile one clean Sherpa turn with a confirmed talker in a wider subtitle."""

from collections import defaultdict

from voice_studio.speech.short_reply_review import has_overlap
from voice_studio.vision.face_reid import cosine

BASIS = "voiced-window-face-and-voice"


def dominant_window_face(window, timeline):
    """Require a unique face speaking in most frames of the measured audio interval."""
    step = timeline["stepMs"]
    start = (window["startMs"] + step - 1) // step
    end = (window["endMs"] + step - 1) // step
    frames = timeline["frames"][start:end]
    if len(frames) != end - start or len(frames) < 12:
        return None
    votes = defaultdict(int)
    for hits in frames:
        active = {hit[6] if len(hit) > 6 else hit[0] for hit in hits if hit[5] == 1}
        if len(active) > 1:
            return None
        if active:
            votes[next(iter(active))] += 1
    if len(votes) != 1:
        return None
    face, count = next(iter(votes.items()))
    return face if count >= .60 * len(frames) else None


def window_candidates(cues, timeline, overlaps):
    """Pair a wrongly split local turn with the sole confirmed voice for its face."""
    anchors = defaultdict(list)
    direct_faces = defaultdict(set)
    for cue in cues:
        face, speaker = cue.get("visualFace"), cue.get("speaker")
        if face is None or speaker is None or has_overlap(cue) or cue.get("speakerNumber") is not None:
            continue
        direct_faces[speaker].add(face)
        anchors[face].append(cue)

    concurrent = {frozenset(span["speakers"]) for span in overlaps}
    for cue in cues:
        window = cue.get("speakerAudioWindow")
        raw = cue.get("speaker")
        if (not window or raw is None or cue.get("visualFace") is not None or has_overlap(cue)
                or cue.get("profileSpeakerNumber") is not None or cue.get("speakerNumber") is not None
                or window["endMs"] - window["startMs"] < 950):
            continue
        face = dominant_window_face(window, timeline)
        if face is None or direct_faces[raw] - {face}:
            continue
        refs = [ref for ref in anchors[face] if ref["endMs"] - ref["startMs"] >= 950]
        owners = {ref["speaker"] for ref in refs}
        if len(owners) != 1:
            continue
        owner = next(iter(owners))
        if raw == owner or any({raw, owner} <= group for group in concurrent):
            continue
        yield cue, face, owner, refs


def review_voiced_windows(cues, timeline, overlaps, wav, offset, extract_embeddings):
    """Verify bounded clean audio crops and change the display label of only that cue."""
    from itertools import islice

    candidates = list(islice(window_candidates(cues, timeline, overlaps), 6))
    if not candidates:
        return []
    spans = {}
    for index, (cue, _, _, refs) in enumerate(candidates):
        spans[index * 2] = [audio_crop(cue, offset)]
        spans[index * 2 + 1] = [audio_crop(ref, offset) for ref in refs[:3]]
    samples = extract_embeddings(wav, spans)
    decisions = []
    for index, (cue, face, owner, _) in enumerate(candidates):
        target = samples.get(index * 2, [])
        anchors = samples.get(index * 2 + 1, [])
        if not target or not anchors:
            continue
        score = max(cosine(target[0], anchor) for anchor in anchors)
        if score >= .50:
            cue.update(profileSpeakerNumber=owner, speakerBasis=BASIS)
            decisions.append({"startMs": cue["startMs"], "from": cue["speaker"], "to": owner,
                              "face": face, "voiceCosine": round(score, 3),
                              "window": cue["speakerAudioWindow"], "basis": BASIS})
    return decisions


def audio_crop(cue, offset):
    """Read a short PCM slice on the decoded audio clock."""
    window = cue.get("speakerAudioWindow") or cue
    start = max(0, window["startMs"] - offset)
    return [start, min(window["endMs"] - offset, start + 3600)]
