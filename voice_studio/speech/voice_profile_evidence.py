"""Validate speaking evidence separately from face presence and acoustic identity."""

from math import isfinite

from voice_studio.vision.static_face_evidence import strong_av_anchor_eligible

EXPLICIT_BASES = {"editorial", "audio-visual-active-speaker", "verified-speaking"}


def speaking_evidence(cue, result, visual_scope=None):
    """Return only explicit talker proof; local face IDs require a stable caller scope."""
    explicit = cue.get("speakerVideoEvidence") or {}
    subject = explicit.get("subjectId")
    if (subject is not None and explicit.get("confirmed") is True
            and explicit.get("state") == "visible-active" and explicit.get("basis") in EXPLICIT_BASES):
        return {"subjectId": str(subject), "basis": explicit["basis"],
                "evidence": explicit.get("evidence", []), "strength": "explicit"}
    if visual_scope is None:
        return None
    phrase = (result.get("videoEvidence") or {}).get("phrases", {}).get(str(cue["startMs"]), {})
    activity = phrase.get("activity") or {}
    active, total = activity.get("activeFrames", 0), activity.get("phraseFrames", 0)
    if not all(isinstance(value, (int, float)) and isfinite(value) for value in (active, total)):
        return None
    face = activity.get("face")
    expressions = (result.get("videoEvidence") or {}).get("facialExpressions") or {}
    times = [cue["startMs"]]
    for span in (expressions.get("staticFaceEvidence") or {}).get("exclusions", ()):
        left, right = span.get("startMs"), span.get("endMs")
        if type(left) is int and type(right) is int and left < cue["endMs"] and cue["startMs"] < right:
            times.append(max(left, cue["startMs"]))
    if any(not strong_av_anchor_eligible(expressions, face, stamp) for stamp in times):
        return None
    if (face is not None and activity.get("state") == "visible-active"
            and 10 <= active <= total and active / total >= .8):
        return {"subjectId": f"{visual_scope}:face:{face}", "basis": "audio-visual-active-speaker",
                "strength": "explicit", "visualScope": str(visual_scope), "identityScope": "window-local",
                "evidence": [{"activeFrames": active, "phraseFrames": total,
                "faceId": face, "timeMs": cue["startMs"]}]}
    return None


def audio_interval(cue):
    """Use a diarization voiced subwindow only when fully inside the original cue."""
    start, end = cue["startMs"], cue["endMs"]
    window = cue.get("speakerAudioWindow") or {}
    left, right = window.get("startMs", start), window.get("endMs", end)
    if start <= left < right <= end:
        return left, right
    return start, end


def is_mixed(cue):
    """Reject explicit overlap in every supported cue representation."""
    return bool(cue.get("overlap") or cue.get("overlaps") or cue.get("mixedSpeakers"))


def overlaps_audio(cue, result):
    """Reject mixed cues and independent overlap spans even before cue decoration."""
    if is_mixed(cue):
        return True
    start, end = cue["startMs"], cue["endMs"]
    for span in result.get("speechOverlaps", ()):
        if span["startMs"] < end and start < span["endMs"]:
            return True
    return False
