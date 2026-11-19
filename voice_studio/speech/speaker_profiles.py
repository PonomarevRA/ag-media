"""Catalog reviewed speaker references for repeated analysis of an exact upload."""

from collections import defaultdict
from uuid import uuid4

from voice_studio.speech.speech_overlap import mixed_label

MAX_PROFILES = 100
APPEARANCE_TRAITS = (("age", "ageSource", "ageSamples"),
                     ("appearanceClass", "appearanceSource", "appearanceSamples"),
                     ("hairSwatch", "hairSource", "hairSamples"))
APPEARANCE_FIELDS = tuple(key for trait in APPEARANCE_TRAITS for key in trait)


def fill_appearance(current, candidates):
    """Fill a missing trait only when the candidates agree on the observed value."""
    result = {key: current[key] for key in APPEARANCE_FIELDS if key in current}
    for value_key, source_key, count_key in APPEARANCE_TRAITS:
        if result.get(value_key) is not None:
            continue
        present = [item for item in candidates if item.get(value_key) is not None]
        if not present or any(item[value_key] != present[0][value_key] for item in present):
            continue
        best = max(present, key=lambda item: item.get(count_key) or 0)
        for key in (value_key, source_key, count_key):
            if key in best:
                result[key] = best[key]
    return result


def face_observations(attributes, face):
    """Read one face observation from either a JSON report or a local analysis."""
    return attributes.get(str(face)) or attributes.get(face) or {}


def cue_number(cue):
    """Use the displayed label, preserving editorial and face-confirmed precedence."""
    if cue.get('speakerNumber') or cue.get('linkedSpeakerNumber'):
        return cue.get('speakerNumber') or cue.get('linkedSpeakerNumber')
    if 'voiceProfileEvidence' in cue:
        # Abstained raw clusters are a different identity namespace from source profiles.
        return cue.get('profileSpeakerNumber') if cue.get('voiceProfileId') else None
    return (cue.get("speakerNumber") or cue.get("linkedSpeakerNumber") or cue.get("profileSpeakerNumber")
            or cue.get("speaker") or cue.get("visualSpeakerNumber"))


def collect_profiles(result):
    """Summarize matching audio offsets and a confirmed face without storing embeddings."""
    grouped = defaultdict(list)
    for cue in result["segments"]:
        number = cue_number(cue)
        if number is not None and not mixed_label(cue):
            grouped[number].append(cue)
    evidence = result.get("videoEvidence", {})
    editor = evidence.get("editorTimeline", {})
    outfits = editor.get("clothingByFace", {})
    attributes = editor.get("faceAttributes", {})
    faces = {binding["speaker"]: (int(face), binding["basis"])
             for face, binding in evidence.get("faceSpeakers", {}).items() if binding["basis"] != "visible-only"}
    identities = {row["number"]: row for row in result.get("speakerIdentityGroups", [])}
    records = []
    for number, cues in list(grouped.items())[:MAX_PROFILES]:
        identity = identities.get(number)
        face, basis = (identity["faceId"], "editorial") if identity else faces.get(number, (None, None))
        anchors = [cue["startMs"] for cue in sorted(cues, key=lambda item: item["startMs"])
                   if not cue.get("overlap") and cue["endMs"] - cue["startMs"] >= 650]
        visible_refs = sum(cue.get("visualFace") == face and not cue.get("overlap") for cue in cues)
        pinned = face is not None and (basis == "editorial" or visible_refs >= 2)
        record = {"number": number, "faceId": face, "faceBasis": basis,
                  "anchorsMs": anchors[:12], "pinned": pinned}
        voice_ids = {cue['voiceProfileId'] for cue in cues if cue.get('voiceProfileId')}
        if len(voice_ids) == 1:
            record['voiceProfileId'] = next(iter(voice_ids))
            public = result.get('voiceProfiles') or {}
            voice = next((row for row in public.get('profiles', []) if row.get('id') == record['voiceProfileId']), None)
            if voice:
                from voice_studio.speech.voice_profile_public import public_voice_profile
                record['voice'] = public_voice_profile(voice)
        if identity:
            record.update(confirmedIdentityId=identity["id"], faceIds=identity["faceIds"])
        outfit = outfits.get(str(face)) or outfits.get(face)
        if outfit:
            record["clothing"] = outfit
        findings = face_observations(attributes, face)
        alternate = [face_observations(attributes, other) for other in record.get("faceIds", ()) if other != face]
        appearance = fill_appearance(findings, alternate)
        if appearance:
            record["appearance"] = appearance
        records.append(record)
    return records


def matching_reference(current, previous):
    """Combine exact-source time anchors and face agreement, rejecting contrary evidence."""
    shared = len(set(current["anchorsMs"]) & set(previous["anchorsMs"]))
    left, right = current["faceId"], previous["faceId"]
    if left is not None and right is not None:
        return (2 + shared) if left == right and shared >= 1 else 0
    if previous["pinned"] or shared < 2:
        return 0
    return shared


def attach_ids(current, previous):
    """Reuse IDs only for mutually unique references across reanalysis."""
    by_current = defaultdict(list)
    by_old = defaultdict(list)
    reserved = {p["confirmedIdentityId"] for p in current if p.get("confirmedIdentityId")}
    for index, profile in enumerate(current):
        if profile.get("confirmedIdentityId"):
            continue
        for old in previous:
            if old["id"] in reserved:
                continue
            score = matching_reference(profile, old)
            if score:
                by_current[index].append(old["id"])
                by_old[old["id"]].append(index)
    for index, profile in enumerate(current):
        if confirmed := profile.get("confirmedIdentityId"):
            profile.update(id=confirmed, reused=True)
            continue
        if voice_id := profile.get('voiceProfileId'):
            profile.update(id=voice_id, reused=any(old.get('id') == voice_id for old in previous))
            continue
        available = by_current[index]
        reused = len(available) == 1 and len(by_old[available[0]]) == 1
        profile["id"] = available[0] if reused else uuid4().hex
        profile["reused"] = reused
    previous_by_id = {profile["id"]: profile for profile in previous}
    for profile in current:
        old = previous_by_id.get(profile["id"])
        if not profile["reused"] or not old or profile["faceId"] is None or profile["faceId"] != old["faceId"]:
            continue
        if appearance := fill_appearance(profile.get("appearance", {}), [old.get("appearance", {})]):
            profile["appearance"] = appearance
    return current
