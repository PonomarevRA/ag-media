"""Reapply explicit speaker anchors to exact cues while retaining competing evidence."""

VERSION = "speaker-context-1"


def cue_key(cue):
    """Use both playback boundaries and original text within the same media digest."""
    return cue["startMs"], cue["endMs"], " ".join(cue["text"].split())


def resolve_speakers(cues, sources):
    """Apply only unanimous saved per-cue overrides; never propagate across a camera track."""
    anchors = {}
    for source in sources:
        for cue in source["cues"]:
            edit = source["edits"].get(str(cue["startMs"]), {})
            number = edit.get("speakerNumber")
            if number is not None:
                anchors.setdefault(cue_key(cue), []).append((source, edit))

    decisions = []
    for cue in cues:
        matches = anchors.get(cue_key(cue), [])
        if not matches:
            continue
        numbers = {edit["speakerNumber"] for _, edit in matches}
        subject = f"cue:{cue['startMs']}"
        edges = []
        if cue.get("speaker") is not None:
            edges.append({"from": subject, "relation": "audio-label", "to": f"sherpa:{cue['speaker']}"})
        if cue.get("visualSpeaker") is not None:
            edges.append({"from": subject, "relation": "visible-speaker", "to": f"light-asd:{cue['visualSpeaker']}"})
        for source, edit in matches:
            anchor = f"{source['id']}@{source['revision']}:{cue['startMs']}"
            edges.extend([
                {"from": anchor, "relation": "same-source-cue", "to": subject},
                {"from": anchor, "relation": "assigns", "to": f"speaker:{edit['speakerNumber']}"},
            ])
            context = edit.get("speakerNumberEvidence", {}).get("contextStartMs")
            if context is not None:
                edges.append({"from": anchor, "relation": "has-context", "to": f"{source['id']}:cue:{context}"})
        state = "conflicting"
        if len(numbers) == 1:
            source, edit = matches[0]
            cue["speakerNumber"] = edit["speakerNumber"]
            cue["speakerNumberEvidence"] = {
                **edit.get("speakerNumberEvidence", {}), "replayed": True,
                "materialId": source["id"], "revision": source["revision"],
            }
            state = "applied"
        decisions.append({"startMs": cue["startMs"], "state": state, "edges": edges})
    return {"version": VERSION, "decisions": decisions}
