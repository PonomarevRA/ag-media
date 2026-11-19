"""Represent simultaneous audio or visual speech without assigning mixed words."""

from collections import Counter, defaultdict

MIN_AUDIO_OVERLAP_MS = 120
MIN_VIDEO_OVERLAP_MS = 200


def audio_overlaps(turns, offset_ms=0):
    """Sweep half-open speaker turns, counting unique speakers even when tracks repeat."""
    events = defaultdict(Counter)
    for turn in turns:
        start, end, speaker = turn["startMs"], turn["endMs"], turn["speaker"]
        if start < end:
            events[start][speaker] += 1
            events[end][speaker] -= 1
    active = Counter()
    previous = 0
    spans = []
    for time, changes in sorted(events.items()):
        speakers = sorted(number for number, count in active.items() if count > 0)
        if len(speakers) > 1 and previous < time:
            append_span(spans, previous + offset_ms, time + offset_ms, "speakers", speakers)
        active.update(changes)
        for speaker in changes:
            if not active[speaker]:
                del active[speaker]
        previous = time
    return [row for row in spans if row["endMs"] - row["startMs"] >= MIN_AUDIO_OVERLAP_MS]


def visual_overlaps(timeline):
    """Find sustained simultaneous Light-ASD activity in distinct visible tracks."""
    step = timeline["stepMs"]
    spans = []
    for index, hits in enumerate(timeline["frames"]):
        tracks = sorted({hit[0] for hit in hits if hit[5] == 1})
        if len(tracks) > 1:
            append_span(spans, index * step, (index + 1) * step, "tracks", tracks)
    return [row for row in spans if row["endMs"] - row["startMs"] >= MIN_VIDEO_OVERLAP_MS]


def append_span(spans, start, end, key, values):
    """Coalesce touching intervals with identical participants."""
    if spans and spans[-1]["endMs"] == start and spans[-1][key] == values:
        spans[-1]["endMs"] = end
    else:
        spans.append({"startMs": start, "endMs": end, key: values})


def clip_overlaps(cues, spans):
    """Attach exact intervals to mixed transcript cues, retaining original text and labels."""
    first = 0
    for cue in cues:
        start, end = cue["startMs"], cue["endMs"]
        while first < len(spans) and spans[first]["endMs"] <= start:
            first += 1
        matches = []
        for index in range(first, len(spans)):
            span = spans[index]
            if span["startMs"] >= end:
                break
            matches.append({**span, "startMs": max(start, span["startMs"]), "endMs": min(end, span["endMs"])})
        if matches:
            cue.update(overlap=True, overlaps=matches)
        elif "overlap" in cue:
            cue["overlap"] = False


def mixed_label(cue, language="ru"):
    """Render one label for a mixed cue while preserving an explicit human correction."""
    if not cue.get("overlap") or cue.get("speakerNumber") or cue.get("speakerName"):
        return None
    speakers = sorted({number for span in cue.get("overlaps", ()) for number in span["speakers"]})
    title, label = ("Overlapping speech", "speakers") if language == "en" else ("Голоса пересекаются", "спикеры")
    return title + (f" · {label} {', '.join(map(str, speakers))}" if speakers else "")
