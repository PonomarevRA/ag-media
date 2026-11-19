"""Measure edit-review intervals on the same millisecond clock as dialogue cues."""

from bisect import bisect_left, bisect_right
from operator import itemgetter

from voice_studio.audio.pauses import MIN_PAUSE_MS, quiet_spans

PCM_QUIET = "pcm-quiet"
SPEAKER_GAP = "speaker-gap"
MINIMUM_MS = {PCM_QUIET: MIN_PAUSE_MS, SPEAKER_GAP: 200}


def pause_row(start, end, position, basis):
    """Package a measured interval and its midpoint for previewing a potential cut."""
    return {"startMs": start, "endMs": end, "cutMs": (start + end) // 2,
            "position": position, "basis": basis}


def clean_dialogue_pauses(rows, duration_ms):
    """Keep bounded, nonoverlapping measured intervals from a report or editor request."""
    if not isinstance(rows, list):
        return []
    valid = []
    for row in rows:
        if not isinstance(row, dict) or row.get("basis") not in (PCM_QUIET, SPEAKER_GAP):
            continue
        start, end, position = row.get("startMs"), row.get("endMs"), row.get("position")
        if (type(start) is int and type(end) is int and 0 <= start < end <= duration_ms
                and end - start >= MINIMUM_MS[row["basis"]] and position in ("between-cues", "near-cue")):
            valid.append(pause_row(start, end, position, row["basis"]))
    result = []
    for row in sorted(valid, key=itemgetter("startMs", "endMs")):
        if not result or result[-1]["endMs"] <= row["startMs"]:
            result.append(row)
    return result


def timed_bounds(rows):
    """Read valid sorted millisecond bounds from speech observations."""
    return sorted((row["startMs"], row["endMs"]) for row in rows if isinstance(row, dict)
                  and type(row.get("startMs")) is int and type(row.get("endMs")) is int
                  and 0 <= row["startMs"] < row["endMs"])


def gaps_between_turns(turns, offset_ms):
    """Find gaps between the union of all detected speakers on the audio track."""
    previous_end = None
    for start, end in timed_bounds(turns):
        if previous_end is not None and start - previous_end >= MINIMUM_MS[SPEAKER_GAP]:
            yield previous_end + offset_ms, start + offset_ms
        previous_end = end if previous_end is None else max(previous_end, end)


def measure_dialogue_pauses(wav, cues, offset_ms=0, overlaps=(), turns=()):
    """Measure PCM quiet and add speaker gaps as separate evidence when available."""
    if not cues:
        return []
    starts = sorted(cue["startMs"] for cue in cues)
    ends = sorted(cue["endMs"] for cue in cues)
    mixed = timed_bounds(overlaps)
    pauses = []
    quiet = ((offset_ms + round(span["start"] * 1000), offset_ms + round(span["end"] * 1000))
             for span in quiet_spans(wav))
    for basis, intervals in [(PCM_QUIET, quiet), (SPEAKER_GAP, gaps_between_turns(turns, offset_ms))]:
        mixed_index = 0
        for begin, stop in intervals:
            start, end = max(starts[0], begin), min(ends[-1], stop)
            if end - start < MINIMUM_MS[basis]:
                continue
            while mixed_index < len(mixed) and mixed[mixed_index][1] <= start:
                mixed_index += 1
            if mixed_index < len(mixed) and mixed[mixed_index][0] < end:
                continue
            between = bisect_right(ends, start) == bisect_left(starts, end)
            pauses.append(pause_row(start, end, "between-cues" if between else "near-cue", basis))
    return prefer_quiet_intervals(pauses)


def prefer_quiet_intervals(pauses):
    """Use the measured quiet portion when both approaches identify the same gap."""
    quiet = [row for row in pauses if row["basis"] == PCM_QUIET]
    ends = [row["endMs"] for row in quiet]
    candidates = quiet.copy()
    for row in pauses:
        if row["basis"] == SPEAKER_GAP:
            index = bisect_right(ends, row["startMs"])
            if index == len(quiet) or quiet[index]["startMs"] >= row["endMs"]:
                candidates.append(row)
    return sorted(candidates, key=itemgetter("startMs"))


def shift_dialogue_pauses(rows, offset_ms, end_ms):
    """Translate measured windows to the full source clock and discard truncated fragments."""
    shifted = []
    for row in rows:
        start = row["startMs"] + offset_ms
        end = min(end_ms, row["endMs"] + offset_ms)
        if end - start >= MINIMUM_MS[row["basis"]]:
            shifted.append(pause_row(start, end, row["position"], row["basis"]))
    return shifted
