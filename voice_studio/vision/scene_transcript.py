"""Reconcile a near-miss surname with repeated title-card OCR during the same cue."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import re

WORD = re.compile(r"(?<![\w-])[А-ЯЁа-яё]{3,32}(?![\w-])")
OCR_NAME = re.compile(r"[А-ЯЁа-яё]{3,32}")
MIN_FRAME_DISTANCE_MS = 200
BASIS = "repeated-title-card-ocr"


def title_pair(top, bottom):
    """Recognize two vertically stacked and horizontally aligned OCR lines."""
    x, y, width, height = top
    u, v, other_width, other_height = bottom
    common = min(x + width, u + other_width) - max(x, u)
    return (0.02 <= v - y <= .23 and abs(x - u) <= .14 and common >= .65 * min(width, other_width)
            and max(height, other_height) <= 1.6 * min(height, other_height))


def visible_names(frame):
    """Read complete name captions and large single title lines from one image."""
    names, lines = set(), []
    for item in frame.get("text") or []:
        value = item.get("text", "")
        box = item.get("box", [])
        if not isinstance(value, str) or not isinstance(box, list) or len(box) != 4:
            continue
        if (any(type(part) not in (int, float) or not 0 <= part <= 1 for part in box)
                or min(box[2:]) <= 0 or box[3] < .055):
            continue
        words = value.split()
        if len(words) not in (1, 2) or any(not OCR_NAME.fullmatch(word) for word in words):
            continue
        if len(words) == 2:
            names.add(tuple(word.title() for word in words))
        else:
            lines.append((words[0].title(), box))
    for first, top in lines:
        for last, bottom in lines:
            if first != last and title_pair(top, bottom):
                names.add((first, last))
    title_lines = [(name, box) for name, box in lines if box[2] >= .35 and box[3] >= .10]
    return names, title_lines


def one_edit_apart(first, second):
    """Check one character substitution, insertion or omission."""
    if len(first) == len(second):
        return sum(a != b for a, b in zip(first, second)) == 1
    shorter, longer = sorted((first, second), key=len)
    if len(longer) - len(shorter) != 1:
        return False
    mismatch = next((index for index, pair in enumerate(zip(shorter, longer)) if pair[0] != pair[1]),
                    len(shorter))
    return shorter[mismatch:] == longer[mismatch + 1:]


def repeated_line(rows):
    """Match a large single title line across independent nearby frames."""
    for index, (first, (x, y, width, height)) in enumerate(rows):
        for last, (u, v, other_width, other_height) in rows[index + 1:]:
            if (last - first >= MIN_FRAME_DISTANCE_MS and abs(x - u) <= .12 and abs(y - v) <= .12
                    and abs(width - other_width) <= .12 and abs(height - other_height) <= .08):
                return [first, last]
    return None


def name_confirmation(first, raw, witnesses, title_lines):
    """Prefer a complete name and require stable size and position for surname-only OCR."""
    readings = witnesses.get(first, {})
    if raw in readings or raw in title_lines:
        return None
    nearby = {name for name in set(readings) | set(title_lines) if one_edit_apart(name.casefold(), raw.casefold())}
    if len(nearby) != 1:
        return None
    replacement = nearby.pop()
    for stamps in readings.get(replacement, {}).values():
        if stamps[-1] - stamps[0] >= MIN_FRAME_DISTANCE_MS:
            return replacement, [stamps[0], stamps[-1]]
    if any(replacement in names for name, names in witnesses.items() if name != first):
        return None
    for rows in title_lines.get(replacement, {}).values():
        if stamps := repeated_line(rows):
            return replacement, stamps
    return None


def supported_correction(cue, witnesses, title_lines):
    """Resolve one edit only when every plausible caption points to the same surname."""
    text = cue["text"]
    words = list(WORD.finditer(text))
    candidates = []
    for first, last in zip(words, words[1:]):
        raw = last.group()
        if not (first.group().istitle() and raw.istitle() and len(raw) >= 5
                and text[first.end():last.start()].isspace()):
            continue
        # The local person recognizer reports UTF-16 coordinates for the browser.
        start = len(text[:first.start()].encode("utf-16-le")) // 2
        end = len(text[:last.end()].encode("utf-16-le")) // 2
        if not any(item["start"] <= start and item["end"] >= end for item in cue.get("personMentions", [])):
            continue
        if confirmation := name_confirmation(first.group(), raw, witnesses, title_lines):
            replacement, stamps = confirmation
            candidates.append({"word": raw, "replacement": replacement, "start": last.start(),
                               "end": last.end(), "timestampsMs": stamps, "basis": BASIS})
    return candidates[0] if len(candidates) == 1 else None


def reconcile_names(cues, scene, language):
    """Update only untouched Russian ASR cues with duplicate in-cue OCR in the same shot."""
    if language != "ru" or scene.get("status") != "ok":
        return 0
    entries = []
    cuts = sorted(set(scene.get("shotStartsMs") or []))
    for frame in scene.get("frames") or []:
        stamp = frame.get("timeMs")
        if type(stamp) is int and stamp >= 0:
            names, lines = visible_names(frame)
            if names or lines:
                entries.append((stamp, names, lines))
    entries.sort(key=lambda entry: entry[0])
    times = [entry[0] for entry in entries]
    changed = 0
    for cue in cues:
        if "asrText" in cue or not cue.get("personMentions"):
            continue
        text = cue["text"]
        start, stop = bisect_left(times, cue["startMs"]), bisect_left(times, cue["endMs"])
        witnesses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        title_lines = defaultdict(lambda: defaultdict(list))
        for stamp, names, lines in entries[start:stop]:
            shot = bisect_right(cuts, stamp)
            for first, last in names:
                witnesses[first][last][shot].append(stamp)
            for last, box in lines:
                title_lines[last][shot].append((stamp, box))
        if fix := supported_correction(cue, witnesses, title_lines):
            cue["asrText"] = text
            cue["asrCorrection"] = fix
            cue["text"] = text[:fix["start"]] + fix["replacement"] + text[fix["end"]:]
            cue.pop("personMentions", None)
            changed += 1
    return changed


def clean_asr_correction(cue, original, correction):
    """Allow a saved OCR correction only when it exactly explains the current cue text."""
    if not isinstance(original, str) or not 0 < len(original) <= 4000 or not isinstance(correction, dict):
        return None
    start, end = correction.get("start"), correction.get("end")
    word, replacement = correction.get("word"), correction.get("replacement")
    stamps = correction.get("timestampsMs")
    if (correction.get("basis") != BASIS or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(original) or original[start:end] != word
            or not isinstance(word, str) or not isinstance(replacement, str) or not OCR_NAME.fullmatch(word)
            or not OCR_NAME.fullmatch(replacement) or not one_edit_apart(word.casefold(), replacement.casefold())):
        return None
    if (not isinstance(stamps, list) or len(stamps) != 2 or any(type(stamp) is not int for stamp in stamps)
            or not cue["startMs"] <= stamps[0] < stamps[1] < cue["endMs"]
            or stamps[1] - stamps[0] < MIN_FRAME_DISTANCE_MS):
        return None
    if original[:start] + replacement + original[end:] != cue["text"]:
        return None
    return {"start": start, "end": end, "word": word, "replacement": replacement,
            "timestampsMs": stamps, "basis": BASIS}
