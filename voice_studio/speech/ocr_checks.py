"""Compare utterance tokens against nearby on-screen text without changing the transcript."""

from bisect import bisect_left, bisect_right
import re
from unicodedata import normalize

from voice_studio.vision.scene_transcript import one_edit_apart

NEARBY_MS = 5_000
MAX_CHECKS = 200
IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,12})[- ]?(\d{4,12})(?![A-Za-z0-9])")
MEASUREMENT = re.compile(
    r"(?<![A-Za-z0-9.])(?<![0-9],)(\d{1,9}(?:[.,]\d{1,8})?)\s?"
    r"(%|％|厘米|毫米|公里|米|秒|年|月|日|см|мм|км|cm|mm|km|kg|ms)(?![A-Za-zА-Яа-яЁё])", re.I
)
ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3,10}(?![A-Za-z0-9])")
NAME = re.compile(r"(?<!\w)([A-Z][a-z]{2,20}|[А-ЯЁ][а-яё]{2,20}) "
                  r"([A-Z][a-z]{3,24}|[А-ЯЁ][а-яё]{3,24})(?!\w)")
CHINESE_NAME = re.compile(r"(?:姓名\s*[:：]\s*([\u3400-\u9fff]{2,4})|"
                          r"([\u3400-\u9fff]{2,4})(?:先生|女士|教授))")
UNITS = {"厘米": "cm", "см": "cm", "毫米": "mm", "мм": "mm", "公里": "km", "км": "km", "％": "%"}


def check_tokens(text):
    """Extract comparable values, keeping units, prefixes and first names as context."""
    text = normalize("NFKC", text)
    identifier_prefixes = set()
    for match in IDENTIFIER.finditer(text):
        prefix, digits = match.groups()
        identifier_prefixes.add(match.start())
        yield "identifier", prefix.casefold(), digits, match.group()
    for match in MEASUREMENT.finditer(text):
        value, unit = match.groups()
        unit = unit.casefold()
        yield "number", UNITS.get(unit, unit), value.replace(",", "."), match.group()
    for match in ACRONYM.finditer(text):
        if match.start() not in identifier_prefixes:
            yield "name", "acronym", match.group().casefold(), match.group()
    for match in NAME.finditer(text):
        first, last = match.groups()
        yield "name", first.casefold(), last.casefold(), match.group()
    for match in CHINESE_NAME.finditer(text):
        name = match[1] or match[2]
        yield "name", "han-name", name, name


def compare_token(token, reading):
    """Use an exact match or one edit, with extra caution for short names and IDs."""
    kind, context, value, _ = token
    other_kind, other_context, other_value, _ = reading
    if (kind, context) != (other_kind, other_context):
        return None
    if value == other_value:
        return "confirmed"
    if (kind == "identifier" or kind == "name" and context != "han-name") and min(len(value), len(other_value)) < 4:
        return None
    return "review" if one_edit_apart(value, other_value) else None


def measure_ocr_checks(cues, scene):
    """Return bounded matches on the source clock, using nearby OCR as advisory evidence."""
    if scene.get("status") != "ok":
        return []
    observations = []
    for frame in scene.get("frames", ()):
        stamp = frame.get("timeMs")
        if type(stamp) is not int or stamp < 0:
            continue
        readings = set()
        for row in frame.get("text") or ():
            if isinstance(value := row.get("text"), str):
                readings.update(check_tokens(value))
        if readings:
            observations.append((stamp, readings))
    observations.sort(key=lambda entry: entry[0])
    stamps = [stamp for stamp, _ in observations]
    checks = []
    for cue in cues:
        start, end = cue["startMs"], cue["endMs"]
        first = bisect_left(stamps, start - NEARBY_MS)
        last = bisect_right(stamps, end + NEARBY_MS)
        if first == last:
            continue
        for token in dict.fromkeys(check_tokens(cue["text"])):
            check = closest_reading(token, observations[first:last], start, end)
            if check:
                checks.append(check)
                if len(checks) == MAX_CHECKS:
                    return checks
    return checks


def closest_reading(token, observations, start, end):
    """Prefer an exact reading and suppress conflicting near-matches."""
    confirmed, alternatives = [], {}
    for stamp, readings in observations:
        for reading in readings:
            status = compare_token(token, reading)
            if status is None:
                continue
            _, _, value, observed = reading
            # Compare distance to the cue first, then distance to the middle of the cue.
            match = (max(start - stamp, stamp - end, 0), abs(2 * stamp - start - end), stamp, observed)
            if status == "confirmed":
                confirmed.append(match)
            elif value not in alternatives or match < alternatives[value]:
                alternatives[value] = match
    if confirmed:
        _, _, stamp, observed = min(confirmed)
        status = "confirmed"
    elif len(alternatives) == 1:
        _, _, stamp, observed = next(iter(alternatives.values()))
        status = "review"
    else:
        return None
    return {"cueStartMs": start, "timeMs": stamp, "kind": token[0], "transcript": token[3],
            "observed": observed, "status": status}
