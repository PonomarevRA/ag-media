"""Locate prompt boundaries in Whisper's timed words before saving a voice sample."""

from pathlib import Path
from collections import Counter

from voice_studio.prosody.reference_script import GUIDES, spoken_words
from voice_studio.speech.recognition import SMALL_MODEL, ready, recognize_file
from voice_studio.synthesis.engines import normalize
from voice_studio.synthesis.prosody_alignment import RATE, read_pcm, write_pcm
from voice_studio.synthesis.speech_metrics import word_error_rate

NUMBERS = {
    "ru": ("один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь"),
    "en": ("one", "two", "three", "four", "five", "six", "seven", "eight"),
}
MARKER = {"ru": "номер", "en": "number"}
EDGE_MS = 42
MAX_REVIEW_WER = .20


def timed_words(segments, duration):
    """Use words at reliable Whisper segment boundaries on the recording clock."""
    words = []
    for row in segments:
        if not isinstance(row, dict):
            continue
        offsets = row.get("offsets") or {}
        start, end = offsets.get("from"), offsets.get("to")
        text = row.get("text")
        if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= duration or (words and start < words[-1][1])):
            continue
        parts = spoken_words(text)
        for index, word in enumerate(parts):
            words.append((end, start, word, index == 0, index == len(parts) - 1))
    return words


def marker_before(words, start, language):
    """Read an optional spoken marker preceding the matched phrase."""
    if start < 2 or words[start - 2][2] != MARKER[language]:
        return None
    marker = words[start - 1][2]
    if marker in NUMBERS[language]:
        return NUMBERS[language].index(marker) + 1
    if len(marker) == 1 and "1" <= marker <= "8":
        return int(marker)
    return None


def normalize_number(word, language):
    """Treat a spoken cardinal and its Whisper digit spelling alike."""
    if len(word) == 1 and "1" <= word <= "8":
        return NUMBERS[language][int(word) - 1]
    return word


def take_bounds(words, start, end, duration):
    """Pad the clip inside adjacent word boundaries and bound its duration."""
    if not words[start][3] or not words[end - 1][4]:
        return None
    left = max(0, words[start][1] - EDGE_MS, words[start - 1][0] if start else 0)
    right = min(duration, words[end - 1][0] + EDGE_MS, words[end][1] if end < len(words) else duration)
    return (left, right) if 350 <= right - left <= 18_000 else None


def find_takes(language, segments, duration):
    """Return unique exact-text matches, with no marker words included in their WAV."""
    words = timed_words(segments, duration)
    texts = [normalize_number(entry[2], language) for entry in words]
    found = []
    for number, prompt in enumerate(GUIDES[language], 1):
        expected = [normalize_number(word, language) for word in spoken_words(prompt.text)]
        matches = []
        for index in range(len(words) - len(expected) + 1):
            end = index + len(expected)
            if texts[index:end] != expected:
                continue
            marker = marker_before(words, index, language)
            if marker is not None and marker != number:
                continue
            bounds = take_bounds(words, index, end, duration)
            if bounds:
                left, right = bounds
                matches.append({"prompt": prompt.id, "number": number, "startMs": left, "endMs": right,
                                "basis": "number-and-text" if marker else "text"})
        if len(matches) == 1:
            found.append(matches[0])
    # Word intervals do not overlap, but the protective padding between adjacent
    # phrases is shared so a later consumer never sees duplicate material.
    found.sort(key=lambda item: item["startMs"])
    for left, right in zip(found, found[1:]):
        if left["endMs"] > right["startMs"]:
            split = (left["endMs"] + right["startMs"]) // 2
            left["endMs"] = right["startMs"] = split
    return found


def review_numbered_takes(language, segments, duration, exact):
    """Propose a probable line only within one explicitly numbered block."""
    words = timed_words(segments, duration)
    marks = [(start, marker) for start in range(2, len(words))
             if (marker := marker_before(words, start, language)) is not None]
    counts = Counter(number for _, number in marks)
    already_saved = {row["number"] for row in exact}
    suggestions = []
    for number_index, (start, number) in enumerate(marks):
        if number in already_saved or counts[number] != 1:
            continue
        end = marks[number_index + 1][0] - 2 if number_index < len(marks) - 1 else len(words)
        if start >= end or not (bounds := take_bounds(words, start, end, duration)):
            continue
        prompt = GUIDES[language][number - 1]
        expected = " ".join(normalize_number(word, language) for word in spoken_words(prompt.text))
        heard = " ".join(normalize_number(row[2], language) for row in words[start:end])
        distance = word_error_rate(expected, heard)
        if distance is not None and 0 < distance <= MAX_REVIEW_WER:
            left, right = bounds
            suggestions.append({"prompt": prompt.id, "number": number, "startMs": left, "endMs": right,
                                "basis": "review-number-and-text", "text": prompt.text, "heard": heard})
    return suggestions


def extract_reference_takes(audio, language, work, confirmed):
    """Decode an upload once and cut recognized matches without changing their waveform."""
    raw, pcm, whisper = (Path(work) / name for name in ("input", "input.wav", "whisper.wav"))
    raw.write_bytes(audio)
    normalize(raw, pcm, duration_limit=90)
    normalize(pcm, whisper, duration_limit=90, sample_rate=16000)
    segments = recognize_file(whisper, work / "word-times", language, words=True,
                              model=SMALL_MODEL if ready(SMALL_MODEL) else None)
    samples = read_pcm(pcm)
    duration = len(samples) * 500 // RATE
    exact = find_takes(language, segments, duration)
    suggestions = review_numbered_takes(language, segments, duration, exact)
    found = exact if confirmed is None else [row for row in suggestions if row["number"] in confirmed]
    if confirmed is not None and {row["number"] for row in found} != confirmed:
        raise ValueError("Границы изменились. Повтори поиск реплик.")
    for row in found:
        output = work / f"take-{row['number']}.wav"
        write_pcm(output, samples[row["startMs"] * RATE // 500:row["endMs"] * RATE // 500])
    return found, [row for row in suggestions if confirmed is None or row["number"] not in confirmed]
