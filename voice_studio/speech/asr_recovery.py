"""Retry structurally unreliable clip transcripts on their original PCM clock."""

from bisect import bisect_right
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata
import wave

from voice_studio.audio.pauses import quiet_spans
from voice_studio.core.analysis_stages import ignore_progress, stage
from voice_studio.speech.recognition import MODEL, RECOVERY_MODEL, ready, run_whisper

TARGET_WINDOW_MS = 8_000
MIN_WINDOW_MS = 5_000
MAX_WINDOW_MS = 10_000
MAX_WINDOWS = 12
MAX_WINDOWS_AUDIO_MS = 100_000
SCRIPT_HINTS = {
    "ar": "ARABIC", "ur": "ARABIC", "ps": "ARABIC",
    "bn": "BENGALI", "hi": "DEVANAGARI",
}
LOOP = re.compile(r"(.{1,8}?)\1{7,}")


def words(text):
    """Preserve combining marks so Bengali vowels stay attached to their words."""
    text = unicodedata.normalize("NFC", text.casefold())
    return " ".join(
        "".join(ch if ch.isalnum() or unicodedata.category(ch).startswith("M") else " " for ch in text).split()
    )


def script_counts(text, native):
    """Count native and Latin letters separately from unexpected writing systems."""
    counts = [0, 0, 0]
    for char in text:
        if not char.isalpha():
            continue
        script = unicodedata.name(char, "").split(" ", 1)[0]
        if script == native:
            counts[0] += 1
        elif script == "LATIN":
            counts[1] += 1
        else:
            counts[2] += 1
    return counts


def unreliable(text, normalized, native):
    """Identify broken encoding, sustained loops, and predominantly wrong script."""
    if "\ufffd" in text or not normalized:
        return True
    tokens = normalized.split()
    if len(tokens) >= 6 and len(set(tokens)) * 4 <= len(tokens):
        return True
    compact = normalized.replace(" ", "")
    match = LOOP.search(compact)
    if match and (len(match[0]) >= 36 or len(match[0]) * 2 >= len(compact)):
        return True
    if native:
        own, latin, other = script_counts(normalized, native)
        return other >= 6 and other >= 2 * max(own, latin, 1)
    return False


def retry_spans(cues, language):
    """Group faulty contiguous cues without including unrelated valid neighbors."""
    native = SCRIPT_HINTS.get(language)
    normalized = [words(cue["text"]) for cue in cues]
    indices = {i for i, cue in enumerate(cues) if unreliable(cue["text"], normalized[i], native)}
    index = 0
    while index < len(cues):
        first = index
        reference = normalized[first]
        index += 1
        if len(reference.split()) < 3 or cues[first]["endMs"] - cues[first]["startMs"] > 3200:
            continue
        while index < len(cues):
            cue = cues[index]
            if (cue["startMs"] - cues[index - 1]["endMs"] > 125
                    or cue["endMs"] - cue["startMs"] > 3200
                    or SequenceMatcher(None, reference[:256], normalized[index][:256]).ratio() < .78):
                break
            index += 1
        if index - first >= 5 and cues[index - 1]["endMs"] - cues[first]["startMs"] >= 8000:
            indices.update(range(first, index))
    # A repeating loop can alternate short captions on the decoder's exact 2 s grid.
    first = 0
    while first < len(cues):
        last = first
        while last < len(cues):
            start, end = cues[last]["startMs"], cues[last]["endMs"]
            if start % 2000 or end - start != 2000 or (last > first and start != cues[last - 1]["endMs"]):
                break
            last += 1
        if last - first >= 5 and len(set(normalized[first:last])) * 2 <= last - first + 1:
            indices.update(range(first, last))
        first = max(first + 1, last)
    spans = []
    for index in sorted(indices):
        start, end = cues[index]["startMs"], cues[index]["endMs"]
        if spans and start - spans[-1][1] <= 125 and index - 1 in indices:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans, indices


def window_plan(spans, pauses):
    """Split retries around measured pauses where possible, subject to a clip budget."""
    windows = []
    spent = 0
    for start, end in spans:
        while start < end:
            stop = end
            if stop - start > MAX_WINDOW_MS:
                remaining = end - start
                target = start + min(TARGET_WINDOW_MS, remaining // 2)
                left = bisect_right(pauses, max(start + MIN_WINDOW_MS - 1, target - 1300))
                right = bisect_right(pauses, min(start + MAX_WINDOW_MS, end - MIN_WINDOW_MS, target + 1300))
                stop = min(pauses[left:right], key=lambda value: abs(value - target), default=target)
            if len(windows) >= MAX_WINDOWS or spent + stop - start > MAX_WINDOWS_AUDIO_MS:
                return windows
            windows.append((start, stop))
            spent += stop - start
            start = stop
    return windows


def extract_windows(wav, work, windows):
    """Store only the WAV ranges required for retry, leaving the original untouched."""
    jobs = []
    with wave.open(str(wav)) as audio:
        rate = audio.getframerate()
        for index, (start, end) in enumerate(windows):
            left, right = start * rate // 1000, end * rate // 1000
            audio.setpos(left)
            path = work / f"asr-retry-{index}.wav"
            with wave.open(str(path), "wb") as excerpt:
                excerpt.setparams(audio.getparams())
                excerpt.writeframes(audio.readframes(right - left))
            jobs.append((path, path.with_suffix("")))
    return jobs


def accepted_cues(report, start, end, native, stronger):
    """Project reliable local text to the media clock and discard padding timestamps."""
    cursor = start
    language = (report.get("result") or {}).get("language")
    result = []
    for row in report["transcription"]:
        text = row["text"].strip()
        if text.endswith("\ufffd"):
            # A final incomplete token must not invalidate a preceding full sentence.
            text = text.rpartition(" ")[0].strip()
        offsets = row.get("offsets") or {}
        left, right = offsets.get("from"), offsets.get("to")
        if not all(type(boundary) is int for boundary in (left, right)):
            continue
        left, right = max(cursor, start + left), min(end, start + right)
        if right - left < 120:
            continue
        normalized = words(text)
        if unreliable(text, normalized, native):
            continue
        if native:
            own, latin, other = script_counts(normalized, native)
            if other > own or (not own and not (language == "en" and latin >= 4)):
                continue
            # Same-family models tend to hallucinate English when a native-language
            # cue broke. Accept English in those retries only from the stronger model.
            if not own and not stronger:
                continue
        result.append({"startMs": left, "endMs": right, "text": text})
        cursor = right
    return result


def decode_windows(jobs, language, model, timeout, on_window):
    """Start one local process for the prepared windows, with independent text context."""
    if not jobs:
        return []
    try:
        options = ["-mc", "0", "-bs", "1", "-bo", "1", "-nf"]
        return run_whisper(jobs, language, options, model, 4, timeout, 12, on_window=on_window)
    except ValueError:
        return []


def review_windows(jobs, windows, language, model, native, stronger, on_prefix, emit, *, health=None):
    """Report completed inputs without restarting Whisper or changing batch boundaries."""
    candidates = {}
    total = len(jobs)

    def publish_progress(completed, count, phase):
        """Publish the counter for the current pass of recovery."""
        emit({"stage": "whisper-recovery", "state": "running", "windowsDone": completed,
              "windowsTotal": count, "recoveryPass": phase})

    def decode_batch(batch, batch_language, timeout, completed, count, phase):
        """Keep a running total even when a later batch only partially completes."""
        def advance(finished):
            publish_progress(completed + finished, count, phase)

        reports = decode_windows(batch, batch_language, model, timeout, advance)
        if health is not None and (len(reports) != len(batch) or any(report is None for report in reports)):
            health["incomplete"] = True
        return reports

    if total:
        publish_progress(0, total, "primary")
    first = min(2, total) if on_prefix and total > 2 else 0
    for begin, end in ((0, first), (first, total)):
        reports = decode_batch(jobs[begin:end], language, 125, len(candidates), total, "primary")
        for index, report in enumerate(reports, begin):
            if report is not None:
                candidates[index] = accepted_cues(report, *windows[index], native, stronger)
        if begin == 0 and end == first and first:
            prefix = [row for rows in candidates.values() for row in rows]
            if prefix:
                on_prefix(prefix)
    primary_done = len(candidates)
    if stronger and native:
        # An English-only span can fail when forced through the primary language.
        retry = [index for index, rows in candidates.items() if not rows][:3]
        alternates = [(jobs[index][0], jobs[index][1].with_name(f"asr-auto-{index}")) for index in retry]
        if alternates:
            publish_progress(0, len(alternates), "alternate")
            for index, report in zip(retry, decode_batch(alternates, "auto", 40, 0, len(alternates), "alternate")):
                if report is not None:
                    candidates[index] = accepted_cues(report, *windows[index], native, stronger)
    return candidates, primary_done


def recover_clip(wav, work, cues, language, *, emit=ignore_progress, on_prefix=None):
    """Return the original clean cues plus selective repairs and unresolved time ranges."""
    spans, indices = retry_spans(cues, language)
    if not spans:
        return cues, None
    native = SCRIPT_HINTS.get(language)
    stronger = ready(RECOVERY_MODEL)
    model = RECOVERY_MODEL if stronger else MODEL
    retry_language = language if native else "auto"
    diagnostics = {"model": "whisper-turbo" if stronger else "whisper-base", "language": retry_language,
                   "retried": [], "unresolved": []}
    unresolved = diagnostics["unresolved"]
    recovered = []
    with stage(emit, "whisper-recovery") as progress:
        progress["detail"] = diagnostics["model"]
        # Use measured quiet boundaries when available, without relying on ASR to choose them.
        try:
            pauses = sorted(round((item["start"] + item["end"]) * 500) for item in quiet_spans(wav))
            windows = window_plan(spans, pauses)
            jobs = extract_windows(wav, Path(work), windows)
        except (OSError, ValueError, wave.Error):
            windows, jobs = [], []
        health = {}
        candidates, primary_done = review_windows(jobs, windows, retry_language, model, native, stronger, on_prefix, emit,
                                                  health=health)
        if jobs:
            progress.update(windowsDone=primary_done, windowsTotal=len(jobs), recoveryPass="primary")
        for index, (start, end) in enumerate(windows):
            additions = candidates.get(index, ())
            recovered.extend(additions)
            diagnostics["retried"].append({"startMs": start, "endMs": end, "cues": len(additions)})
        if not jobs:
            progress["state"] = "skipped"
        elif primary_done < len(jobs):
            progress["state"] = "partial" if primary_done else "failed"
        elif health.get("incomplete"):
            progress["state"] = "partial"
        # Distinguish a completed but uncertain decode from a timed-out worker.
        # An incremental caller may reuse the former, but must retry the latter.
        diagnostics.update(status=progress["state"], windowsDone=primary_done, windowsTotal=len(jobs))
    cursor = 0
    for start, end in spans:
        while cursor < len(recovered) and recovered[cursor]["endMs"] <= start:
            cursor += 1
        for cue in recovered[cursor:]:
            if cue["startMs"] >= end:
                break
            if cue["startMs"] - start >= 1200:
                unresolved.append({"startMs": start, "endMs": cue["startMs"]})
            start = cue["endMs"]
        if end - start >= 1200:
            unresolved.append({"startMs": start, "endMs": end})
    fixed = [cue for index, cue in enumerate(cues) if index not in indices]
    fixed.extend(recovered)
    fixed.sort(key=lambda cue: cue["startMs"])
    return fixed, diagnostics
