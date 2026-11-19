"""Compare the voice in matching, timed word windows from an original and a take."""

from pathlib import Path
from voice_paths import temporary_directory

from voice_studio.synthesis.engines import ffmpeg
from voice_studio.prosody.morpheme_graph import normalized
from voice_studio.synthesis.speech_metrics import speaker_cosine

MIN_CONTEXT_MS = 1500


def timed_range(graph, first, last):
    """Return the approximate span of a contiguous sequence of recognized words."""
    rows = graph["rows"]
    positions = [position for position, row in enumerate(rows)
                 if row["index"] is not None and first <= row["index"] <= last]
    if len(positions) != last - first + 1:
        return None
    span = rows[positions[0]:positions[-1] + 1]
    for row in span:
        if row["kind"] != "match" or row["heard"]["startMs"] is None:
            return None
    start, end = span[0]["heard"]["startMs"], span[-1]["heard"]["endMs"]
    if end - start >= MIN_CONTEXT_MS:
        return start, end
    return None


def matching_windows(candidate, source, index, source_offset):
    """Find the smallest shared context long enough for an experimental score."""
    candidate_words = {row["index"]: row["expected"] for row in candidate["rows"] if row["index"] is not None}
    source_words = {row["index"]: row["expected"] for row in source["rows"] if row["index"] is not None}
    for span in range(5):
        for before in sorted(range(span + 1), key=lambda value: abs(2 * value - span)):
            first, last = index - before, index + span - before
            for position in range(first, last + 1):
                left, right = candidate_words.get(position), source_words.get(position + source_offset)
                if not left or not right or normalized(left) != normalized(right):
                    break
            else:
                candidate_range = timed_range(candidate, first, last)
                original_range = timed_range(source, first + source_offset, last + source_offset)
                if candidate_range and original_range:
                    label = " ".join(candidate_words[position] for position in range(first, last + 1))
                    return candidate_range, original_range, label
    return None


def score_windows(root, candidate, original, windows):
    """Cut both windows into temporary WAVs and run the local speaker encoder."""
    candidate_range, original_range, label = windows
    with temporary_directory() as folder:
        cuts = []
        for index, (source, (start, end)) in enumerate(((candidate, candidate_range), (original, original_range))):
            target = Path(folder) / f"clip-{index}.wav"
            ffmpeg(["-ss", str(start / 1000), "-i", str(source), "-t", str((end - start) / 1000), str(target)])
            cuts.append(target)
        return {"cosine": speaker_cosine(cuts[1], cuts[0]), "text": label,
                "originalRangeMs": original_range, "candidateRangeMs": candidate_range}
