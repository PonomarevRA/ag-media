"""Inspect local takes and rank a clean representative for reference-based TTS."""

from array import array
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from math import isfinite, log10, sqrt
from pathlib import Path
from statistics import median
import subprocess
import sys
import wave

from voice_paths import temporary_directory
from voice_studio.audio.pauses import quiet_spans
from voice_studio.mod_runtime import current_snapshot
from voice_studio.prosody.reference_script import spoken_words
from voice_studio.speech.speaker_review import extract_embeddings
from voice_studio.synthesis.prosody_alignment import RATE, contour_error, measured_contour, read_pcm, write_pcm

MAX_TAKES = 8
VERSION = "reference-enrollment-v1"
FRAME_MS = 20
FRAME_SAMPLES = RATE * FRAME_MS // 1000
MIN_ACTIVE_MS = 1700
MAX_REFERENCE_MS = 18_000
EMBEDDING_MS = 3600


def dbfs(value):
    """Convert mean-square PCM power to dBFS with a finite reporting floor."""
    return round(10 * log10(max(1.0, value) / 32768**2), 1)


def inspect_take(row, path):
    """Measure bounded PCM, quiet gaps and reliable pitch zones for a saved take."""
    pcm = read_pcm(path)
    if len(pcm) < RATE // 2 or len(pcm) % 2:
        raise ValueError("Incomplete reference PCM")
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    levels = []
    clipped = 0
    for start in range(0, len(samples), FRAME_SAMPLES):
        frame = samples[start:start + FRAME_SAMPLES]
        power = 0
        for sample in frame:
            power += sample * sample
            clipped += int(sample <= -32760 or sample >= 32760)
        levels.append(dbfs(power / len(frame)))
    sorted_levels = sorted(levels)
    loud = sorted_levels[9 * (len(levels) - 1) // 10]
    threshold = max(-53, min(-32, loud - 18))
    active = [index for index, level in enumerate(levels) if level >= threshold]
    duration_ms = len(samples) * 1000 // RATE
    active_ms = min(duration_ms, len(active) * FRAME_MS)
    clipping = clipped / len(samples)
    flags = []
    if active_ms < MIN_ACTIVE_MS:
        flags.append("short-speech")
    if duration_ms > MAX_REFERENCE_MS:
        flags.append("long-take")
    if clipping >= .001:
        flags.append("clipping")
    if loud < -38:
        flags.append("quiet")
    pitch = None if flags else measured_contour(path)
    pauses = [] if not active else quiet_spans(path, 200)
    result = {"id": row["id"], "prompt": row.get("reference_prompt"), "durationMs": duration_ms,
              "activeMs": active_ms, "speechDbfs": loud,
              "quietDbfs": sorted_levels[(len(levels) - 1) // 10],
              "clippedFraction": round(clipping, 5), "pauseCount": len(pauses),
              "pauseMs": round(sum(gap["end"] - gap["start"] for gap in pauses) * 1000),
              "pitchHz": pitch["pitchHz"] if pitch else None,
              "wordsPerMinute": round(len(spoken_words(row["text"])) * 60000 / duration_ms),
              "issues": flags}
    windows = []
    if not flags:
        # Select the most active excerpts from distinct halves of a long take.
        left, right = active[0] * FRAME_MS, min(duration_ms, (active[-1] + 1) * FRAME_MS)
        if right - left >= EMBEDDING_MS * 2:
            middle = left + ((right - left) // (2 * FRAME_MS)) * FRAME_MS
            windows = [best_window(levels, threshold, left, middle),
                       best_window(levels, threshold, middle, right)]
        else:
            windows = [best_window(levels, threshold, left, right)]
    return result, pcm, windows


def best_window(levels, threshold, start, end):
    """Choose the fixed-duration excerpt with the most active signal on the PCM clock."""
    length = EMBEDDING_MS // FRAME_MS
    first, last = start // FRAME_MS, end // FRAME_MS
    if last - first <= length:
        return start, end
    activity = [level >= threshold for level in levels[first:last]]
    count = sum(activity[:length])
    best = (count, first)
    for index in range(length, len(activity)):
        count += activity[index] - activity[index - length]
        if count > best[0]:
            best = (count, first + index - length + 1)
    return best[1] * FRAME_MS, (best[1] + length) * FRAME_MS


def unit(values):
    """Normalize finite nonzero features before cross-take cosine comparison."""
    if not values or not all(isfinite(value) for value in values):
        return None
    magnitude = sqrt(sum(value * value for value in values))
    return [value / magnitude for value in values] if magnitude else None


def mean_embedding(values):
    """Pool measurements only within one take and one embedding dimension."""
    vectors = [vector for value in values if (vector := unit(value)) is not None]
    if not vectors or len({len(vector) for vector in vectors}) != 1:
        return None
    return unit([sum(column) for column in zip(*vectors, strict=True)])


def batch_takes(rows, media):
    """Pack up to sixteen bounded PCM spans into one local encoder request."""
    takes, corpus, spans, seen = [], bytearray(), {}, set()
    for index, row in enumerate(rows):
        summary, pcm, windows = inspect_take(row, media / f"{row['id']}.wav")
        digest = sha256(pcm).digest()
        if digest in seen:
            summary["issues"].append("duplicate")
            windows = []
        seen.add(digest)
        takes.append(summary)
        if not windows:
            continue
        spans[index] = []
        for start, end in windows:
            offset = len(corpus) * 500 // RATE
            corpus.extend(pcm[start * RATE // 500:end * RATE // 500])
            spans[index].append((offset, offset + end - start))
    return takes, corpus, spans


def pair_scores(takes, vectors):
    """Expose comparable, uncalibrated cosine scores and each take's peer scores."""
    pairs, peers = [], defaultdict(list)
    for left, right in combinations(vectors, 2):
        a, b = vectors[left], vectors[right]
        if len(a) != len(b):
            continue
        score = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))
        peers[left].append(score)
        peers[right].append(score)
        pairs.append({"left": takes[left]["id"], "right": takes[right]["id"], "cosine": round(score, 3)})
    return pairs, peers


def check_references(rows, media):
    """Return signal quality, pairwise evidence and a neutral-preferred reference order."""
    try:
        takes, corpus, spans = batch_takes(rows, media)
    except (OSError, wave.Error, ValueError) as error:
        raise ValueError("Не удалось прочитать PCM выбранной записи.") from error
    vectors = {}
    if spans:
        with temporary_directory(prefix="voice-reference-") as folder:
            audio = Path(folder) / "samples.wav"
            write_pcm(audio, corpus)
            try:
                measured = extract_embeddings(audio, spans)
                vectors = {key: vector for key, values in measured.items()
                           if (vector := mean_embedding(values)) is not None}
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
    pairs, peers = pair_scores(takes, vectors)
    ranked = sorted(peers, key=lambda key: (-median(peers[key]), -min(peers[key]), key))
    # Let the medoid use an unmarked take. Prefer neutral only within comparable agreement.
    neutral = next((key for key in ranked if takes[key]["prompt"] in {"neutral-a", "neutral-b"}
                    and median(peers[key]) >= median(peers[ranked[0]]) - .04), None)
    if neutral is not None:
        ranked.remove(neutral)
        ranked.insert(0, neutral)
    ordered = [takes[key]["id"] for key in ranked]
    return {"version": VERSION, "metric": "speaker-embedding-cosine",
            "encoderFingerprint": current_snapshot().fingerprint(["speech.embedding", "speech.embedding-resource"]),
            "status": "compared" if peers else "insufficient-evidence",
            "takes": takes, "pairs": pairs, "referenceIds": ordered}


def delivery_comparison(original, generated):
    """Compare the same held-out text across human and generated recordings."""
    def duration(path):
        """Read the validated PCM duration in seconds without decoding the body."""
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()

    def pause_ms(path):
        """Count milliseconds in measured internal quiet spans."""
        return round(sum(gap["end"] - gap["start"] for gap in quiet_spans(path, 200)) * 1000)

    error = contour_error(measured_contour(original), measured_contour(generated))
    return {"durationRatio": round(duration(generated) / duration(original), 3),
            "pauseDifferenceMs": pause_ms(generated) - pause_ms(original),
            "pitchShapeSemitones": None if error is None else round(error, 2)}
