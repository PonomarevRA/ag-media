"""Choose a continuous sequence of whole TTS phrases and smooth the PCM joins."""

from array import array
from dataclasses import dataclass
from math import log10, log2
from pathlib import Path
from statistics import median
import subprocess
import sys
import wave

from voice_studio.audio.speech_intonation import measure_intonation
from voice_studio.speech.speaker_review import extract_embeddings
from voice_studio.synthesis.prosody_alignment import RATE, read_pcm, write_pcm
from voice_studio.synthesis.reference_enrollment import FRAME_MS, FRAME_SAMPLES, best_window, mean_embedding
from voice_studio.synthesis.speech_metrics import word_error_rate

TAKES = 2
MAX_PHRASES = 4
MAX_WER = 1 / 3
MAX_GAIN_DB = 3.0
PEAK_LIMIT = 29200
FADE = RATE * 6 // 1000


@dataclass
class Take:
    """A full phrase candidate and the acoustic estimates used during selection."""

    path: Path
    number: int
    wer: float
    level: float
    pitch: float | None
    vector: list[float] | None = None


def samples_from_pcm(pcm):
    """Decode little-endian samples without depending on host byte order."""
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def activity(samples):
    """Measure active-window RMS and choose a robust background threshold."""
    levels = []
    for index in range(0, len(samples), FRAME_SAMPLES):
        block = samples[index:index + FRAME_SAMPLES]
        power = sum(sample * sample for sample in block) / len(block)
        levels.append(10 * log10(max(1.0, power) / 32768**2))
    if not levels:
        raise ValueError("Синтез вернул пустую фразу.")
    strongest = sorted(levels)[9 * (len(levels) - 1) // 10]
    threshold = max(-53, min(-32, strongest - 18))
    active = [level for level in levels if level >= threshold]
    if not active:
        raise ValueError("В фразе нет достаточного уровня сигнала.")
    return levels, threshold, median(active)


def affinity(left, right):
    """Use normalized embeddings from a single frozen extraction run."""
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


def join_cost(left, right):
    """Penalize changes in timbre and level, with a small voiced-register term."""
    voice = 0.60 * (1 - affinity(left.vector, right.vector))
    level = 0.24 * min(1.0, abs(left.level - right.level) / 12)
    pitch = 0.0
    if left.pitch and right.pitch:
        pitch = 0.08 * min(1.0, abs(log2(left.pitch / right.pitch)) * 2)
    return voice + level + pitch


def choose_sequence(groups, anchor):
    """Minimize the total anchored and adjacent cost while keeping phrase order."""
    def target(take):
        """Express distance to the requested voice by one cosine cost."""
        return 1 - affinity(anchor, take.vector)

    scores = [target(take) for take in groups[0]]
    parents = []
    for previous, group in zip(groups, groups[1:]):
        row = []
        for take in group:
            cost, index = min((score + join_cost(left, take), index)
                              for index, (score, left) in enumerate(zip(scores, previous, strict=True)))
            row.append((cost + target(take), index))
        scores = [cost for cost, _ in row]
        parents.append([index for _, index in row])
    cost, index = min((score, index) for index, score in enumerate(scores))
    selected = [groups[-1][index]]
    for group, parent in zip(reversed(groups[:-1]), reversed(parents), strict=True):
        index = parent[index]
        selected.append(group[index])
    return selected[::-1], cost


def measure_candidates(phrases, folder, render, recognize):
    """Generate independent takes and keep the most faithful recognized wording per phrase."""
    groups = []
    for index, (text, _) in enumerate(phrases):
        takes = []
        for number in range(TAKES):
            path = folder / f"sequence-{index}-{number}.wav"
            try:
                render(text, path, index, number)
                wer = word_error_rate(text, recognize(path))
                if wer is None or wer > MAX_WER:
                    continue
                pcm = read_pcm(path)
                _, _, level = activity(samples_from_pcm(pcm))
                rows = measure_intonation(path, [{"startMs": 0, "endMs": len(pcm) * 500 // RATE}], 0)
                pitch = median(rows[0]["pitchHz"]) if rows else None
                takes.append(Take(path, number, wer, level, pitch))
            except (OSError, ValueError, wave.Error, subprocess.SubprocessError):
                continue
        if not takes:
            raise ValueError(f"Whisper не подтвердил слова фразы №{index + 1}.")
        best = min(take.wer for take in takes)
        groups.append([take for take in takes if take.wer == best])
    return groups


def fill_embeddings(reference, groups, folder):
    """Batch the most active excerpts with the profile reference in one encoder call."""
    takes = [take for group in groups for take in group]
    audio, spans = bytearray(), {}
    for index, path in enumerate([reference, *(take.path for take in takes)]):
        pcm = read_pcm(path)
        levels, threshold, _ = activity(samples_from_pcm(pcm))
        left, right = best_window(levels, threshold, 0, len(pcm) * 500 // RATE)
        offset = len(audio) * 500 // RATE
        audio.extend(pcm[left * RATE // 500:right * RATE // 500])
        spans[index] = [(offset, offset + right - left)]
    packed = folder / "sequence-embeddings.wav"
    write_pcm(packed, audio)
    vectors = extract_embeddings(packed, spans)
    anchor = mean_embedding(vectors.get(0, []))
    if anchor is None:
        raise ValueError("Недостаточно голосовых данных в основном образце.")
    for index, take in enumerate(takes, 1):
        vector = mean_embedding(vectors.get(index, []))
        if vector is not None and len(vector) == len(anchor):
            take.vector = vector
    filtered = [[take for take in group if take.vector is not None] for group in groups]
    if not all(filtered):
        raise ValueError("Энкодер не смог сравнить голоса во всех фразах.")
    return filtered, anchor


def smooth_take(pcm, gain_db):
    """Apply bounded gain and edge fades without shifting pitch, time or speech boundaries."""
    values = samples_from_pcm(pcm)
    peak = max(map(abs, values))
    gain = min(10 ** (gain_db / 20), PEAK_LIMIT / peak if peak else 1)
    fade = min(FADE, len(values) // 2)
    for index, sample in enumerate(values):
        edge = min(index, len(values) - index - 1)
        factor = edge / fade if edge < fade else 1
        values[index] = round(sample * gain * factor)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes(), 20 * log10(gain)


def assemble_sequence(selected, pauses, output, anchor):
    """Assemble selected takes with their original durations and planned punctuation pauses."""
    target = median(take.level for take in selected)
    report = []
    with wave.open(str(output), "wb") as writer:
        writer.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        for take, gap in zip(selected, pauses, strict=True):
            level_step = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, target - take.level))
            pcm, applied = smooth_take(read_pcm(take.path), level_step)
            writer.writeframesraw(pcm)
            writer.writeframesraw(bytes(gap * RATE // 500))
            report.append({"take": take.number + 1, "wordErrorRate": take.wer,
                           "voiceCosine": round(affinity(anchor, take.vector), 3),
                           "gainDb": round(applied, 2), "inputLevelDbfs": round(take.level, 2),
                           "pauseAfterMs": gap})
    return report


def render_voice_sequence(phrases, reference, folder, render, recognize):
    """Produce the reviewed optional variant with a bounded candidate grid."""
    if not 2 <= len(phrases) <= MAX_PHRASES:
        raise ValueError("Для подбора фраз введи от двух до четырёх предложений.")
    groups = measure_candidates(phrases, folder, render, recognize)
    groups, anchor = fill_embeddings(reference, groups, folder)
    selected, cost = choose_sequence(groups, anchor)
    output = folder / "voice-sequence.wav"
    report = assemble_sequence(selected, [gap for _, gap in phrases], output, anchor)
    return output, {"basis": "speaker-and-join-cost-v1", "cost": round(cost, 4), "phrases": report}
