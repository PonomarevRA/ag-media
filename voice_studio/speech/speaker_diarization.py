"""Run local speaker segmentation and label timed Whisper words."""

from voice_studio.mod_runtime import model_resource, worker_environment

from voice_studio.mod_runtime import model_provider

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from voice_paths import model_path
from voice_studio.core.media_resources import stage_threads
from voice_studio.core.worker_process import worker_command

MODEL_PATH = model_path("diarization", "segmentation.onnx").parent


def ready():
    """Check whether the local inference package and both models are installed."""
    if not importlib.util.find_spec('sherpa_onnx') or not (MODEL_PATH / 'segmentation.onnx').is_file():
        return False
    embedding = model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', MODEL_PATH / 'embedding.onnx')
    return embedding.is_file()


@model_provider('speech.diarization', 'builtin.pyannote-segmentation')
def detect_turns(wav, speaker_count, *, merge_similar=False):
    """Isolate native inference in a child process and return millisecond intervals."""
    if not ready():
        raise ValueError("Локальная модель разделения говорящих недоступна.")
    try:
        output = subprocess.check_output(
            worker_command(sys.executable, "voice_studio.speech.speaker_diarization", wav, speaker_count or -1,
                           "0.72" if merge_similar else "0.50"),
            env=worker_environment(), stderr=subprocess.DEVNULL, timeout=180,
        )
        return json.loads(output)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("Не удалось разделить голоса в этой записи.") from exc


def infer(wav, count, threshold=.50):
    """Extract voice turns using Sherpa, with speaker IDs ordered by first appearance."""
    import sherpa_onnx as sherpa
    from array import array
    import wave

    threads = stage_threads(2)
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(str(MODEL_PATH / "segmentation.onnx")),
            num_threads=threads,
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(str(model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', MODEL_PATH / 'embedding.onnx')), num_threads=threads),
        clustering=sherpa.FastClusteringConfig(num_clusters=count, threshold=threshold),
    )
    with wave.open(str(wav)) as audio:
        pcm = array("h", audio.readframes(audio.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
        samples = array("f", (sample / 32768 for sample in pcm))
    speaker_ids = {}
    turns = []
    for turn in sherpa.OfflineSpeakerDiarization(config).process(samples).sort_by_start_time():
        position = speaker_ids.setdefault(turn.speaker, len(speaker_ids) + 1)
        start, end = round(turn.start * 1000), round(turn.end * 1000)
        if start < end:
            turns.append({"startMs": start, "endMs": end, "speaker": position})
    return turns


def speaker_for(turns, start, end):
    """Choose the strongest temporal match and retain simultaneous speech as a flag."""
    coverage = {}
    active = []
    simultaneous = False
    for turn in turns:
        left, right = max(start, turn["startMs"]), min(end, turn["endMs"])
        if left >= right:
            continue
        speaker = turn["speaker"]
        coverage[speaker] = coverage.get(speaker, 0) + (right - left)
        if any(other != speaker and min(right, stop) > max(left, begin) for begin, stop, other in active):
            simultaneous = True
        active.append((left, right, speaker))
    ordered = sorted(coverage, key=coverage.get, reverse=True)
    if not ordered or coverage[ordered[0]] * 2 < end - start:
        return None, simultaneous
    # An even split between adjacent turns is ambiguous but need not be simultaneous.
    if len(ordered) > 1 and coverage[ordered[0]] == coverage[ordered[1]]:
        return None, simultaneous
    return ordered[0], simultaneous


def voiced_window(phrase, turns, speaker):
    """Find clean voiced coverage inside a phrase and preserve the strongest turn."""
    text = phrase["text"].lstrip()
    if text.startswith(("- ", "– ", "— ")) or "\n" in text:
        return None, None
    start, end = phrase["startMs"], phrase["endMs"]
    windows = [(max(start, row["startMs"]), min(end, row["endMs"]), row["speaker"])
               for row in turns if row["startMs"] < end and start < row["endMs"]]
    if not windows or len({number for _, _, number in windows}) != 1:
        return None, None
    first, last, number = max(windows, key=lambda row: row[1] - row[0])
    voiced = sum(right - left for left, right, _ in windows)
    if last - first < 650 or (speaker is None and voiced < .22 * (end - start)):
        return None, None
    return number, {"startMs": first, "endMs": last}


def label_phrases(phrases, turns):
    """Attribute intact Whisper phrases without counting long silent padding as speech."""
    labeled = []
    for phrase in phrases:
        speaker, overlap = speaker_for(turns, phrase["startMs"], phrase["endMs"])
        item = {**phrase, "speaker": speaker, "overlap": overlap}
        if not overlap:
            candidate, window = voiced_window(phrase, turns, speaker)
            if window:
                item["speakerAudioWindow"] = window
                if speaker is None:
                    item.update(speaker=candidate, speakerBasis="single-voiced-turn")
        labeled.append(item)
    return labeled


def label_words(words, turns):
    """Group consecutive Whisper words into readable cues without losing the transcript."""
    cues = []
    first = 0
    for word in words:
        start, end, text = word["startMs"], word["endMs"], word["text"]
        while first < len(turns) and turns[first]["endMs"] <= start:
            first += 1
        last = first
        while last < len(turns) and turns[last]["startMs"] < end:
            last += 1
        speaker, overlap = speaker_for(turns[first:last], start, end)
        if cues:
            previous = cues[-1]
            if (previous["speaker"] == speaker and previous["overlap"] == overlap
                    and start - previous["endMs"] <= 700 and end - previous["startMs"] <= 6500):
                joiner = "" if text[0] in ",.!?…;:%)]»" else " "
                previous["text"] += joiner + text
                previous["endMs"] = end
                continue
        cues.append({**word, "speaker": speaker, "overlap": overlap})
    return cues


if __name__ == "__main__":
    print(json.dumps(infer(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else .50)))
