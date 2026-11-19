"""Run the local Sherpa-ONNX audio tagger on overlapping PCM windows."""

from voice_studio.mod_runtime import model_provider

from array import array
import json
from pathlib import Path
import subprocess
import sys
import wave

from voice_paths import model_path
from voice_studio.core.worker_process import worker_command
from voice_studio.core.optional_runtime import runtime_environment, sherpa_ready

MODEL_DIR = model_path("audio-tagging", "zipformer-small-int8", "model.int8.onnx").parent
MODEL = MODEL_DIR / "model.int8.onnx"
LABELS = MODEL_DIR / "class_labels_indices.csv"
WINDOW_MS = 1500
HOP_MS = 750
BASIS = "zipformer-small-int8-2024-04-15"
# Use AudioSet identifiers from the bundled label index, including classes with compound names.
CLASS_MAP = {
    137: "music", 16: "laughter", 47: "cough", 67: "applause", 69: "crowd",
    75: "dog_bark", 111: "bird", 112: "bird", 127: "cricket_chirp",
    283: "wind", 285: "wind", 289: "rain", 290: "rain", 291: "rain",
    327: "traffic_noise", 359: "knock", 384: "typing_computer_keyboard",
    386: "typing_computer_keyboard", 396: "siren", 397: "siren",
}


@model_provider('audio.classification', 'builtin.zipformer')
def classify_zipformer(wav, duration_ms):
    """Isolate the native tagger so a runtime error cannot stop transcription."""
    if duration_ms < WINDOW_MS:
        return [], "too-short"
    if not MODEL.is_file() or not LABELS.is_file() or not sherpa_ready():
        return [], "unavailable"
    try:
        raw = subprocess.check_output(
            worker_command(sys.executable, "voice_studio.audio.zipformer_audio", wav),
            timeout=max(15, duration_ms / 1000), env=runtime_environment("sherpa"), stderr=subprocess.DEVNULL,
        )
        return json.loads(raw), "done"
    except (OSError, ValueError, subprocess.SubprocessError):
        return [], "failed"


def window_tags(wav):
    """Return scored events using one CPU session and a bounded audio buffer."""
    import sherpa_onnx as sherpa

    model = sherpa.AudioTaggingModelConfig(
        zipformer=sherpa.OfflineZipformerAudioTaggingModelConfig(str(MODEL)),
    )
    config = sherpa.AudioTaggingConfig(model=model, labels=str(LABELS))
    tagger = sherpa.AudioTagging(config)
    with wave.open(str(wav)) as audio:
        rate = audio.getframerate()
        if rate != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Zipformer expects 16 kHz mono PCM16.")
        size = rate * WINDOW_MS // 1000
        hop = rate * HOP_MS // 1000
        frames = audio.getnframes()
        for start in range(0, frames - size + hop, hop):
            audio.setpos(start)
            pcm = array("h", audio.readframes(size))
            if sys.byteorder != "little":
                pcm.byteswap()
            samples = [value / 32768 for value in pcm]
            samples.extend([0.0] * (size - len(pcm)))
            stream = tagger.create_stream()
            stream.accept_waveform(rate, samples)
            scores = {}
            # Music and background sounds can coexist with a higher scoring speech class.
            for event in tagger.compute(stream, top_k=527):
                label = CLASS_MAP.get(event.index)
                if label and event.prob >= .15:
                    scores[label] = max(scores.get(label, 0), event.prob)
            yield {"start": start / rate, "end": (start + len(pcm)) / rate,
                   "scores": [{"label": label, "score": score} for label, score in scores.items()]}


if __name__ == "__main__":
    json.dump(list(window_tags(sys.argv[1])), sys.stdout)
