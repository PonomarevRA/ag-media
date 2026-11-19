"""Extract the accompaniment of a stereo WAV using the installed HDemucs weights."""

import argparse
from pathlib import Path
from voice_paths import publish_file, temporary_directory

import numpy as np
import soundfile as sf
import torch
from torchaudio.models import hdemucs_high

RATE = 44_100
WINDOW = 8 * RATE
OVERLAP = RATE


def accompaniment(model, audio, destination):
    """Write a sample-aligned stereo bed using bounded windows and a linear crossfade."""
    tail = None
    ramp = np.linspace(0, 1, OVERLAP, dtype=np.float32)[:, None]
    for start in range(0, len(audio), WINDOW - OVERLAP):
        audio.seek(start)
        chunk = audio.read(WINDOW, dtype="float32")
        # Real audio near EOF stays inside the model window without changing the sample clock.
        padded = np.pad(chunk, ((0, WINDOW - len(chunk)), (0, 0)))
        separated = model(torch.from_numpy(padded.T.copy())[None])[0]
        bed = separated[:3].sum(dim=0).T.numpy()[:len(chunk)]
        if not np.isfinite(bed).all():
            raise ValueError("HDemucs returned non-finite samples.")
        if tail is not None:
            size = len(tail)
            bed[:size] = tail * (1 - ramp[:size]) + bed[:size] * ramp[:size]
        if start + WINDOW >= len(audio):
            destination.write(bed)
            break
        destination.write(bed[:-OVERLAP])
        tail = bed[-OVERLAP:].copy()


def separate(source, checkpoint, output):
    """Load local weights and separate a stereo 44.1 kHz file without web access."""
    torch.set_num_threads(4)
    # Construct the same topology as the bundle without invoking its remote downloader.
    network = hdemucs_high(sources=["drums", "bass", "other", "vocals"]).eval()
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with temporary_directory() as scratch:
        staged = Path(scratch) / "background.wav"
        with sf.SoundFile(source) as recording:
            if recording.samplerate != RATE or recording.channels != 2 or not len(recording):
                raise ValueError("Expected nonempty stereo audio at 44.1 kHz.")
            with torch.inference_mode(), sf.SoundFile(staged, "w", RATE, 2, subtype="FLOAT") as destination:
                accompaniment(network, recording, destination)
        publish_file(staged, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    separate(args.source, args.checkpoint, args.output)
