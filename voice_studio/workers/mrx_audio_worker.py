"""Run the pinned MERL three-source model in the existing TorchAudio environment."""

from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

RATE = 44_100
WINDOW = 6 * RATE
OVERLAP = RATE
# The order is the SOURCE_NAMES order used to train this checkpoint.
STEMS = ("music", "dialogue", "effects")


def write_tracks(model, recording, tracks):
    """Crossfade bounded model windows and write exactly one sample per input sample."""
    tail = None
    ramp = np.linspace(0, 1, OVERLAP, dtype=np.float32)[None, :, None]
    for start in range(0, len(recording), WINDOW - OVERLAP):
        recording.seek(start)
        audio = recording.read(WINDOW, dtype="float32", always_2d=True)
        # Torch's centered STFT needs more than half the longest FFT window.
        padded = np.pad(audio, ((0, max(0, 8192 - len(audio))), (0, 0)))
        waves = model(torch.from_numpy(padded.T.copy())).numpy()
        waves = waves.transpose(0, 2, 1)[:, :len(audio)]
        if not np.isfinite(waves).all():
            raise ValueError("MRX returned non-finite samples.")
        if tail is not None:
            waves[:, :OVERLAP] = tail * (1 - ramp) + waves[:, :OVERLAP] * ramp
        last = start + len(audio) >= len(recording)
        for track, samples in zip(tracks, waves):
            track.write(samples if last else samples[:-OVERLAP])
        if last:
            break
        tail = waves[:, -OVERLAP:].copy()


def separate(source, installed, output, threads):
    """Load the local checkpoint once and publish float WAVs on the same clock."""
    sys.path.insert(0, str(installed))
    from mrx import MRX

    torch.set_num_threads(threads)
    model = MRX().eval()
    model.load_state_dict(torch.load(installed / "default_mrx_pre_trained_weights.pth",
                                     map_location="cpu", weights_only=True))
    with sf.SoundFile(source) as recording, torch.inference_mode(), ExitStack() as files:
        if recording.samplerate != RATE or recording.channels != 2 or not len(recording):
            raise ValueError("Expected nonempty 44.1 kHz stereo PCM.")
        tracks = [files.enter_context(sf.SoundFile(output / f"{stem}.wav", "w", RATE, 2, subtype="FLOAT"))
                  for stem in STEMS]
        write_tracks(model, recording, tracks)


if __name__ == "__main__":
    source, installed, output, threads = sys.argv[1:]
    separate(Path(source), Path(installed), Path(output), int(threads))
