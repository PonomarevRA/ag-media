"""Measure the cosine of normalized speaker embeddings using the fixed base XTTS encoder."""

from voice_studio.mod_runtime import model_resource

import json
import sys
from zipfile import is_zipfile


def cosine(weights, original, candidate):
    """Return the cosine in [-1, 1] for the first 30 seconds of two local WAVs.

    Args:
        weights: Local XTTS checkpoint containing the speaker encoder.
        original: Path to the reference WAV.
        candidate: Path to the synthesized WAV.

    Returns:
        Cosine similarity of the two normalized embeddings.

    Loading and inference errors propagate.
    """
    weights = str(model_resource('speech.metric-resource', 'builtin.xtts-speaker-encoder', weights))
    import torch
    from trainer.io import load_fsspec
    from TTS.encoder.models.resnet import ResNetSpeakerEncoder
    from TTS.tts.models.xtts import load_audio

    torch.set_num_threads(4)
    encoder = ResNetSpeakerEncoder(
        input_dim=64, proj_dim=512, log_input=True, use_torch_spec=True,
        audio_config={"fft_size": 512, "win_length": 400, "hop_length": 160,
                      "sample_rate": 16000, "preemphasis": .97, "num_mels": 64},
    )
    # PyTorch ZIP checkpoints can map tensor storage without materializing unrelated XTTS weights.
    if is_zipfile(weights):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)["model"]
    else:
        checkpoint = load_fsspec(weights, map_location="cpu")["model"]
    prefix = "hifigan_decoder.speaker_encoder."
    encoder.load_state_dict({key.removeprefix(prefix): tensor for key, tensor in checkpoint.items()
                             if key.startswith(prefix)})
    del checkpoint
    encoder.eval()
    with torch.inference_mode():
        embeddings = [encoder(load_audio(path, 16000)[:, :480000], l2_norm=True)
                      for path in (original, candidate)]
        result = torch.nn.functional.cosine_similarity(*embeddings).item()
    return max(-1.0, min(1.0, result))


if __name__ == "__main__":
    print(json.dumps({"cosine": cosine(*sys.argv[1:])}))
