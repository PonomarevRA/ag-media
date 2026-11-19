"""Run inference in the separately installed F5 or XTTS environment."""

import json
import sys


def synthesize(job):
    """Generate a WAV from a trusted checkpoint and the recorded references."""
    if job["model"] == "f5":
        from f5_tts.api import F5TTS

        engine = F5TTS(
            model=job["architecture"],
            ckpt_file=job["checkpoint"],
            vocab_file=job["vocab"],
            device=job["device"],
            vocoder_local_path=job["vocoder"],
        )
        reference = job["references"][0]
        engine.infer(
            ref_file=reference["path"], ref_text=reference["text"],
            gen_text=job["text"], file_wave=job["output"], seed=job["seed"],
        )
    else:
        from pathlib import Path
        import soundfile
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(job["config"])
        checkpoint = Path(job["checkpoint"])
        engine = Xtts.init_from_config(config)
        engine.load_checkpoint(config, checkpoint_dir=str(checkpoint.parent),
                               checkpoint_path=str(checkpoint), eval=True)
        engine.to(job["device"])
        torch.manual_seed(job["seed"])
        references = [item["path"] for item in job["references"]]
        conditioning, embedding = engine.get_conditioning_latents(audio_path=references)
        result = engine.inference(job["text"], job["language"], conditioning, embedding,
                                  enable_text_splitting=True)
        soundfile.write(job["output"], result["wav"], config.audio.output_sample_rate)


if __name__ == "__main__":
    synthesize(json.load(sys.stdin))
