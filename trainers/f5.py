"""Fine-tune the local F5 base or an earlier profile with the upstream trainer."""

import json
import errno
import os
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_paths import model_path, publish_file, temporary_directory


def inference_weights(trained):
    """Package the final optimizer-updated tensors in the F5 loader format."""
    weights = {f"ema_model.{name}": tensor for name, tensor in trained["model_state_dict"].items()}
    for key in ("initted", "step"):
        weights[key] = trained["ema_model_state_dict"][key]
    return weights


def corpus_rows(corpus, directory):
    """Extract only manifest-referenced normalized recordings with training lengths."""
    rows = []
    with ZipFile(corpus) as archive:
        for line in archive.read("manifest.jsonl").splitlines():
            item = json.loads(line)
            if not 0.35 <= item["seconds"] <= 30:
                continue
            destination = directory / Path(item["audio"]).name
            destination.write_bytes(archive.read(item["audio"]))
            rows.append({"audio_path": str(destination), "text": item["text"], "duration": item["seconds"]})
    if len(rows) < 2:
        raise ValueError("F5 requires at least two recordings of at most 30 seconds.")
    return rows


def train(job):
    """Train another version from the parent weights and record a deployable checkpoint."""
    from importlib.resources import files
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    from f5_tts.model import CFM, Trainer
    from f5_tts.model.dataset import CustomDataset
    from f5_tts.model.utils import get_tokenizer

    class InferenceTrainer(Trainer):
        """Save deployable weights without writing an optimizer checkpoint."""

        def save_checkpoint(self, update, last=False):
            """Publish the final weights, failing if all inference tensors match the parent."""
            if not last or not self.is_main:
                return
            model = self.accelerator.unwrap_model(self.model)
            trained = {"model_state_dict": model.state_dict(),
                       "ema_model_state_dict": self.ema_model.state_dict()}
            weights = inference_weights(trained)
            parent_path = Path(self.checkpoint_path) / "pretrained_voice.safetensors"
            with safe_open(str(parent_path), framework="pt", device="cpu") as parent:
                parent_names = set(parent.keys())
                changed = any(name.startswith("ema_model.") and name in parent_names
                              and not torch.equal(tensor.detach().cpu(), parent.get_tensor(name))
                              for name, tensor in weights.items())
            if not changed:
                raise ValueError("F5 training did not change any inference tensors.")
            checkpoint_dir = Path(job["output_dir"])
            temporary = Path(self.checkpoint_path) / "inference.safetensors"
            save_file({name: tensor.detach().cpu().contiguous() for name, tensor in weights.items()},
                      str(temporary))
            publish_file(temporary, checkpoint_dir / "model.safetensors")
            print(f"Saved inference tensors after {update} updates.", flush=True)

    vocab = os.getenv("VOICE_F5_VOCAB", str(model_path("f5", "vocab.txt")))
    weights = job["parent"]["path"] if job["parent"] else os.getenv(
        "VOICE_F5_CHECKPOINT", str(model_path("f5", "model.safetensors")))
    architecture = os.getenv("VOICE_F5_ARCHITECTURE", "F5TTS_Base")
    config = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{architecture}.yaml")))
    model_class = get_class(f"f5_tts.model.{config.model.backbone}")
    char_map, vocab_size = get_tokenizer(vocab, "custom")
    mel = OmegaConf.to_container(config.model.mel_spec, resolve=True)
    model = CFM(transformer=model_class(**config.model.arch, text_num_embeds=vocab_size,
                                        mel_dim=mel["n_mel_channels"]),
                mel_spec_kwargs=mel, vocab_char_map=char_map)
    output = Path(job["output_dir"])
    with temporary_directory(prefix="f5-training-") as folder:
        work = Path(folder)
        rows = corpus_rows(job["corpus"], work)
        checkpoint = work / "pretrained_voice.safetensors"
        try:
            os.link(weights, checkpoint)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(weights, checkpoint)
        epochs = int(os.getenv("VOICE_F5_EPOCHS", "1"))
        if not 1 <= epochs <= 30:
            raise ValueError("VOICE_F5_EPOCHS must be between 1 and 30.")
        # Profiles have at most 500 recordings and 30 epochs, so these intervals
        # leave exactly one save at the end of the training session.
        trainer = InferenceTrainer(model, epochs, 1e-5, num_warmup_updates=1,
                                   save_per_updates=100000, last_per_updates=100000,
                                   keep_last_n_checkpoints=0, checkpoint_path=str(work),
                                   batch_size_per_gpu=1, logger=None)
        trainer.train(CustomDataset(rows, **mel), num_workers=1, resumable_with_seed=42)
        (output / "checkpoint.json").write_text(json.dumps({"checkpoint": "model.safetensors"}))


if __name__ == "__main__":
    train(json.load(sys.stdin))
