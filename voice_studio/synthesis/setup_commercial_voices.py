"""Install exact checkpoint revisions after creating the MLX audio environment."""

from os.path import relpath

from huggingface_hub import snapshot_download

from voice_studio.synthesis.commercial_voice import REVISIONS
from voice_paths import MODELS_DIR

MODEL_ROOT = MODELS_DIR / "commercial"


def install():
    """Stage only inference assets and prepare the dependency's offline cache."""
    for name, (repo, revision) in REVISIONS.items():
        folder = MODEL_ROOT / name
        snapshot_download(repo_id=repo, revision=revision, local_dir=folder,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "README.md", "LICENSE*"])
        if name == "s3tokenizer":
            # MLX Audio 0.6.5 resolves this model by repository name without a revision argument.
            cache = MODEL_ROOT / "hub" / "models--mlx-community--S3TokenizerV2"
            snapshot = cache / "snapshots" / revision
            snapshot.mkdir(parents=True, exist_ok=True)
            for filename in ("config.json", "model.safetensors"):
                link = snapshot / filename
                if not link.exists():
                    if link.is_symlink():
                        link.unlink()
                    link.symlink_to(relpath(folder / filename, snapshot))
            refs = cache / "refs"
            refs.mkdir(exist_ok=True)
            (refs / "main").write_text(revision)


if __name__ == "__main__":
    install()
