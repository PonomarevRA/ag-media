"""Evaluate the same short, medium and long held-out lines using local models."""

import argparse
from pathlib import Path
import time

from voice_studio.synthesis.corpus import HELD_OUT
from voice_studio.speech.recognition import transcribe
from voice_studio.synthesis.speech_metrics import word_error_rate
from voice_studio.app.studio import Studio


def await_results(studio, ids):
    """Read final job rows without retaining generated text in MLflow."""
    while True:
        results = [studio.rows("SELECT * FROM jobs WHERE id=?", (key,))[0] for key in ids]
        if all(row["status"] in {"done", "failed"} for row in results):
            return results
        time.sleep(1)


def main():
    """Run the evaluation against a supplied, consented or synthetic WAV fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("transcript")
    args = parser.parse_args()
    studio = Studio(args.workspace)
    try:
        reference = studio.record("ru", args.transcript, args.reference.read_bytes())
        for band, text in zip(("short", "medium", "long"), HELD_OUT["ru"]):
            ids = studio.synthesize("ru", {"text": text, "models": ["f5", "coqui"], "refs": [reference]})
            for row in await_results(studio, ids):
                if row["status"] == "failed":
                    raise RuntimeError(row["error"])
                audio = studio.media / f"{row['id']}.wav"
                recognized = transcribe(studio.root, "ru", audio.read_bytes())
                score = word_error_rate(text, recognized)
                run_id = row["run_id"]
                studio.tracking.client.log_param(run_id, "evaluation_band", band)
                studio.tracking.client.log_metric(run_id, "whisper_word_error_rate", score)
                print(f"{band:6} {row['model']:5} {row['seconds']:.2f}s WER={score:.3f}", flush=True)
    finally:
        studio.worker.shutdown()


if __name__ == "__main__":
    main()
