"""Compare F5 checkpoints and XTTS with paired seeds and human-independent fixtures."""

import argparse
import csv
from pathlib import Path
import subprocess

from evaluate_speech import await_results
from voice_studio.speech.recognition import transcribe
from voice_studio.synthesis.speech_metrics import word_error_rate
from voice_studio.app.studio import Studio

REFERENCE = "Два друга назначили встречу, но каждый пришёл в своё любимое кафе."
HELD_OUT = "На пятой минуте мы поменяли правила и снова запустили таймер."
FIELDS = ("case", "variant", "seed", "seconds", "word_error_rate", "speaker_cosine", "run_id", "wav")


def create_fixtures(studio, reference_wav):
    """Create the conditioning sample and synthetic held-out recording."""
    reference = studio.record("ru", REFERENCE, reference_wav.read_bytes())
    held_out_aiff = studio.root / "held-out.aiff"
    subprocess.run(["say", "-v", "Milena", "-o", str(held_out_aiff), HELD_OUT], check=True)
    held_out = studio.record("ru", HELD_OUT, held_out_aiff.read_bytes(), evaluation_only=True)
    return reference, held_out


def evaluate(studio, reference, held_out, checkpoints, append=False):
    """Generate paired takes per variant and append new variants to an existing run if requested."""
    variants = [] if append else [("F5 base", "f5", None), ("XTTS base", "coqui", None)]
    for label, path in checkpoints:
        if not path.is_file():
            raise ValueError(f"Checkpoint missing: {path}")
        key = f"evaluation-{label}"
        studio.write("INSERT INTO checkpoints VALUES(?,?,?,?,?,?)",
                     (key, "f5", "ru", 0, str(path.resolve()), ""))
        variants.append((label, "f5", key))

    with (studio.root / "results.csv").open("a" if append else "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if not append:
            writer.writeheader()
        for case, target, text in (("repeat", reference, REFERENCE), ("held-out", held_out, HELD_OUT)):
            original = studio.media / f"{target}.wav"
            original_wer = word_error_rate(text, transcribe(studio.root, "ru", original.read_bytes()))
            print(f"{case}: original Whisper WER {original_wer:.3f}", flush=True)
            for variant, model, checkpoint in variants:
                ids = studio.synthesize("ru", {"text": text, "refs": [reference], "models": [model],
                                                "comparison": target, "repeats": 2,
                                                "checkpoints": {model: checkpoint}})
                for job in await_results(studio, ids):
                    if job["status"] != "done" or job["word_error_rate"] is None or job["speaker_cosine"] is None:
                        raise RuntimeError(f"{variant} {case}: job failed or metric missing: {job['error']}")
                    row = {name: job[name] for name in ("seed", "seconds", "word_error_rate", "speaker_cosine", "run_id")}
                    row.update(case=case, variant=variant, wav=f"media/{job['id']}.wav")
                    writer.writerow(row)
                    output.flush()
                    client = studio.tracking.client
                    client.log_param(job["run_id"], "evaluation_case", case)
                    client.log_param(job["run_id"], "evaluation_variant", variant)
                    client.log_metric(job["run_id"], "original_whisper_word_error_rate", original_wer)
                    print(case, variant, job["seed"], round(job["word_error_rate"], 3),
                          round(job["speaker_cosine"], 3), flush=True)


def main():
    """Run in a new workspace so existing recordings and experiments remain separate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs=2)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=False)
    studio = Studio(args.workspace)
    try:
        reference, held_out = create_fixtures(studio, args.reference)
        checkpoints = [(f"F5 v{index}", path) for index, path in enumerate(args.checkpoints, 1)]
        evaluate(studio, reference, held_out, checkpoints)
    finally:
        studio.worker.shutdown()


if __name__ == "__main__":
    main()
