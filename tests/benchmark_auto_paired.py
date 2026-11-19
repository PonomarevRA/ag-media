"""Run one fixed-seed, two-engine automatic benchmark in an isolated workspace."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch
from uuid import UUID, uuid4

from voice_studio.app.studio import Studio

SCRIPT = "Ты у пиццы, он у лапши. Зато никто не голодный."
REFERENCES = ("Два друга назначили встречу и договорились ждать друг друга.",
              "Сегодня мы проверим короткую фразу и сравним результаты.")


def main():
    """Measure the real worker, fixing UUID-derived seeds and preserving its graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    args.result.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as folder:
        studio = Studio(Path(folder))
        try:
            refs = [studio.record("ru", text, (args.fixture / f"ref{i}.wav").read_bytes())
                    for i, text in enumerate(REFERENCES, 1)]
            comparison = studio.record("ru", SCRIPT, (args.fixture / "control.wav").read_bytes(),
                                       evaluation_only=True)
            call = 0

            def scoped_uuid():
                """Replace the seed draws after the run ID and each attempt ID."""
                nonlocal call
                call += 1
                if call in (3, 5):
                    return UUID(int=args.seed)
                return uuid4()

            started = monotonic()
            # This override only touches the automatic runner; recorded IDs are generated normally.
            with patch('voice_studio.synthesis.auto_synthesis.uuid4', side_effect=scoped_uuid):
                studio.automatic.start("ru", {
                    "comparison": comparison, "refs": refs, "models": ["f5", "coqui"],
                    "target": 100, "max_attempts": 2, "minutes": 5,
                })
                studio.worker.submit(lambda: None).result(timeout=330)
            elapsed = monotonic() - started
            run, = studio.automatic.state("ru")
            assert run["status"] == "stopped" and run["reason"] == "attempt_limit", run
            assert [item["model"] for item in run["attempts"]] == ["f5", "coqui"], run
            incumbent = None
            for item in run["attempts"]:
                assert item["seed"] == args.seed and item["audio_status"] == "done", item
                assert item["quality"] is not None and item["incumbent_id"] == incumbent, item
                if item["outcome"] == "promoted":
                    incumbent = item["job_id"]
                wav = (studio.media / f"{item['job_id']}.wav").read_bytes()
                (args.result / f"{item['model']}.wav").write_bytes(wav)
                item["wavSha256"] = hashlib.sha256(wav).hexdigest()
            assert incumbent is not None and run["best_id"] == incumbent, run
            (args.result / "metrics.json").write_text(json.dumps({"elapsed": elapsed, **run}, ensure_ascii=False))
        finally:
            studio.worker.shutdown()


if __name__ == "__main__":
    main()
