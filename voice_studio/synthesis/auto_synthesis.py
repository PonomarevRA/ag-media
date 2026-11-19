"""Persist and run bounded synthesis trials with an immutable evaluation recording."""

from contextlib import closing
import json
from math import isfinite
from time import monotonic, time
from uuid import uuid4

from voice_studio.synthesis.engines import AUTO_ENGINES, SINGLE_REFERENCE_ENGINES
from voice_studio.synthesis.speech_metrics import scorer_ready, scoring_version
from voice_studio.speech.recognition import ready
from voice_studio.prosody.revolver import synthesis_variants

SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_runs (
    id TEXT PRIMARY KEY, language TEXT NOT NULL, demo INTEGER NOT NULL, text TEXT NOT NULL,
    refs TEXT NOT NULL, original_id TEXT NOT NULL, target INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
    created REAL NOT NULL, deadline REAL NOT NULL, finished REAL, status TEXT NOT NULL, reason TEXT,
    cancelled INTEGER NOT NULL DEFAULT 0, best_id TEXT
);
CREATE TABLE IF NOT EXISTS auto_attempts (
    run_id TEXT NOT NULL, position INTEGER NOT NULL, job_id TEXT UNIQUE NOT NULL,
    previous_job TEXT, incumbent_id TEXT, outcome TEXT, quality REAL, cosine REAL, wer REAL,
    started REAL NOT NULL, finished REAL, model TEXT, variant TEXT, reference_count INTEGER, PRIMARY KEY (run_id, position)
);
"""


def number(data, name, minimum, maximum):
    """Read a bounded integer provided by the automatic-run form."""
    value = data.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Некорректный параметр {name}: от {minimum} до {maximum}.")
    return value


def assess(candidate, incumbent):
    """Return the score and promotion verdict while guarding both metric components."""
    cosine, wer = candidate["cosine"], candidate["wer"]
    if cosine is None or wer is None or not isfinite(cosine) or not isfinite(wer):
        return None, "unscored"
    quality = 60 * max(0, min(1, cosine)) + 40 * max(0, 1 - wer)
    if incumbent is None:
        return quality, "promoted"
    if cosine < incumbent["cosine"] - 1e-9 or wer > incumbent["wer"] + 1e-9:
        return quality, "regression"
    return quality, "promoted" if quality > incumbent["quality"] + 1e-9 else "unchanged"


class AutoSynthesis:
    """Coordinate one local run on the Studio worker and publish its decision graph."""

    def __init__(self, studio):
        self.studio = studio
        with closing(studio.connect()) as db, db:
            db.executescript(SCHEMA)
            run_columns = {row["name"] for row in db.execute("PRAGMA table_info(auto_runs)")}
            if "scoring_version" not in run_columns:
                db.execute("ALTER TABLE auto_runs ADD COLUMN scoring_version TEXT NOT NULL DEFAULT 'xtts60-whisper40-v1'")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(auto_attempts)")}
            for name, kind in (("model", "TEXT"), ("variant", "TEXT"), ("reference_count", "INTEGER")):
                if name not in columns:
                    db.execute(f"ALTER TABLE auto_attempts ADD COLUMN {name} {kind}")
            db.execute("""UPDATE auto_attempts SET outcome='interrupted', finished=?
                          WHERE outcome IS NULL""", (time(),))
            db.execute("""UPDATE auto_runs SET status='stopped', reason='restart', finished=?
                          WHERE status IN ('queued','running')""", (time(),))

    def enabled(self):
        """Report readiness of both local scoring components."""
        return not self.studio.demo and ready() and scorer_ready()

    def start(self, language, data):
        """Reserve a single active run after validating the evaluation and conditioning inputs."""
        studio = self.studio
        if not studio.demo and not self.enabled():
            raise ValueError("Для автосинтеза нужны локальные Whisper и энкодер голоса.")
        target = number(data, "target", 50, 100)
        limit = number(data, "max_attempts", 1, 12)
        minutes = number(data, "minutes", 1, 30)
        original, = studio.references(language, [data.get("comparison")], include_evaluation=True)
        ids = data.get("refs")
        rows = studio.references(language, ids)
        rows = [row for row in rows if row["id"] != original["id"]]
        if len(rows) < 2:
            raise ValueError("Выбери как минимум два образца голоса помимо отдельной записи для оценки.")
        # Fish serves inference independently, so Studio cannot enforce the same local deadline for it.
        models = data.get("models")
        if not isinstance(models, list) or not models or any(not isinstance(model, str) or model not in AUTO_ENGINES for model in models):
            raise ValueError("Выбери модель с поддержкой автосинтеза.")
        plan = studio.synthesis_plan(language, list(dict.fromkeys(models)), data.get("checkpoints", {}))
        settings = {"stress": language == "ru" and data.get("stress") is True}
        key, created = uuid4().hex, time()
        deadline = created + minutes * 60
        with closing(studio.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM auto_runs WHERE status IN ('queued','running')").fetchone():
                raise ValueError("Автосинтез уже выполняется.")
            studio.references(language, [row["id"] for row in rows])
            studio.references(language, [original["id"]], include_evaluation=True)
            db.execute("""INSERT INTO auto_runs(id,language,demo,text,refs,original_id,target,max_attempts,
                          created,deadline,status,scoring_version) VALUES (?,?,?,?,?,?,?,?,?,?,'queued',?)""",
                       (key, language, int(studio.demo), original["text"], json.dumps([r["id"] for r in rows]),
                        original["id"], target, limit, created, deadline, scoring_version()))
        studio.worker.submit(self.work, key, language, original, rows, plan, settings, target, limit, deadline)
        return key

    def stop(self, language, key):
        """Request a stop after the current attempt, preserving the best available version."""
        if not isinstance(key, str):
            raise ValueError("Прогон не найден.")
        studio = self.studio
        with closing(studio.connect()) as db, db:
            db.execute("""UPDATE auto_runs SET cancelled=1 WHERE id=? AND language=? AND demo=?
                          AND status IN ('queued','running')""", (key, language, int(studio.demo)))
            db.execute("""UPDATE auto_runs SET status='stopped',reason='cancelled',finished=?
                          WHERE id=? AND language=? AND demo=? AND status='queued'""",
                       (time(), key, language, int(studio.demo)))

    def state(self, language):
        """Publish recent maps and their immutable edges, including removed audio."""
        studio = self.studio
        with closing(studio.connect()) as db:
            # Both queries must observe the same winner and attempt rows during a promotion.
            db.execute("BEGIN")
            runs = [dict(row) for row in db.execute(
                """SELECT id,text,target,max_attempts,created,deadline,finished,status,reason,cancelled,best_id,refs,scoring_version
                   FROM auto_runs WHERE language=? AND demo=? ORDER BY created DESC LIMIT 20""",
                (language, int(studio.demo)))]
            if not runs:
                return []
            placeholders = ",".join("?" for _ in runs)
            attempts = db.execute(
                f"""SELECT a.run_id,a.position,a.job_id,a.previous_job,a.incumbent_id,a.outcome,
                           a.quality,a.cosine,a.wer,a.started,a.finished,
                           j.status AS audio_status,coalesce(a.model,j.model) AS model,
                           coalesce(a.variant,j.variant) AS variant,
                           a.reference_count,j.seconds,j.error,j.seed,j.refs,j.checkpoint_id
                    FROM auto_attempts a LEFT JOIN jobs j ON j.id=a.job_id
                    WHERE a.run_id IN ({placeholders}) ORDER BY a.run_id,a.position""",
                [run["id"] for run in runs])
            grouped = {run["id"]: [] for run in runs}
            for row in attempts:
                attempt = dict(row)
                refs = attempt.pop("refs")
                count = attempt.pop("reference_count")
                attempt["references"] = count if count is not None else len(json.loads(refs)) if refs else None
                grouped[row["run_id"]].append(attempt)
        for run in runs:
            run["references"] = len(json.loads(run.pop("refs")))
            run["attempts"] = grouped[run["id"]]
        return runs

    def work(self, key, language, original, rows, plan, settings, target, limit, deadline):
        """Try one candidate at a time; stop on target, cancellation or an exhausted budget."""
        studio = self.studio
        best = None
        previous_job = None
        reason = "attempt_limit"
        try:
            # Claim the queued run before rendering; queued cancellation never enters the loop.
            with closing(studio.connect()) as db, db:
                updated = db.execute("UPDATE auto_runs SET status='running' WHERE id=? AND status='queued'", (key,))
                if not updated.rowcount:
                    return
            for index in range(limit):
                state, = studio.rows("SELECT cancelled FROM auto_runs WHERE id=?", (key,))
                if time() >= deadline or state["cancelled"]:
                    reason = "time_limit" if time() >= deadline else "cancelled"
                    break
                model, checkpoint = plan[index % len(plan)]
                round_number = index // len(plan)
                offset = round_number % len(rows)
                chosen = rows[offset:] + rows[:offset]
                if model in SINGLE_REFERENCE_ENGINES:
                    usable = chosen[:1]
                else:
                    # Alternate the full reference set with a smaller independent subset.
                    usable = chosen[:2] if round_number % 2 and len(chosen) > 2 else chosen[:5]
                variants = synthesis_variants(original["text"], language, model, settings)
                variant, phrases = variants[round_number % len(variants)]
                budget = monotonic() + max(0, deadline - time())
                attempt = self.run_attempt(key, language, original, usable, model, checkpoint, variant, phrases,
                                           index + 1, previous_job, best, budget)
                previous_job = attempt["job_id"]
                if attempt["outcome"] == "promoted":
                    best = attempt
                if best and best["quality"] >= target:
                    reason = "target"
                    break
                if time() >= deadline:
                    reason = "time_limit"
                    break
                if studio.rows("SELECT cancelled FROM auto_runs WHERE id=?", (key,))[0]["cancelled"]:
                    reason = "cancelled"
                    break
        except Exception as exc:
            print(f"Auto synthesis {key}: {type(exc).__name__}: {exc}", flush=True)
            reason = "error"
        finally:
            with closing(studio.connect()) as db, db:
                db.execute("UPDATE auto_attempts SET outcome='interrupted',finished=? WHERE run_id=? AND outcome IS NULL",
                           (time(), key))
                db.execute("""UPDATE auto_runs SET status='stopped',reason=?,finished=?
                              WHERE id=? AND status!='stopped'""",
                           (reason, time(), key))

    def run_attempt(self, key, language, original, rows, model, checkpoint, variant, phrases,
                    position, previous_job, incumbent, budget):
        """Persist the node before rendering and atomically update the selected version afterward."""
        studio = self.studio
        job = uuid4().hex
        seed = uuid4().int % 2**32
        best_id = incumbent["job_id"] if incumbent else None
        with closing(studio.connect()) as db, db:
            db.execute("""INSERT INTO jobs(id, kind, model, status, language, text, refs, demo, profile,
                                           comparison_id, seed, variant, checkpoint_id)
                          VALUES (?,'synthesis',?,'queued',?,?,?,?,0,?,?,?,?)""",
                       (job, model, language, original["text"], json.dumps([row["id"] for row in rows]),
                        int(studio.demo), original["id"], seed, f"auto-{variant}", checkpoint["id"] if checkpoint else None))
            db.execute("""INSERT INTO auto_attempts(run_id,position,job_id,previous_job,incumbent_id,started,
                                                         model,variant,reference_count)
                          VALUES (?,?,?,?,?,?,?,?,?)""",
                       (key, position, job, previous_job, best_id, time(), model, f"auto-{variant}", len(rows)))
            db.execute("UPDATE auto_runs SET status='running' WHERE id=?", (key,))
        studio.run_synthesis(job, model, language, rows, 0, checkpoint, original["id"], seed,
                             f"auto-{variant}", phrases, deadline=budget)
        finished, = studio.rows("SELECT status,run_id,word_error_rate AS wer,speaker_cosine AS cosine FROM jobs WHERE id=?", (job,))
        quality, outcome = assess(finished, incumbent) if finished["status"] == "done" else (None, "failed")
        with closing(studio.connect()) as db, db:
            db.execute("""UPDATE auto_attempts SET outcome=?,quality=?,cosine=?,wer=?,finished=? WHERE job_id=?""",
                       (outcome, quality, finished["cosine"], finished["wer"], time(), job))
            if outcome == "promoted":
                db.execute("UPDATE auto_runs SET best_id=? WHERE id=?", (job, key))
        if finished["run_id"]:
            try:
                tracker = studio.tracking.client
                tracker.set_tag(finished["run_id"], "auto_run", key)
                tracker.set_tag(finished["run_id"], "auto_decision", outcome)
                tracker.set_tag(finished["run_id"], "auto_scoring", scoring_version())
                if quality is not None:
                    tracker.log_metric(finished["run_id"], "auto_quality", quality)
            except Exception as exc:
                print(f"MLflow automatic synthesis: {type(exc).__name__}", flush=True)
        return {**finished, "job_id": job, "outcome": outcome, "quality": quality}
