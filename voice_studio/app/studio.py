"""Persist voice samples, queue model runs, and assemble approved audio."""

from voice_studio.builtin_mods.synthesis import capture_synthesis
from voice_studio.mod_runtime import current_snapshot

from voice_studio.app.transcript_editor import TranscriptEditor

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from shutil import copyfile
from voice_paths import publish_file, temporary_directory
from threading import Lock
from time import monotonic
from uuid import uuid4
import wave
from zipfile import ZipFile

from voice_studio.synthesis.auto_synthesis import AutoSynthesis
from voice_studio.synthesis.block_assembly import checked_blocks, render_blocks
from voice_studio.synthesis.corpus import EXTRA, reading_plan
from voice_studio.synthesis.commercial_voice import MODELS as MLX_VOICES, REVISIONS as VOICE_REVISIONS
from voice_studio.synthesis.engines import (F5_ACCENT_WEIGHTS, MODELS, SINGLE_REFERENCE_ENGINES, availability, ffmpeg, generate,
                     local_weights, normalize, research_enabled)
from voice_studio.synthesis.fragment_similarity import matching_windows, score_windows
from voice_studio.prosody.reading_story import recording_story
from voice_studio.prosody.reference_script import recorded_prompt, reference_script
from voice_studio.prosody.reference_take_matching import extract_reference_takes
from voice_studio.speech.recognition import SMALL_MODEL, ready as recognition_ready, transcribe
from voice_studio.prosody.prosody import VERSION as PROSODY_VERSION
from voice_studio.prosody.revolver import (VERSION as REVOLVER_VERSION, fragment_radius, fragment_trials, mark_lapshi,
                      review_words, synthesis_variants)
from voice_studio.synthesis.speech_metrics import measure, speaker_encoder
from voice_studio.synthesis.accent_profiles import reference_language, saved_accent
from voice_studio.synthesis.reference_enrollment import MAX_TAKES, check_references, delivery_comparison, inspect_take
from voice_studio.synthesis.voice_sequence import render_voice_sequence
from voice_studio.synthesis.voice_catalog import profile_name, voice_catalog
from voice_studio.vision.tracking import Tracking

PROMPTS = {
    "ru": [
        "Два друга назначили встречу, но каждый пришёл в своё любимое кафе.",
        "Ты уже на месте? Подожди минутку, я вижу вывеску с другого конца улицы!",
        "Шесть маленьких шариков шумно покатились по деревянному полу.",
        "Сначала мы проверим один вариант, а потом сравним его с остальными.",
        "В среду в семь часов встречаемся у входа. Пожалуйста, не опаздывай.",
        "Звучит заманчиво, но что произойдёт, если каждый выберет своё?",
        "Один билет стоит двести рублей, а на двоих действует скидка.",
        "Отлично! Теперь попробуй прочитать эту фразу медленнее и спокойнее.",
    ],
    "en": [
        "Two friends planned to meet, but each went to a different café.",
        "Are you there already? Wait a minute, I can see the sign from here!",
        "Six shiny marbles rolled across the rough wooden floor.",
        "First we will try one option, then we can compare it with the others.",
        "Meet me at the entrance at seven on Wednesday. Please be on time.",
        "That sounds tempting, but what if everyone picks their own option?",
        "One ticket costs twenty dollars, and two tickets come with a discount.",
        "Great! Now try reading this sentence slowly and calmly.",
    ],
}
TEST_TEXT = {
    "ru": "Ты у пиццы, он у лапши. Ноль очков за встречу. Зато никто не голодный.",
    "en": "You are having pizza. Your friend is having noodles. At least nobody is hungry.",
}


def require_text(value, limit):
    """Check and trim an input text field."""
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ValueError(f"Введите текст длиной от 1 до {limit} символов.")
    return value.strip()


class Studio:
    """Single local workspace with SQLite state, WAV files, and one worker."""

    def __init__(self, root, demo=False):
        self.root = root.resolve()
        self.demo = demo
        self.media = self.root / "media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.tracking = Tracking(self.root)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.fragment_scoring = Lock()
        with closing(self.connect()) as db, db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS recordings (
                id TEXT PRIMARY KEY, language TEXT, text TEXT, seconds REAL,
                prompt_number INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP
              );
              CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT, model TEXT, status TEXT, language TEXT, text TEXT,
                refs TEXT, demo INTEGER, run_id TEXT, seconds REAL, score INTEGER, error TEXT,
                profile INTEGER, checkpoint_id TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP
              );
              CREATE TABLE IF NOT EXISTS profiles (
                version INTEGER PRIMARY KEY AUTOINCREMENT, language TEXT, refs TEXT,
                created TEXT DEFAULT CURRENT_TIMESTAMP
              );
              CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY, model TEXT, language TEXT, profile INTEGER, path TEXT, config TEXT
              );
            """)
            recording_columns = {row["name"] for row in db.execute("PRAGMA table_info(recordings)")}
            if "prompt_number" not in recording_columns:
                db.execute("ALTER TABLE recordings ADD COLUMN prompt_number INTEGER")
            if "evaluation_only" not in recording_columns:
                db.execute("ALTER TABLE recordings ADD COLUMN evaluation_only INTEGER NOT NULL DEFAULT 0")
            if "reference_prompt" not in recording_columns:
                db.execute("ALTER TABLE recordings ADD COLUMN reference_prompt TEXT")
            profile_columns = {row["name"] for row in db.execute("PRAGMA table_info(profiles)")}
            if "reference_report" not in profile_columns:
                db.execute("ALTER TABLE profiles ADD COLUMN reference_report TEXT")
            if "name" not in profile_columns:
                db.execute("ALTER TABLE profiles ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "checkpoint_id" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN checkpoint_id TEXT")
            for name, kind in (("comparison_id", "TEXT"), ("seed", "INTEGER"),
                               ("word_error_rate", "REAL"), ("speaker_cosine", "REAL"), ("variant", "TEXT"), ("baseline_id", "TEXT"),
                               ("comparison_start", "INTEGER"), ("retry_focus", "INTEGER"), ("retry_stress", "INTEGER"),
                               ("batch_id", "TEXT"), ("accent_mode", "TEXT"), ("speaker_encoder", "TEXT"),
                               ("delivery_comparison", "TEXT"), ("voice_sequence", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
            for row in db.execute("SELECT run_id FROM jobs WHERE status IN ('queued', 'running')"):
                self.tracking.interrupt(row["run_id"])
            db.execute("""UPDATE jobs SET status='failed', error='Работа прервана при перезапуске.'
                          WHERE status IN ('queued', 'running')""")

        self.automatic = AutoSynthesis(self)
        self.transcripts = TranscriptEditor(self.connect)

    def connect(self):
        """Open a short-lived connection with dictionary-compatible rows."""
        db = sqlite3.connect(self.root / "studio.db", timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def rows(self, sql, args=()):
        """Read rows from a short-lived connection."""
        with closing(self.connect()) as db:
            return [dict(row) for row in db.execute(sql, args)]

    def write(self, sql, args=()):
        """Commit one statement and return the inserted row ID when applicable."""
        with closing(self.connect()) as db, db:
            return db.execute(sql, args).lastrowid

    def state(self, language):
        """Return only UI fields, current prompt and engine status."""
        base_whisper_ready = recognition_ready()
        recordings = self.rows("SELECT * FROM recordings WHERE language=? ORDER BY rowid", (language,))
        corpus_recordings = [row for row in recordings if not row["evaluation_only"]]
        plan = reading_plan(language, PROMPTS[language], corpus_recordings)
        profiles = self.rows("SELECT * FROM profiles WHERE language=? ORDER BY version DESC", (language,))
        checkpoints = self.rows("SELECT id,model,profile FROM checkpoints WHERE language=?", (language,))
        jobs = self.rows("""SELECT jobs.id,kind,jobs.model,status,text,seconds,score,error,jobs.profile,
                                   demo,run_id,checkpoint_id,checkpoints.profile AS checkpoint_profile,
                                   comparison_id,seed,word_error_rate,speaker_cosine,speaker_encoder,jobs.variant,
                                   jobs.baseline_id,jobs.retry_focus,jobs.batch_id,jobs.delivery_comparison,jobs.voice_sequence
                            FROM jobs LEFT JOIN checkpoints ON jobs.checkpoint_id=checkpoints.id
                            WHERE jobs.language=? AND demo=? ORDER BY jobs.rowid DESC""",
                         (language, int(self.demo)))
        for job in jobs:
            if job["variant"] == "pilot-plain" and job["status"] == "done" and job["run_id"]:
                job["listener_choice"] = self.tracking.client.get_run(job["run_id"]).data.tags.get(
                    "listener_stress_choice")
        return {"recordings": recordings, "profiles": profiles, **plan, "recognition": base_whisper_ready,
                "voiceCatalog": voice_catalog(profiles, recordings),
                "referenceBatchReady": base_whisper_ready or recognition_ready(SMALL_MODEL),
                "checkpoints": checkpoints, "jobs": jobs, "demo": self.demo, "readingStory": recording_story(language),
                "referenceScript": reference_script(language), "referenceTakeLimit": MAX_TAKES,
                "autoRuns": self.automatic.state(language), "autoReady": self.automatic.enabled(),
                "stressComparisonReady": research_enabled() and not self.demo and language == "ru" and F5_ACCENT_WEIGHTS.is_file(),
                "modelNames": MODELS,
                "engines": availability(self.demo, language, any(c["model"] == "f5" for c in checkpoints)),
                "trainers": [m for m in MODELS if (self.demo or research_enabled()) and self.trainer(m, language)], "testText": TEST_TEXT[language]}

    def record(self, language, text, audio, prompt_number=None, evaluation_only=False, *, reference_prompt=None):
        """Decode, validate, and save a user recording with its transcript."""
        text = require_text(text, 500)
        if reference_prompt is not None:
            prompt = recorded_prompt(language, reference_prompt, text)
            evaluation_only = evaluation_only or prompt.role == "evaluation"
            prompt_number = None
        if evaluation_only:
            prompt_number = None
        if prompt_number is not None and not 1 <= prompt_number <= len(PROMPTS[language]) + len(EXTRA[language]):
            raise ValueError("Фраза корпуса не найдена.")
        key = uuid4().hex
        destination = self.media / f"{key}.wav"
        with temporary_directory() as work:
            raw = Path(work) / "upload"
            raw.write_bytes(audio)
            wav = Path(work) / "take.wav"
            seconds = normalize(raw, wav, duration_limit=45)
            publish_file(wav, destination)
        try:
            self.write("""INSERT INTO recordings(id,language,text,seconds,prompt_number,evaluation_only,reference_prompt)
                          VALUES(?,?,?,?,?,?,?)""",
                       (key, language, text, seconds, prompt_number, int(evaluation_only), reference_prompt))
        except sqlite3.Error:
            destination.unlink()
            raise
        return key

    def record_reference_batch(self, language, audio, *, confirmed=None):
        """Save exact matches or selected candidates after local Whisper alignment."""
        guide = {row["id"]: row for row in reference_script(language)["prompts"]}
        created = []
        with temporary_directory(prefix="voice-reference-batch-") as folder:
            work = Path(folder)
            found, suggestions = extract_reference_takes(audio, language, work, confirmed)
            try:
                with closing(self.connect()) as db, db:
                    db.execute("BEGIN IMMEDIATE")
                    for row in found:
                        key = uuid4().hex
                        prompt = guide[row["prompt"]]
                        row["id"] = key
                        row["evaluationOnly"] = prompt["role"] == "evaluation"
                        destination = self.media / f"{key}.wav"
                        created.append(destination)
                        publish_file(work / f"take-{row['number']}.wav", destination)
                        db.execute("""INSERT INTO recordings(id,language,text,seconds,evaluation_only,reference_prompt)
                                      VALUES(?,?,?,?,?,?)""", (key, language, prompt["text"],
                                   (row["endMs"] - row["startMs"]) / 1000, row["evaluationOnly"], prompt["id"]))
            except (OSError, sqlite3.Error):
                for path in created:
                    path.unlink(missing_ok=True)
                raise
        return {"saved": found, "review": suggestions}

    def references(self, language, ids, maximum=500, include_evaluation=False):
        """Resolve an ordered selection of recordings in the requested language."""
        if (not isinstance(ids, list) or not 1 <= len(ids) <= maximum
                or not all(isinstance(key, str) for key in ids) or len(set(ids)) != len(ids)):
            raise ValueError(f"Выберите от 1 до {maximum} разных записей.")
        rows = self.rows("SELECT * FROM recordings WHERE language=?", (language,))
        index = {item["id"]: item for item in rows}
        if any(key not in index for key in ids):
            raise ValueError("Некоторые записи больше недоступны.")
        selected = [index[key] for key in ids]
        if not include_evaluation and any(row["evaluation_only"] for row in selected):
            raise ValueError("Записи «Только для проверки» можно использовать только как исходник для сравнения.")
        return selected

    def save_profile(self, language, ids, *, reference_report=None, name=None):
        """Save an immutable version of the reference set."""
        label = profile_name(name)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.references(language, ids)
            report = json.dumps(reference_report, ensure_ascii=False) if reference_report else None
            return db.execute("INSERT INTO profiles(language,refs,reference_report,name) VALUES(?,?,?,?)",
                              (language, json.dumps(ids), report, label)).lastrowid

    def check_voice_profile(self, language, ids, *, save=False, name=None):
        """Check saved takes and optionally commit the measured reference order."""
        if type(save) is not bool:
            raise ValueError("Некорректный режим сохранения профиля.")
        label = profile_name(name) if save else None
        rows = self.references(language, ids, maximum=MAX_TAKES)
        report = check_references(rows, self.media)
        if save and report["referenceIds"]:
            report["profileVersion"] = self.save_profile(language, report["referenceIds"],
                                                         reference_report=report, name=label)
        return report

    def import_accent_profile(self, source, metadata):
        """Validate one accent sample and publish its recording and profile together."""
        style = metadata.get("style")
        language = reference_language(style, metadata.get("referenceLanguage"))
        text = require_text(metadata.get("text"), 500)
        name = profile_name(metadata.get("name"))
        if not name:
            raise ValueError("Введи имя профиля.")
        key = uuid4().hex
        destination = self.media / f"{key}.wav"
        with temporary_directory(prefix="accent-profile-") as folder:
            wav = Path(folder) / "reference.wav"
            seconds = normalize(source, wav, duration_limit=18)
            take, _, _ = inspect_take({"id": key, "text": text}, wav)
            if take["issues"]:
                raise ValueError("Нужен чёткий образец речи до 18 секунд без перегруза.")
            report = {"metric": "reference-signal-quality", "accentStyle": style, "takes": [take], "pairs": []}
            try:
                with closing(self.connect()) as db, db:
                    db.execute("BEGIN IMMEDIATE")
                    publish_file(wav, destination)
                    db.execute("INSERT INTO recordings(id,language,text,seconds) VALUES(?,?,?,?)",
                               (key, language, text, seconds))
                    profile = db.execute("INSERT INTO profiles(language,refs,reference_report,name) VALUES(?,?,?,?)",
                                         (language, json.dumps([key]), json.dumps(report, ensure_ascii=False), name))
                    version = profile.lastrowid
            except (OSError, sqlite3.Error):
                destination.unlink(missing_ok=True)
                raise
        return {"version": version}

    def list_voice_catalog(self, language, *, dubbing=False):
        """Read catalog metadata for one language without touching the model workers."""
        with closing(self.connect()) as db, db:
            db.execute("BEGIN")
            if dubbing:
                profiles = [row for row in db.execute("SELECT * FROM profiles ORDER BY version DESC")
                            if row["language"] == language or saved_accent(row["reference_report"])]
                recordings = db.execute("SELECT id,evaluation_only FROM recordings").fetchall()
            else:
                profiles = db.execute("SELECT * FROM profiles WHERE language=? ORDER BY version DESC", (language,)).fetchall()
                recordings = db.execute("SELECT id,evaluation_only FROM recordings WHERE language=?", (language,)).fetchall()
        return voice_catalog(profiles, recordings)

    def accent_preview(self, version):
        """Resolve an accent preview from the reviewed recording, never a caller-supplied path."""
        if not 1 <= version < 2**63:
            return None
        rows = self.rows("SELECT * FROM profiles WHERE version=? AND reference_report IS NOT NULL", (version,))
        if not rows or not saved_accent(rows[0]["reference_report"]):
            return None
        refs = json.loads(rows[0]["refs"])
        if not refs:
            return None
        recordings = self.rows("SELECT id FROM recordings WHERE id=? AND language=?", (refs[0], rows[0]["language"]))
        path = self.media / f"{refs[0]}.wav"
        return path if recordings and path.is_file() else None

    def rename_voice_profile(self, language, version, name):
        """Update a label while keeping its stable version and exact reference audio."""
        if type(version) is not int or not 1 <= version < 2**63:
            raise ValueError("Неизвестная версия профиля.")
        label = profile_name(name)
        with closing(self.connect()) as db, db:
            changed = db.execute("UPDATE profiles SET name=? WHERE language=? AND version=?", (label, language, version))
            if not changed.rowcount:
                raise ValueError("Неизвестная версия профиля.")
        return label

    def list_dub_profiles(self, language):
        """List reviewed versions available for a speaker assignment in video dubbing."""
        return self.rows("""SELECT version FROM profiles WHERE language=? AND reference_report IS NOT NULL
                            ORDER BY version DESC LIMIT 100""", (language,))

    def copy_dub_prompts(self, language, assignments, folder):
        """Resolve reviewed profiles and snapshot their first reference under the workspace write lock."""
        if language not in ("ru", "en") or not isinstance(assignments, dict) or len(assignments) > 24:
            raise ValueError("Некорректные профили для дубляжа.")
        prompts = {}
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            for speaker, version in assignments.items():
                if (not isinstance(speaker, str) or not speaker.isascii() or not speaker.isdecimal()
                        or len(speaker) > 3 or str(int(speaker)) != speaker or speaker == "0"
                        or type(version) is not int or not 1 <= version < 2**63):
                    raise ValueError("Некорректный спикер или версия голосового профиля.")
                profile = db.execute("""SELECT refs,language,reference_report FROM profiles
                                        WHERE version=? AND reference_report IS NOT NULL""", (version,)).fetchone()
                accent = saved_accent(profile["reference_report"]) if profile else None
                if not profile or (profile["language"] != language and not accent):
                    raise ValueError("Проверенный профиль нужного языка недоступен.")
                recorded_language = profile["language"]
                key = json.loads(profile["refs"])[0]
                row = db.execute("""SELECT text FROM recordings WHERE language=? AND id=?
                                    AND evaluation_only=0""", (recorded_language, key)).fetchone()
                if row is None:
                    raise ValueError("Запись голосового профиля недоступна.")
                path = folder / f"reference-profile-{speaker}.wav"
                try:
                    copyfile(self.media / f"{key}.wav", path)
                except OSError as error:
                    raise ValueError("Аудио голосового профиля недоступно.") from error
                prompts[int(speaker)] = {"path": str(path), "text": row["text"],
                                         "profileVersion": version, "recordedLanguage": recorded_language}
                if accent:
                    prompts[int(speaker)]["accentStyle"] = accent
        return prompts

    def export(self, language, ids, destination):
        """Package the selected human takes and their exact text as a portable corpus."""
        rows = self.references(language, ids, maximum=500)
        with ZipFile(destination, "w") as archive:
            self.write_corpus(archive, rows)
        return len(rows)

    def delete(self, language, kind, key):
        """Delete a local item while protecting active jobs and saved reference sets."""
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if kind == "profile":
                deleted = db.execute("DELETE FROM profiles WHERE version=? AND language=?", (key, language))
            elif kind == "recording":
                saved_sets = db.execute("SELECT refs FROM profiles WHERE language=?", (language,))
                if any(key in json.loads(row["refs"]) for row in saved_sets):
                    raise ValueError("Запись входит в сохранённый профиль. Сначала удали этот профиль.")
                active_runs = db.execute("SELECT refs,original_id FROM auto_runs WHERE status IN ('queued','running')")
                if any(key == row["original_id"] or key in json.loads(row["refs"]) for row in active_runs):
                    raise ValueError("Запись используется в автосинтезе. Дождись завершения.")
                active_sets = db.execute("SELECT refs FROM jobs WHERE status IN ('queued','running')")
                if any(key in json.loads(row["refs"]) for row in active_sets):
                    raise ValueError("Запись используется в активном задании. Дождись завершения.")
                if db.execute("SELECT 1 FROM jobs WHERE comparison_id=? AND status IN ('queued','running')",
                              (key,)).fetchone():
                    raise ValueError("Запись используется для оценки дублей. Дождись завершения.")
                deleted = db.execute("DELETE FROM recordings WHERE id=? AND language=?", (key, language))
            elif kind == "result":
                if db.execute("""SELECT 1 FROM auto_attempts a JOIN auto_runs r ON a.run_id=r.id
                                 WHERE a.job_id=? AND r.status IN ('queued','running')""", (key,)).fetchone():
                    raise ValueError("Дубль используется в автосинтезе. Дождись завершения.")
                deleted = db.execute("""DELETE FROM jobs WHERE id=? AND language=? AND demo=?
                                      AND status IN ('done','failed') AND kind!='training'""",
                                     (key, language, int(self.demo)))
                db.execute("""UPDATE auto_runs SET best_id=(
                                SELECT a.job_id FROM auto_attempts a JOIN jobs j ON j.id=a.job_id
                                WHERE a.run_id=auto_runs.id AND a.outcome='promoted' ORDER BY a.position DESC LIMIT 1)
                              WHERE best_id=?""", (key,))
            else:
                raise ValueError("Неизвестный тип записи.")
            if not deleted.rowcount:
                raise ValueError("Элемент недоступен для удаления.")
            if kind != "profile":
                (self.media / f"{key}.wav").unlink(missing_ok=True)

    def write_corpus(self, archive, rows):
        """Write normalized WAVs and JSON Lines manifest to an open archive."""
        manifest = []
        for row in rows:
            relative = f"wavs/{row['id']}.wav"
            archive.write(self.media / f"{row['id']}.wav", relative)
            manifest.append(json.dumps({"audio": relative, "text": row["text"],
                                        "language": row["language"], "seconds": row["seconds"]},
                                       ensure_ascii=False))
        archive.writestr("manifest.jsonl", "\n".join(manifest) + "\n")

    def synthesize(self, language, data):
        """Queue the same held-out text and reference set for selected engines."""
        text = require_text(data.get("text"), 500)
        comparison_id = data.get("comparison")
        repeats = data.get("repeats", 1)
        if type(repeats) is not int or not 1 <= repeats <= 5:
            raise ValueError("Можно сделать от одного до пяти дублей на модель.")
        if comparison_id is not None:
            original = self.references(language, [comparison_id], include_evaluation=True)[0]
            if original["text"] != text:
                raise ValueError("Для оценки дубля используй расшифровку исходной записи.")
        elif repeats != 1:
            raise ValueError("Для серии дублей выбери исходную запись.")
        models = data.get("models")
        if (not isinstance(models, list) or not 1 <= len(models) <= len(MODELS)
                or any(not isinstance(item, str) or item not in MODELS for item in models)):
            raise ValueError("Выберите доступные модели.")
        ids = data.get("refs")
        profile = data.get("profile", 0)
        if type(profile) is not int or not 0 <= profile < 2**63:
            raise ValueError("Неизвестная версия профиля.")
        if profile:
            stored = self.rows("SELECT refs FROM profiles WHERE version=? AND language=?", (profile, language))
            saved_ids = json.loads(stored[0]["refs"]) if stored else None
            if saved_ids is None or (ids is not None and ids != saved_ids):
                raise ValueError("Набор записей не совпадает с версией профиля.")
            ids = saved_ids
        rows = self.references(language, ids)
        jobs = self.synthesis_plan(language, models, data.get("checkpoints", {}))
        settings = data.get("prosody", {})
        if (not isinstance(settings, dict) or set(settings) - {"stress", "pauses", "tone", "sequence"}
                or any(type(settings.get(flag, False)) is not bool for flag in ("stress", "pauses", "sequence"))
                or settings.get("tone", "") not in ("", "question", "exclamation")):
            raise ValueError("Некорректные настройки просодии.")
        # Prepare the text once per engine, before expanding the sampling plan.
        variants = {model: synthesis_variants(text, language, model, settings) for model, _ in jobs}
        if settings.get("sequence"):
            if "qwen" not in variants or not recognition_ready():
                raise ValueError("Для подбора фраз нужны Qwen и локальный Whisper.")
            if not any(variant == "voice-sequence" for variant, _ in variants["qwen"]):
                raise ValueError("Для подбора фраз введи от двух до четырёх предложений.")
        candidates = []
        for take in range(repeats):
            # Reuse the same sampling plan when switching models or checkpoints.
            seed_material = f"{comparison_id}:{take}" if comparison_id else text
            seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big")
            for model, checkpoint in jobs:
                candidates.extend((model, checkpoint, seed, variant, phrases)
                                  for variant, phrases in variants[model] if take == 0 or variant != "voice-sequence")
        return self.enqueue_synthesis(language, text, rows, profile, candidates, comparison_id)

    def compare_lapshi(self, language, data):
        """Queue three paired A/B rounds against a held-out Russian recording."""
        if self.demo or language != "ru" or not F5_ACCENT_WEIGHTS.is_file():
            raise ValueError("Для сравнения нужны локальные веса F5 accent-tune.")
        original, = self.references(language, [data.get("comparison")], include_evaluation=True)
        text = original["text"]
        marked = mark_lapshi(text)
        if marked == text:
            raise ValueError("Выбери запись со словом «лапши».")
        rows = self.references(language, data.get("refs"), maximum=1)
        if any(row["id"] == original["id"] for row in rows):
            raise ValueError("Для образца голоса выбери другую запись.")
        self.synthesis_plan(language, ["f5", "coqui"], {})
        accent = self.accent_checkpoint("lapshi-plus")
        candidates = []
        for seed in (17027, 88219, 451007):
            candidates.extend((
                ("f5", None, seed, "lapshi-f5", [(text, 0)]),
                ("f5", accent, seed, "lapshi-accent", [(marked, 0)]),
                ("coqui", None, seed, "lapshi-coqui", [(text, 0)]),
            ))
        # Both F5 variants and Coqui receive the exact same held-out conditioning take.
        return self.enqueue_synthesis(language, text, rows, 0, candidates, original["id"], batch_id=uuid4().hex)

    def accent_checkpoint(self, mode):
        """Resolve the built-in accent-tune weights and the repeated text policy."""
        if mode not in {"lapshi-plus", "plain"} or not F5_ACCENT_WEIGHTS.is_file():
            raise ValueError("Локальные веса F5 accent-tune недоступны.")
        return {"id": None, "path": str(F5_ACCENT_WEIGHTS), "config": "",
                "architecture": "F5TTS_v1_Base", "accent_mode": mode}

    def rate_prosody_pair(self, language, baseline_id, choice):
        """Store the pair verdict on the plain run, the canonical identity of the pair."""
        if choice not in ("plain", "plus", "both", "neither", "unclear"):
            raise ValueError("Выбери результат прослушивания пары.")
        baseline = self.synthesis_result(language, baseline_id)
        if baseline["variant"] != "pilot-plain":
            raise ValueError("Пара пилота не найдена.")
        rows = self.rows("""SELECT id FROM jobs WHERE baseline_id=? AND variant='pilot-plus'
                           AND seed=? AND batch_id=? AND language=? AND demo=? AND status='done'""",
                         (baseline_id, baseline["seed"], baseline["batch_id"], language, int(self.demo)))
        if len(rows) != 1:
            raise ValueError("Обе дорожки пары должны быть готовы.")
        candidate = self.synthesis_result(language, rows[0]["id"])
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT COUNT(*) FROM jobs WHERE id IN (?,?)",
                                (baseline_id, candidate["id"])).fetchone()[0]
            if active != 2:
                raise ValueError("Одна из дорожек больше недоступна.")
            self.tracking.client.set_tag(baseline["run_id"], "listener_stress_choice", choice)

    def synthesis_plan(self, language, models, selected):
        """Resolve engines and optional checkpoints for both synthesis entry points."""
        if not isinstance(selected, dict):
            raise ValueError("Некорректные checkpoints.")
        checkpoint_rows = self.rows("SELECT * FROM checkpoints WHERE language=?", (language,))
        checkpoints = {row["id"]: row for row in checkpoint_rows}
        available = availability(self.demo, language, any(row["model"] == "f5" for row in checkpoint_rows))
        jobs = []
        for model in dict.fromkeys(models):
            if model not in available:
                raise ValueError("Модель недоступна в текущем режиме.")
            checkpoint = checkpoints.get(str(selected.get(model)))
            if selected.get(model) and (not checkpoint or checkpoint["model"] != model):
                raise ValueError("Checkpoint не соответствует выбранной модели.")
            f5_reference_only = model == "f5" and language == "ru" and not self.demo and not checkpoint
            tts_override = current_snapshot().to_wire()['selections'].get('synthesis.tts')
            if f5_reference_only and not tts_override and not all(Path(path).is_file() for path in local_weights("f5")):
                raise ValueError("Выберите русский checkpoint F5.")
            if not available[model]["ready"]:
                raise ValueError(f"{MODELS[model]}: {available[model]['reason']}.")
            jobs.append((model, checkpoint))
        return jobs

    def enqueue_synthesis(self, language, text, rows, profile, jobs, comparison_id,
                          parent_id=None, comparison_start=None, retry_focus=None, retry_stress=False, *, batch_id=None):
        """Commit the complete batch before handing it to the worker."""
        refs = json.dumps([row["id"] for row in rows])
        keys = [uuid4().hex for _ in jobs]
        baselines = {(model, seed): key for key, (model, _, seed, variant, _) in zip(keys, jobs)
                     if variant in {"baseline", "pilot-plain"}}
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.references(language, [row["id"] for row in rows])
            if comparison_id:
                self.references(language, [comparison_id], include_evaluation=True)
            db.executemany(
                """INSERT INTO jobs(id,kind,model,status,language,text,refs,demo,profile,checkpoint_id,
                                    comparison_id,seed,variant,baseline_id,comparison_start,retry_focus,
                                    retry_stress,batch_id,accent_mode)
                   VALUES(?,'synthesis',?,'queued',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [(key, model, language, text, refs, int(self.demo), profile,
                  checkpoint["id"] if checkpoint else None, comparison_id, seed, variant,
                  None if variant in {"baseline", "pilot-plain"} else parent_id or baselines.get((model, seed)),
                  comparison_start, retry_focus, int(retry_stress), batch_id,
                  checkpoint.get("accent_mode") if checkpoint else None)
                 for key, (model, checkpoint, seed, variant, _) in zip(keys, jobs)],
            )
        for key, (model, checkpoint, seed, variant, phrases) in zip(keys, jobs):
            self.worker.submit(self.run_synthesis, key, model, language, rows, profile,
                               checkpoint, comparison_id, seed, variant, phrases)
        return keys

    def synthesis_result(self, language, key):
        """Resolve a completed result in this workspace with available audio."""
        if not isinstance(key, str):
            raise ValueError("Дубль не найден.")
        rows = self.rows("SELECT * FROM jobs WHERE id=? AND language=? AND demo=? "
                         "AND kind='synthesis' AND status='done'", (key, language, int(self.demo)))
        if not rows or not (self.media / f"{key}.wav").is_file():
            raise ValueError("Дубль не найден.")
        return rows[0]

    def inspect_words(self, language, key):
        """Compare the completed WAV with its own saved synthesis text."""
        job = self.synthesis_result(language, key)
        graph = self.cached_word_graph(language, key, job["text"])
        original_id = job["comparison_id"]
        original_exists = original_id and (self.media / f"{original_id}.wav").is_file()
        return {**graph, "hasOriginal": bool(original_exists) and not self.demo, "focusIndex": job["retry_focus"]}

    @lru_cache(maxsize=32)
    def cached_word_graph(self, language, key, text):
        """Reuse successful alignment for immutable WAVs in the open workspace."""
        try:
            audio = (self.media / f"{key}.wav").read_bytes()
        except FileNotFoundError as exc:
            raise ValueError("Дорожка больше недоступна.") from exc
        return review_words(self.root, language, text, audio)

    def fragment_score(self, language, key, index):
        """Score the matched word's context against the recording chosen at synthesis."""
        if type(index) is not int or index < 0:
            raise ValueError("Слово для оценки не найдено.")
        job = self.synthesis_result(language, key)
        if self.demo or not job["comparison_id"]:
            return {"cosine": None, "reason": "Для оценки выбери исходник перед синтезом."}
        original_id = job["comparison_id"]
        original = self.references(language, [original_id], include_evaluation=True)[0]
        if not self.fragment_scoring.acquire(blocking=False):
            raise ValueError("Энкодер голоса уже сравнивает фрагменты. Попробуй ещё раз.")
        try:
            return self.cached_fragment_score(language, key, job["text"], original_id, original["text"],
                                              job["comparison_start"] or 0, index)
        finally:
            self.fragment_scoring.release()

    @lru_cache(maxsize=128)
    def cached_fragment_score(self, language, key, text, original_id, transcript, source_offset, index):
        """Compare the same script tokens on the two tracks and retain successful scores."""
        candidate = self.cached_word_graph(language, key, text)
        original = self.cached_word_graph(language, original_id, transcript)
        windows = matching_windows(candidate, original, index, source_offset)
        if not windows:
            return {"cosine": None, "reason": "Нужно совпадение слов и около 1,5 с речи вокруг выбранного слова."}
        return score_windows(self.root, self.media / f"{key}.wav", self.media / f"{original_id}.wav", windows)

    def retry_words(self, language, key, index, count, context="short"):
        """Queue independent takes of a contextual fragment using its parent's engine and references."""
        parent = self.synthesis_result(language, key)
        text, trials = fragment_trials(parent["text"], index, count, context)
        radius = fragment_radius(context)
        rows = self.references(language, json.loads(parent["refs"]))
        checkpoint = None
        accent_mode = self.retry_accent_mode(language, parent) if parent["model"] == "f5" else None
        if accent_mode:
            checkpoint = self.accent_checkpoint(accent_mode)
            marked = mark_lapshi(text) if accent_mode == "lapshi-plus" else text
            trials = [(seed, variant, [(marked, 0)]) for seed, variant, _ in trials]
        elif parent["checkpoint_id"]:
            checkpoints = self.rows("SELECT * FROM checkpoints WHERE id=? AND language=? AND model=?",
                                    (parent["checkpoint_id"], language, parent["model"]))
            if not checkpoints:
                raise ValueError("Веса исходного дубля больше недоступны.")
            checkpoint = checkpoints[0]
        model = parent["model"]
        if not availability(self.demo, language, bool(checkpoint)).get(model, {}).get("ready"):
            raise ValueError("Модель исходного дубля сейчас недоступна.")
        inherited_stress = not accent_mode and model == "f5" and (
            parent["variant"] == "stress" or bool(parent["retry_stress"]))
        if inherited_stress:
            variants = dict(synthesis_variants(text, language, model, {"stress": True}))
            marked = variants.get("stress", variants["baseline"])
            trials = [(seed, variant, marked) for seed, variant, _ in trials]
        jobs = [(model, checkpoint, seed, variant, phrases)
                for seed, variant, phrases in trials]
        comparison_id = parent["comparison_id"]
        if comparison_id and not self.rows("SELECT 1 FROM recordings WHERE language=? AND id=?",
                                           (language, comparison_id)):
            comparison_id = None
        offset = (parent["comparison_start"] or 0) + max(0, index - radius)
        return self.enqueue_synthesis(language, text, rows, parent["profile"], jobs, comparison_id,
                                      parent_id=key, comparison_start=offset if comparison_id else None,
                                      retry_focus=min(index, radius), retry_stress=inherited_stress)

    def retry_accent_mode(self, language, parent):
        """Inherit saved accent policy, following legacy retry links within the same workspace."""
        seen = set()
        while parent and parent["id"] not in seen:
            if parent["accent_mode"]:
                return parent["accent_mode"]
            if parent["variant"] == "lapshi-accent":
                return "lapshi-plus"
            if parent["variant"] != "word-retry" or not parent["baseline_id"]:
                break
            seen.add(parent["id"])
            rows = self.rows("""SELECT id,accent_mode,variant,baseline_id FROM jobs
                                WHERE id=? AND language=? AND demo=? AND model='f5' AND kind='synthesis'""",
                             (parent["baseline_id"], language, int(self.demo)))
            parent = rows[0] if rows else None
        return None

    def execute(self, key, model, language, rows, profile, mode, operation,
                checkpoint_id=None, comparison_id=None, seed=None, variant=None, deadline=None, *, accent_mode=None):
        """Run one queued operation, recording final status and MLflow metadata."""
        started = monotonic()
        run_id = None
        try:
            run_id = self.tracking.start(model, language, len(rows), self.demo, mode, profile, checkpoint_id)
            if variant:
                self.tracking.client.log_param(run_id, "prosody_variant", variant)
                self.tracking.client.log_param(run_id, "prosody_version", PROSODY_VERSION)
                self.tracking.client.log_param(run_id, "revolver_version", REVOLVER_VERSION)
            if model in MLX_VOICES:
                self.tracking.client.log_param(run_id, "model_revision", VOICE_REVISIONS[model][1])
            if seed is not None:
                self.tracking.client.log_param(run_id, "seed", seed)
            if comparison_id:
                self.tracking.client.log_param(run_id, "comparison_id", comparison_id)
            if accent_mode:
                self.tracking.client.log_param(run_id, "accent_mode", accent_mode)
                self.tracking.client.log_param(run_id, "builtin_checkpoint", "f5-accent-tune")
            self.write("UPDATE jobs SET status='running', run_id=? WHERE id=?", (run_id, key))
            duration = operation()
            generated_in = monotonic() - started
            if not self.demo and (comparison_id or variant == "word-retry"):
                try:
                    if variant == "word-retry":
                        self.score_fragment(key, language, run_id)
                    else:
                        self.score_comparison(key, language, comparison_id, run_id, deadline=deadline)
                except Exception as exc:
                    print(f"Comparison score unavailable for {key}: {type(exc).__name__}", flush=True)
            self.tracking.finish(run_id, generated_in, duration)
            self.write("UPDATE jobs SET status='done', seconds=? WHERE id=?", (duration, key))
        except Exception as exc:
            if run_id:
                try:
                    self.tracking.finish(run_id, monotonic() - started, failed=True)
                except Exception as tracking_error:
                    print(f"MLflow: {tracking_error}", flush=True)
            print(f"Job {key} failed: {type(exc).__name__}", flush=True)
            if deadline is not None and monotonic() >= deadline:
                message = "Время автосинтеза истекло."
            else:
                message = str(exc) if isinstance(exc, ValueError) else "Задание завершилось ошибкой. Подробности в терминале."
            self.write("UPDATE jobs SET status='failed', error=? WHERE id=?", (message[:300], key))

    def score_fragment(self, key, language, run_id):
        """Log word errors and contextual speaker similarity for the requested word."""
        job = self.rows("SELECT text, comparison_id, comparison_start, retry_focus FROM jobs WHERE id=?", (key,))[0]
        graph = self.cached_word_graph(language, key, job["text"])
        rate = graph["wordErrorRate"]
        if rate is not None:
            self.tracking.client.log_metric(run_id, "whisper_word_error_rate", rate)
            self.write("UPDATE jobs SET word_error_rate=? WHERE id=?", (rate, key))
        if job["comparison_id"] and job["retry_focus"] is not None:
            original = self.references(language, [job["comparison_id"]], include_evaluation=True)[0]
            with self.fragment_scoring:
                score = self.cached_fragment_score(language, key, job["text"], original["id"], original["text"],
                                                   job["comparison_start"], job["retry_focus"])
            if score["cosine"] is not None:
                encoder = speaker_encoder()
                self.tracking.client.log_metric(run_id, f"{encoder}_context_cosine", score["cosine"])
                self.write("UPDATE jobs SET speaker_cosine=?, speaker_encoder=? WHERE id=?",
                           (score["cosine"], encoder, key))

    def score_comparison(self, key, language, comparison_id, run_id, *, deadline=None):
        """Store optional word and speaker scores for an already generated WAV."""
        text = self.rows("SELECT text FROM jobs WHERE id=?", (key,))[0]["text"]
        metrics = measure(self.root, language, text, self.media / f"{comparison_id}.wav",
                          self.media / f"{key}.wav", **({"deadline": deadline} if deadline else {}))
        encoder = "eres2net" if "eres2net_encoder_cosine" in metrics else "xtts"
        cosine = metrics.get(f"{encoder}_encoder_cosine")
        self.write("UPDATE jobs SET word_error_rate=?, speaker_cosine=?, speaker_encoder=? WHERE id=?",
                   (metrics.get("whisper_word_error_rate"), cosine, encoder if cosine is not None else None, key))
        for name, value in metrics.items():
            self.tracking.client.log_metric(run_id, name, value)
        prompt = self.rows("SELECT reference_prompt FROM recordings WHERE id=? AND language=?",
                           (comparison_id, language))
        if prompt and prompt[0]["reference_prompt"] == "holdout" and (deadline is None or monotonic() < deadline):
            delivery = delivery_comparison(self.media / f"{comparison_id}.wav", self.media / f"{key}.wav")
            self.write("UPDATE jobs SET delivery_comparison=? WHERE id=?", (json.dumps(delivery), key))

    @capture_synthesis
    def run_synthesis(self, key, model, language, rows, profile, checkpoint, comparison_id,
                      seed, variant, phrases, *, deadline=None):
        """Render a baseline or bounded prosody variant and publish a validated WAV."""
        refs = [{"path": str(self.media / f"{row['id']}.wav"), "text": row["text"]}
                for row in (rows[:1] if model in SINGLE_REFERENCE_ENGINES else rows[:5])]

        def render(text, path, take_seed):
            """Run one normalized take using the requested engine and reference."""
            raw = path.with_name(f"raw-{path.name}")
            generate(model, {"output": str(raw), "text": text, "language": language,
                             "references": refs, "seed": take_seed,
                             "checkpoint": checkpoint["path"] if checkpoint else "",
                             "config": checkpoint["config"] if checkpoint else "",
                             "architecture": checkpoint.get("architecture") if checkpoint else None,
                             "deadline": deadline}, self.demo)
            normalize(raw, path, deadline=deadline)

        def render_candidate(text, path, index, number):
            """Assign a reproducible independent seed to each numbered take."""
            take_seed = (seed + 7919 * index + 1009 * number) % 2**32
            render(text, path, take_seed)

        def recognize_candidate(path):
            """Check the generated wording without adding a corpus recording."""
            return transcribe(self.root, language, path.read_bytes(), deadline=deadline)

        def operation():
            """Render segments, assemble with explicit rests, and validate the result."""
            with temporary_directory() as folder:
                work = Path(folder)
                if variant == "voice-sequence":
                    output, report = render_voice_sequence(
                        phrases, Path(refs[0]["path"]), work, render_candidate, recognize_candidate)
                    seconds = normalize(output, work / "final.wav", deadline=deadline)
                    publish_file(work / "final.wav", self.media / f"{key}.wav")
                    self.write("UPDATE jobs SET voice_sequence=? WHERE id=?", (json.dumps(report), key))
                    return seconds
                parts = []
                for index, (text, pause_ms) in enumerate(phrases):
                    part = work / f"part-{index}.wav"
                    render(text, part, seed)
                    parts.append((part, pause_ms))
                output = work / "output.wav"
                if len(parts) == 1:
                    output = parts[0][0]
                else:
                    self.join_prosody_parts(parts, work / "combined.wav")
                    normalize(work / "combined.wav", output, deadline=deadline)
                with wave.open(str(output)) as audio:
                    seconds = audio.getnframes() / audio.getframerate()
                publish_file(output, self.media / f"{key}.wav")
                return seconds

        mode = "checkpoint" if checkpoint else "reference"
        self.execute(key, model, language, refs, profile, mode, operation,
                     checkpoint["id"] if checkpoint else None, comparison_id, seed, variant, deadline=deadline,
                     accent_mode=checkpoint.get("accent_mode") if checkpoint else None)

    @staticmethod
    def join_prosody_parts(parts, output):
        """Join normalized PCM and place fixed silence between complete sentences."""
        with wave.open(str(output), "wb") as writer:
            writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            for part, pause_ms in parts:
                with wave.open(str(part)) as audio:
                    writer.writeframesraw(audio.readframes(audio.getnframes()))
                writer.writeframesraw(bytes(48 * pause_ms))

    def rate(self, language, key, score):
        """Persist a 1–5 listening score and send it to the related MLflow run."""
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError("Оценка должна быть от 1 до 5.")
        rows = self.rows("""SELECT run_id FROM jobs WHERE id=? AND kind='synthesis'
                            AND status='done' AND language=? AND demo=?""",
                         (key, language, int(self.demo)))
        if not rows:
            raise ValueError("Аудиорезультат не найден.")
        self.tracking.rate(rows[0]["run_id"], score)
        self.write("UPDATE jobs SET score=? WHERE id=?", (score, key))

    def assemble(self, language, ids):
        """Concatenate selected synthesis takes in order with short pauses."""
        if (not isinstance(ids, list) or not 2 <= len(ids) <= 12
                or not all(isinstance(key, str) for key in ids) or len(set(ids)) != len(ids)):
            raise ValueError("Выберите от 2 до 12 дублей в нужном порядке.")
        rows = self.rows("""SELECT id,text,seconds FROM jobs WHERE language=? AND status='done'
                            AND kind='synthesis' AND demo=?""", (language, int(self.demo)))
        sources = {row["id"]: row for row in rows}
        if any(key not in sources for key in ids):
            raise ValueError("Некоторые дубли недоступны.")
        key = uuid4().hex
        with temporary_directory() as folder:
            output = Path(folder) / "assembly.wav"
            inputs = []
            for source in ids:
                inputs.extend(["-i", str(self.media / f"{source}.wav")])
            filters = []
            for i in range(len(ids) - 1):
                filters.append(f"[{i}:a]apad=pad_dur=0.18[part{i}]")
            labels = "".join(f"[part{i}]" for i in range(len(ids) - 1)) + f"[{len(ids)-1}:a]"
            filters.append(f"{labels}concat=n={len(ids)}:v=0:a=1[out]")
            ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", str(output)])
            publish_file(output, self.media / f"{key}.wav")
        seconds = sum(sources[source]["seconds"] for source in ids) + 0.18 * (len(ids) - 1)
        text = " ".join(sources[source]["text"] for source in ids)
        self.write("""INSERT INTO jobs(id,kind,model,status,language,text,refs,demo,seconds)
                       VALUES(?,'assembly','assembly','done',?,?,?,?,?)""",
                   (key, language, text, json.dumps(ids), int(self.demo), seconds))
        return key

    def assemble_blocks(self, language, blocks):
        """Publish a WAV built from approved ranges and retain source provenance."""
        rows = self.rows("""SELECT id, seconds FROM jobs WHERE language=? AND status='done'
                            AND kind='synthesis' AND demo=?""", (language, int(self.demo)))
        parts = checked_blocks(blocks, {row["id"]: row for row in rows})
        key = uuid4().hex
        target = self.media / f"{key}.wav"
        with temporary_directory() as folder:
            output = Path(folder) / "blocks.wav"
            seconds = render_blocks(parts, self.media, output)
            publish_file(output, target)
        try:
            self.write("""INSERT INTO jobs(id,kind,model,status,language,text,refs,demo,seconds)
                           VALUES(?,'assembly','assembly','done',?,?,?,?,?)""",
                       (key, language, f"Сборка · {len(parts)} фрагм.", json.dumps(parts), int(self.demo), seconds))
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return key

    def trainer(self, model, language):
        """Find a trusted local training adapter configured by the operator."""
        if self.demo or model == "fish":
            return None
        if model == "f5" and (language != "ru" or not all(Path(item).is_file() for item in local_weights(model))):
            return None
        value = os.getenv(f"VOICE_{model.upper()}_TRAINER", "")
        path = Path(value)
        return path.resolve() if value and path.is_file() and os.access(path, os.X_OK) else None

    def train(self, language, model, profile):
        """Queue a trainer against an immutable profile and its parent checkpoint."""
        trainer = (self.trainer(model, language)
                   if isinstance(model, str) and model in MODELS and (self.demo or research_enabled()) else None)
        if not trainer:
            raise ValueError("Подключите локальный адаптер дообучения для этой модели.")
        if type(profile) is not int or profile < 1:
            raise ValueError("Сначала сохраните версию профиля.")
        saved = self.rows("SELECT refs FROM profiles WHERE version=? AND language=?", (profile, language))
        if not saved:
            raise ValueError("Версия профиля не найдена.")
        rows = self.references(language, json.loads(saved[0]["refs"]))
        if len(rows) < 2:
            raise ValueError("Для дообучения нужно как минимум две записи. Рецепт может требовать больше.")
        key = uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM profiles WHERE version=?", (profile,)).fetchone():
                raise ValueError("Версия профиля больше недоступна.")
            db.execute("""INSERT INTO jobs(id,kind,model,status,language,text,refs,demo,profile)
                          VALUES(?,'training',?,'queued',?,'',?,0,?)""",
                       (key, model, language, saved[0]["refs"], profile))
        self.worker.submit(self.run_training, key, model, language, rows, profile, trainer)
        return key

    def run_training(self, key, model, language, rows, profile, trainer):
        """Run a configured recipe and register its output without replacing prior checkpoints."""
        def operation():
            """Pass the corpus and parent to the recipe, then verify its output manifest."""
            output = self.root / "checkpoints" / key
            output.mkdir(parents=True)
            parents = self.rows("""SELECT path,config FROM checkpoints WHERE language=? AND model=?
                                   AND profile<=? ORDER BY profile DESC, rowid DESC LIMIT 1""",
                                (language, model, profile))
            with temporary_directory() as folder:
                corpus = Path(folder) / "corpus.zip"
                with ZipFile(corpus, "w") as archive:
                    self.write_corpus(archive, rows)
                command = {"model": model, "language": language, "profile": profile,
                           "corpus": str(corpus), "output_dir": str(output),
                           "parent": parents[0] if parents else None}
                with (output / "trainer.log").open("w") as log:
                    subprocess.run([str(trainer)], input=json.dumps(command), text=True,
                                   cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=8 * 3600, check=True)
            result = json.loads((output / "checkpoint.json").read_text())
            path = (output / result["checkpoint"]).resolve()
            config = (output / result["config"]).resolve() if result.get("config") else None
            if (not path.is_relative_to(output) or not path.is_file()
                    or (config and (not config.is_relative_to(output) or not config.is_file()))
                    or (model == "coqui" and not config)):
                raise ValueError("Адаптер вернул некорректный checkpoint.")
            self.write("INSERT INTO checkpoints VALUES(?,?,?,?,?,?)",
                       (key, model, language, profile, str(path), str(config) if config else ""))
            return None

        self.execute(key, model, language, rows, profile, "fine-tune", operation)
