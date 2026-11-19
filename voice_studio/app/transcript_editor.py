"""Store timed transcript material and revision-checked edits in the local studio."""

from contextlib import closing
import json
import re
from time import time
from uuid import uuid4
from voice_studio.core.media_limits import MAX_SOURCE_MS, MAX_TRANSCRIPT_BYTES, MAX_TRANSCRIPT_CUES, MAX_RAW_SPEAKER

from voice_studio.audio.dialogue_pauses import clean_dialogue_pauses
from voice_studio.vision.face_speakers import bind_faces
from voice_studio.speech.person_mentions import clean_mentions
from voice_studio.speech.media_transcript import render_srt, transcript_result
from voice_studio.vision.scene_transcript import clean_asr_correction
from voice_studio.speech.short_reply_review import BASIS as SHORT_REPLY_BASIS
from voice_studio.speech.speaker_context import resolve_speakers
from voice_studio.speech.speaker_identity import apply_confirmations
from voice_studio.speech.speaker_profiles import MAX_PROFILES, attach_ids, collect_profiles, cue_number
from voice_studio.speech.speech_overlap import mixed_label
from voice_studio.speech.voice_profile_public import public_voice_evidence

MAX_DOCUMENT_BYTES = MAX_TRANSCRIPT_BYTES
MAX_DURATION_MS = MAX_SOURCE_MS


class RevisionConflict(ValueError):
    """Reject stale writes from an editor that has not read the latest revision."""


class TranscriptEditor:
    """Manage the original cues and a sparse editor overlay in the existing studio database."""

    def __init__(self, connect):
        self.connect = connect
        with closing(connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS editable_transcripts (
                id TEXT PRIMARY KEY, language TEXT NOT NULL, title TEXT NOT NULL,
                cues TEXT NOT NULL, edits TEXT NOT NULL DEFAULT '{}',
                revision INTEGER NOT NULL DEFAULT 1, updated REAL NOT NULL
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(editable_transcripts)")}
            if "source_digest" not in columns:
                db.execute("ALTER TABLE editable_transcripts ADD COLUMN source_digest TEXT")
            if "dialogue_pauses" not in columns:
                db.execute("ALTER TABLE editable_transcripts ADD COLUMN dialogue_pauses TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS transcript_source ON editable_transcripts(language, source_digest)")
            db.execute("""CREATE TABLE IF NOT EXISTS speaker_identity_confirmations (
                language TEXT NOT NULL, source_digest TEXT NOT NULL, profile_id TEXT NOT NULL,
                confirmation TEXT NOT NULL, PRIMARY KEY(language, source_digest, profile_id)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS speaker_profile_snapshots (
                language TEXT NOT NULL, source_digest TEXT NOT NULL, profiles TEXT NOT NULL, updated REAL NOT NULL,
                PRIMARY KEY(language, source_digest)
            )""")

    def list(self, language, limit=100, offset=0):
        """Return lightweight summaries ordered by most recent edit."""
        with closing(self.connect()) as db:
            rows = db.execute("""SELECT id, title, revision, updated FROM editable_transcripts
                               WHERE language=? ORDER BY updated DESC, id DESC LIMIT ? OFFSET ?""",
                              (language, limit, offset))
            return [dict(row) for row in rows]

    def delete(self, language, key, revision):
        """Remove a confirmed revision of a saved transcript in its own language."""
        if type(revision) is not int or revision < 1:
            raise ValueError("Укажи версию материала.")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = read_row(db, language, key)
            if row["revision"] != revision:
                raise RevisionConflict("Материал изменился. Проверь свежую версию перед удалением.")
            db.execute("DELETE FROM editable_transcripts WHERE id=? AND language=?", (key, language))

    def create(self, language, title, segments, source_digest=None, dialogue_pauses=None):
        """Save timed text, speaker evidence and measured pauses without storing the uploaded media."""
        title = field_text(title, 160, "название")
        cues = clean_cues(segments)
        if source_digest is not None and (not isinstance(source_digest, str)
                                          or not re.fullmatch(r"[0-9a-f]{64}", source_digest)):
            raise ValueError("Некорректный идентификатор исходного файла.")
        edits = draft_edits(segments)
        duration = max(cue["endMs"] for cue in cues)
        pauses = dump(clean_dialogue_pauses(dialogue_pauses, duration)) if isinstance(dialogue_pauses, list) else None
        key = uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute("""INSERT INTO editable_transcripts
                          (id, language, title, cues, updated, source_digest, edits, dialogue_pauses)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                       (key, language, title, dump(cues), time(), source_digest, dump(edits), pauses))
        return self.get(language, key)

    def recover(self, language, digest, result):
        """Reuse confirmed speaker edits only for the exact same uploaded source."""
        with closing(self.connect()) as db:
            entries = db.execute("""SELECT id, revision, cues, edits FROM editable_transcripts
                                    WHERE language=? AND source_digest=? AND edits <> '{}'
                                    ORDER BY updated DESC, id""", (language, digest))
            sources = []
            for row in entries:
                sources.append({"id": row["id"], "revision": row["revision"],
                                "cues": json.loads(row["cues"]), "edits": json.loads(row["edits"])})
            has_confirmations = db.execute("SELECT 1 FROM sqlite_master WHERE name='speaker_identity_confirmations'").fetchone()
            confirmations = [json.loads(row[0]) for row in db.execute(
                "SELECT confirmation FROM speaker_identity_confirmations WHERE language=? AND source_digest=? ORDER BY profile_id",
                (language, digest))] if has_confirmations else []
        result["sourceDigest"] = digest
        graph = resolve_speakers(result["segments"], sources)
        if graph["decisions"]:
            result["speakerContextGraph"] = graph
            bind_faces(result)
        identities = apply_confirmations(result, confirmations)
        if graph["decisions"] or identities:
            result["srt"] = render_srt(result["segments"], language)
        return graph

    def confirm_identity(self, language, digest, confirmation):
        """Save a human-confirmed group, scoped to a single source and analysis language."""
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Некорректный идентификатор исходного файла.")
        with closing(self.connect()) as db, db:
            db.execute("""INSERT INTO speaker_identity_confirmations
                          (language, source_digest, profile_id, confirmation) VALUES (?, ?, ?, ?)
                          ON CONFLICT(language, source_digest, profile_id)
                          DO UPDATE SET confirmation=excluded.confirmation""",
                       (language, digest, confirmation["id"], dump(confirmation)))

    def get(self, language, key):
        """Read the current editable transcript and its generated subtitle text."""
        with closing(self.connect()) as db:
            result = material(read_row(db, language, key))
            if result["sourceDigest"] is not None:
                stored = db.execute("SELECT profiles FROM speaker_profile_snapshots WHERE language=? AND source_digest=?",
                                    (language, result["sourceDigest"])).fetchone()
                if stored:
                    result["speakerProfileIndex"] = json.loads(stored["profiles"])
            return result

    def remember_profiles(self, language, digest, result):
        """Persist references for this exact file, keeping editorial priority on every rerun."""
        if "speakerCount" not in result:
            return
        profiles = collect_profiles(result)
        if not profiles:
            return
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT profiles FROM speaker_profile_snapshots WHERE language=? AND source_digest=?",
                             (language, digest)).fetchone()
            previous = json.loads(row["profiles"]) if row else []
            attach_ids(profiles, previous)
            current_ids = {profile["id"] for profile in profiles}
            extras = [profile for profile in previous if profile["pinned"] and profile["id"] not in current_ids]
            saved = (profiles + extras)[:MAX_PROFILES]
            db.execute("""INSERT INTO speaker_profile_snapshots VALUES (?, ?, ?, ?)
                          ON CONFLICT(language, source_digest) DO UPDATE SET profiles=excluded.profiles,
                          updated=excluded.updated""", (language, digest, dump(saved), time()))
        result["speakerProfileIndex"] = profiles
        by_number = {profile["number"]: profile["id"] for profile in profiles}
        for cue in result["segments"]:
            number = cue_number(cue)
            if number in by_number and not mixed_label(cue):
                cue["speakerProfileId"] = by_number[number]

    def patch(self, language, data):
        """Apply one per-cue edit atomically and retain both the raw cue and its evidence."""
        key, revision, start = data.get("id"), data.get("revision"), data.get("startMs")
        if type(revision) is not int or revision < 1 or type(start) is not int:
            raise ValueError("Укажи версию материала и начало реплики в миллисекундах.")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = read_row(db, language, key)
            if row["revision"] != revision:
                raise RevisionConflict("Материал уже изменён. Открой свежую версию перед правкой.")
            cues = {cue["startMs"]: cue for cue in json.loads(row["cues"])}
            if start not in cues:
                raise ValueError("Реплика с таким таймкодом не найдена.")
            edits = json.loads(row["edits"])
            previous = edits.get(str(start), {})
            edited = apply_edit(cues, start, previous, edits, data)
            if edited != previous:
                edits[str(start)] = edited
                db.execute("""UPDATE editable_transcripts SET edits=?, revision=revision+1, updated=? WHERE id=?""",
                           (dump(edits), time(), key))
                row = read_row(db, language, key)
        return material(row)


def dump(value):
    """Encode editorial data losslessly for SQLite."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def field_text(value, limit, name, *, empty=False):
    """Normalize one single-line editorial field and enforce its size."""
    if not isinstance(value, str):
        raise ValueError(f"Некорректное поле «{name}».")
    value = " ".join(value.split())
    if len(value) > limit or not (value or empty) or any(ord(char) < 32 for char in value):
        raise ValueError(f"Некорректное поле «{name}».")
    return value


def clean_cues(segments):
    """Validate ordered unique timestamps and keep only transcript fields."""
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_TRANSCRIPT_CUES:
        raise ValueError(f"Материал должен содержать от 1 до {MAX_TRANSCRIPT_CUES} реплик.")
    cues = []
    previous = -1
    for row in segments:
        if not isinstance(row, dict):
            raise ValueError("Некорректная реплика.")
        start, end = row.get("startMs"), row.get("endMs")
        if type(start) is not int or type(end) is not int or not previous < start < end <= MAX_DURATION_MS:
            raise ValueError("Проверь порядок таймкодов реплик.")
        cue = {"startMs": start, "endMs": end, "text": field_text(row.get("text"), 5000, "текст реплики")}
        if type(row.get("isQuestion")) is bool:
            cue["isQuestion"] = row["isQuestion"]
        if cue["text"] == row.get("text") and isinstance(row.get("personMentions"), list):
            cue["personMentions"] = clean_mentions(cue["text"], row["personMentions"])
        original = row.get("asrText")
        if fix := clean_asr_correction(cue, original, row.get("asrCorrection")):
            cue.update(asrText=original, asrCorrection=fix)
        previous = start
        number = row.get("speaker")
        if "speaker" in row and (number is None or type(number) is int and 1 <= number <= MAX_RAW_SPEAKER):
            cue["speaker"] = number
        raw = row.get('rawSpeaker')
        if raw is None or type(raw) is int and 1 <= raw <= MAX_RAW_SPEAKER:
            if 'rawSpeaker' in row:
                cue['rawSpeaker'] = raw
        voice_id = row.get('voiceProfileId')
        if isinstance(voice_id, str) and re.fullmatch(r'[0-9a-f]{32}', voice_id):
            cue['voiceProfileId'] = voice_id
        if isinstance(row.get('voiceProfileEvidence'), dict):
            # Keeping abstention evidence prevents a raw window label being mistaken
            # for the source profile with the same number after saving/reloading.
            cue['voiceProfileEvidence'] = public_voice_evidence(row['voiceProfileEvidence'])
        profile_number = row.get("profileSpeakerNumber")
        if type(profile_number) is int and 1 <= profile_number <= 1000:
            cue["profileSpeakerNumber"] = profile_number
        profile_id = row.get("speakerProfileId")
        if isinstance(profile_id, str) and re.fullmatch(r"[0-9a-f]{32}", profile_id):
            cue["speakerProfileId"] = profile_id
        if row.get("overlap") is True:
            cue["overlap"] = True
            spans = clean_overlap_spans(row.get("overlaps"), start, end)
            if spans:
                cue["overlaps"] = spans
        visible = row.get("visualSpeaker")
        if type(visible) is int and 1 <= visible <= 100_000:
            cue["visualSpeaker"] = visible
        visual_number = row.get("visualSpeakerNumber")
        if "visualSpeaker" in cue and type(visual_number) is int and 1 <= visual_number <= 100_000:
            cue["visualSpeakerNumber"] = visual_number
        if "visualSpeaker" in cue:
            basis = row.get("visualSpeakerBasis")
            if basis in ("editorial-with-activity", SHORT_REPLY_BASIS):
                cue["visualSpeakerBasis"] = basis
            for key in ("visualFace", "linkedSpeakerNumber"):
                value = row.get(key)
                if type(value) is int and 1 <= value <= 100_000:
                    cue[key] = value
        cues.append(cue)
    return cues


def clean_overlap_spans(spans, start, end):
    """Keep only bounded, well-formed overlap intervals from a saved cue."""
    if not isinstance(spans, list):
        return []
    cleaned = []
    for span in spans[:64]:
        if not isinstance(span, dict):
            continue
        left, right, speakers = span.get("startMs"), span.get("endMs"), span.get("speakers")
        if type(left) is not int or type(right) is not int or not start <= left < right <= end:
            continue
        if not isinstance(speakers, list):
            continue
        ids = sorted({number for number in speakers if type(number) is int and 1 <= number <= 1000})
        if len(ids) > 1:
            cleaned.append({"startMs": left, "endMs": right, "speakers": ids})
    return cleaned


def draft_edits(segments):
    """Preserve explicit per-cue numbers and earlier evidence when saving a draft."""
    edits = {}
    by_start = {cue["startMs"]: cue for cue in segments}
    for cue in segments:
        number = cue.get("speakerNumber")
        if type(number) is not int or not 1 <= number <= 1000:
            continue
        start = cue["startMs"]
        evidence = cue.get("speakerNumberEvidence")
        context = evidence.get("contextStartMs") if isinstance(evidence, dict) else None
        if type(context) is not int or context >= start or context not in by_start:
            context = None
        edit = {"speakerNumber": number, "contextStartMs": context}
        if isinstance(evidence, dict) and evidence.get("source") == "agent":
            edit["source"] = "agent"
        edits[str(start)] = apply_edit(by_start, start, {}, {}, edit)
    return edits


def apply_edit(cues, start, previous, all_edits, data):
    """Validate the edited values and snapshot any earlier cue used as evidence."""
    edited = dict(previous)
    source = "agent" if data.get("source") == "agent" else "user"
    if "text" in data:
        text = field_text(data["text"], 5000, "текст реплики")
        if text == cues[start]["text"]:
            edited.pop("text", None)
        else:
            edited["text"] = text
    for field in ("speakerNumber", "speakerName"):
        if field not in data:
            continue
        value = data[field]
        if field == "speakerName":
            value = field_text(value, 100, "имя говорящего", empty=True)
        elif value is not None and (type(value) is not int or not 1 <= value <= 1000):
            raise ValueError("Номер спикера должен быть от 1 до 1000.")
        evidence_key = field + "Evidence"
        if not value:
            edited.pop(field, None)
            edited.pop(evidence_key, None)
            continue
        context = data.get("contextStartMs")
        evidence = {"source": source}
        if context is not None:
            if type(context) is not int or context >= start or context not in cues:
                raise ValueError("Укажи более раннюю реплику из этого материала.")
            reference = all_edits.get(str(context), {})
            evidence.update(contextStartMs=context, contextText=field_text(reference.get("text", cues[context]["text"]), 5000, "текст основания"))
        edited[field] = value
        edited[evidence_key] = evidence
    return edited


def read_row(db, language, key):
    """Resolve a transcript ID only inside the selected workspace language."""
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key):
        raise ValueError("Некорректный идентификатор материала.")
    row = db.execute("SELECT * FROM editable_transcripts WHERE id=? AND language=?", (key, language)).fetchone()
    if row is None:
        raise ValueError("Материал не найден.")
    return row


def material(row):
    """Overlay the edits and rebuild SRT/JSON from the same named cues."""
    cues = json.loads(row["cues"])
    edits = json.loads(row["edits"])
    for cue in cues:
        edit = edits.get(str(cue["startMs"]), {})
        if "text" in edit:
            cue.setdefault("asrText", cue["text"])
            cue.pop("asrCorrection", None)
            cue.pop("personMentions", None)
        cue.update(edit)
    result = transcript_result(cues, max(cue["endMs"] for cue in cues), row["language"])
    if row["dialogue_pauses"] is not None:
        result["dialoguePauses"] = json.loads(row["dialogue_pauses"])
    return {"id": row["id"], "title": row["title"], "revision": row["revision"], "updated": row["updated"],
            "sourceDigest": row["source_digest"], **result}
