"""Immutable local media snapshots with indexed time ranges, text and relations."""

from contextlib import closing
from datetime import datetime, timezone
from hashlib import file_digest
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
from stat import S_ISREG
from threading import Lock
from voice_paths import publish_file, temporary_directory
from uuid import uuid4
import zlib

from voice_studio.graph.media_graph_interfaces import INTERFACE_KINDS, restore_interface_observation, restore_verified_panel
from voice_studio.graph.media_graph_health import data_health
from voice_studio.graph.media_graph_mod_extensions import snapshot_mod_provenance

MAX_WINDOW_MS = 5 * 60 * 1000
MAX_PAGE = 60
IDENTIFIER = re.compile(r"[a-f0-9]{32}")
SOURCE_CHECK_LOCK = Lock()


# Keep insertion order for equal starts while covering both interval bounds for counts.
SCHEMA = """
CREATE TABLE manifest (value TEXT NOT NULL);
CREATE TABLE nodes (
    seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, start_ms INTEGER, end_ms INTEGER,
    label TEXT NOT NULL, data TEXT NOT NULL, description TEXT NOT NULL
);
CREATE INDEX nodes_by_time ON nodes(start_ms, end_ms);
CREATE INDEX nodes_by_kind ON nodes(kind, start_ms, seq, end_ms);
CREATE TABLE edges (
    source INTEGER NOT NULL REFERENCES nodes(seq), relation TEXT NOT NULL,
    target INTEGER NOT NULL REFERENCES nodes(seq), PRIMARY KEY(source, relation, target)
) WITHOUT ROWID;
CREATE INDEX edges_by_target ON edges(target, relation, source);
CREATE VIEW edge_refs AS
    SELECT a.id AS source, e.relation, b.id AS target FROM edges e
    JOIN nodes a ON a.seq=e.source JOIN nodes b ON b.seq=e.target;
CREATE VIRTUAL TABLE search USING fts5(label, description, content='nodes', content_rowid='rowid', tokenize='unicode61');
CREATE TABLE face_chunks (bucket INTEGER PRIMARY KEY, payload BLOB NOT NULL);
"""


def encode(value):
    """Encode compact UTF-8 JSON suitable for database columns and local handoffs."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def signature(path):
    """Check local regular-file identity without hashing gigabytes on each screenshot."""
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError("Исходный файл недоступен.") from exc
    if not S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError("Нужен непустой локальный медиафайл.")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def identify(path):
    """Verify the selected file has a stable identity while calculating its digest."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("Укажи абсолютный путь к локальному медиафайлу.")
    path = path.resolve()
    before = signature(path)
    with path.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    if signature(path) != before:
        raise ValueError("Исходник изменился во время чтения.")
    return {"path": str(path), "signature": before, "sha256": digest}


@lru_cache(maxsize=32)
def verified_digest(path, expected_signature):
    """Hash a metadata-changed file once per stable signature, without modifying its snapshot."""
    if tuple(signature(path)) != expected_signature:
        raise ValueError("Исходник изменился во время чтения.")
    try:
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError("Исходный файл недоступен.") from exc
    if tuple(signature(path)) != expected_signature:
        raise ValueError("Исходник изменился во время чтения.")
    return digest


def source_verification(record):
    """Keep a cheap identity fast path; verify bytes after a remount or metadata-only change."""
    source = Path(record["path"])
    current = signature(source)
    if current == record["signature"]:
        return source, "signature"
    if len(record["signature"]) != 5 or current[2] != record["signature"][2]:
        raise ValueError("Исходник изменился. Построй новый снимок графа.")
    # Serialise first-time hashes so simultaneous frame/audio calls do not duplicate the I/O.
    with SOURCE_CHECK_LOCK:
        digest = verified_digest(source, tuple(current))
    if signature(source) != current or digest != record["sha256"]:
        raise ValueError("Исходник изменился. Построй новый снимок графа.")
    return source, "sha256"


def checked_source(record):
    """Permit only the recorded content, including after a verified metadata-only change."""
    return source_verification(record)[0]


def describe_source(record, tracks):
    """Expose the stored original and its graph tracks without returning filesystem IDs."""
    path = Path(record["path"])
    verification = None
    try:
        signature(path)
    except ValueError:
        status = "unavailable"
    else:
        try:
            _, verification = source_verification(record)
            status = "available"
        except ValueError:
            status = "signature-changed"
    return {"kind": "local-file", "path": str(path), "sha256": record["sha256"],
            "status": status, "verification": verification, "tracks": [
                {"id": row["id"], "kind": row["kind"], "streamIndex": json.loads(row["data"]).get("index")}
                for row in tracks
            ]}


def face_chunks(samples):
    """Group sparse geometry into independent one-second blocks for random access."""
    chunks = {}
    for stamp, faces in samples:
        chunks.setdefault(stamp // 1000, {})[stamp] = faces
    for bucket, frames in sorted(chunks.items()):
        yield bucket, zlib.compress(encode(frames).encode(), level=6)


def write_snapshot(path, manifest, graph):
    """Write an integrity-checked database in a private staging directory."""
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript(SCHEMA)
        db.execute("INSERT INTO manifest VALUES (?)", (encode(manifest),))
        ids = {node["id"]: seq for seq, node in enumerate(graph["nodes"], 1)}
        db.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            (ids[node["id"]], node["id"], node["kind"], node["startMs"], node["endMs"], node["label"],
             encode(node["data"]), node["data"].get("description", ""))
            for node in graph["nodes"]
        ))
        db.executemany("INSERT OR IGNORE INTO edges VALUES (?, ?, ?)", (
            (ids[source], relation, ids[target]) for source, relation, target in graph["edges"]
        ))
        db.execute("INSERT INTO search(search) VALUES ('rebuild')")
        db.executemany("INSERT INTO face_chunks VALUES (?, ?)", face_chunks(graph.get("faceFrames", [])))
    path.chmod(0o600)


def brief(row):
    """Project an indexed node without expanding its attributes or edges."""
    result = {"id": row["id"], "kind": row["kind"], "startMs": row["start_ms"],
              "endMs": row["end_ms"], "label": row["label"][:240]}
    if row["kind"] == "pause":
        facts = json.loads(row["data"])
        result.update(cutMs=facts["cutMs"], position=facts["position"], basis=facts["basis"])
    if row["kind"] in ("audio_music", "audio_background", "audio_intonation"):
        facts = json.loads(row["data"])
        result.update({key: facts[key] for key in ("basis", "confidence", "pitchHz", "semitones") if key in facts})
    if row["kind"] in ("camera_motion", "person_motion"):
        facts = json.loads(row["data"])
        result.update({key: facts[key] for key in ("horizontal", "vertical", "zoom", "depth", "basis", "faceId")
                       if key in facts})
    if row["kind"] == "shot":
        color = json.loads(row["data"]).get("color")
        if color:
            result["color"] = color
    if row["kind"] == "object_observation":
        facts = json.loads(row["data"])
        if "score" in facts:
            result["score"] = facts["score"]
    return result


def node_kind_filter(kind):
    """Resolve a category or exact kind to a fixed SQL predicate and bound arguments."""
    if kind is None:
        return "1", ()
    if kind == "ui":
        return (
            "(n.kind IN (?, ?, ?, ?, ?) OR (n.kind = 'object_profile' "
            "AND json_extract(n.data, '$.interfaceEvidence.type') IN (?, ?)))",
            (*INTERFACE_KINDS, "ui_panel", "ui_activity", "ui", "mobile-ui"),
        )
    return "n.kind = ?", (kind,)


def page_limit(limit, after):
    """Validate monotonic pagination parameters shared by all list tools."""
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE or type(after) is not int or after < 0:
        raise ValueError("Проверь limit (1–60) и after (целое число от нуля).")


def page(rows, limit):
    """Return one deterministic page and a cursor only if another page exists."""
    return {"nodes": [brief(row) for row in rows[:limit]],
            "nextAfter": rows[limit - 1]["cursor"] if len(rows) > limit else None}


class MediaGraphStore:
    """Publish atomic graph snapshots and answer bounded, read-only queries by ID."""

    def __init__(self, root):
        self.root = Path(root)

    def publish(self, source, media, graph, language, provenance):
        """Fix all observed relations to one immutable snapshot of the original source."""
        original = checked_source(source)
        identity = uuid4().hex
        manifest = {"schemaVersion": "video-as-code/1", "snapshotId": identity,
                    "title": original.name, "kind": media["kind"], "language": language,
                    "durationMs": media["durationMs"], "source": source,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "coverage": graph.get("coverage", {}), "faceSampleMs": graph.get("faceSampleMs"),
                    "speakerScope": graph.get("speakerScope", "source"), "provenance": snapshot_mod_provenance(provenance),
                    "uiActivity": graph.get("uiActivity", {"version": 1, "truncated": False}),
                    "modExtensions": graph.get("modExtensions", {"apiVersion": 1, "nodeCount": 0, "edgeCount": 0, "rejectedCount": 0, "contributors": []})}
        video = next((stream for stream in media.get("streams", []) if stream["kind"] == "video"), {})
        if video.get("width") and video.get("height"):
            manifest["frameSize"] = [video["width"], video["height"]]
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with temporary_directory() as folder:
            path = Path(folder) / "building.sqlite"
            write_snapshot(path, manifest, graph)
            checked_source(source)
            publish_file(path, self.root / (identity + ".sqlite"))
        return self.overview(identity)

    def database(self, identity):
        """Open exactly the named immutable snapshot as a read-only SQLite database."""
        if not isinstance(identity, str) or not IDENTIFIER.fullmatch(identity):
            raise ValueError("Некорректный ID снимка графа.")
        path = self.root / (identity + ".sqlite")
        if not path.is_file():
            raise ValueError("Снимок графа не найден.")
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'edge_refs'").fetchone():
            db.execute("CREATE TEMP VIEW edge_refs AS SELECT source, relation, target FROM main.edges")
        return closing(db)

    @staticmethod
    def manifest(db):
        """Read the snapshot envelope from its single manifest row."""
        return json.loads(db.execute("SELECT value FROM manifest").fetchone()[0])

    def overview(self, identity, *, include_source=False):
        """Return coverage and counts, with original-file details for local MCP clients."""
        with self.database(identity) as db:
            metadata = self.manifest(db)
            source = metadata.pop("source")
            metadata["sourceSha256"] = source["sha256"]
            if include_source:
                tracks = db.execute("SELECT id, kind, data FROM nodes "
                                    "WHERE kind IN ('audio_track', 'video_track') ORDER BY seq")
                metadata["source"] = describe_source(source, tracks)
            metadata["nodesByKind"] = dict(db.execute("SELECT kind, count(*) FROM nodes GROUP BY kind").fetchall())
            metadata["dataHealth"] = data_health(db, metadata["language"], metadata["nodesByKind"])
            ui_filter, ui_kinds = node_kind_filter("ui")
            metadata["nodesByCategory"] = {
                "ui": db.execute(f"SELECT count(*) FROM nodes n WHERE {ui_filter}", ui_kinds).fetchone()[0],
                "extension": metadata["nodesByKind"].get("extension", 0)
            }
            metadata["edgeCount"] = db.execute("SELECT count(*) FROM edges").fetchone()[0]
            metadata["storageBytes"] = (self.root / (identity + ".sqlite")).stat().st_size
        return metadata

    def list(self, limit=20, offset=0):
        """Discover saved snapshots without embedding their graphs in the response."""
        page_limit(limit, offset)
        files = sorted((path for path in self.root.glob("*.sqlite") if IDENTIFIER.fullmatch(path.stem)),
                       key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
        rows = []
        for path in files[offset:offset + limit]:
            with self.database(path.stem) as db:
                metadata = self.manifest(db)
            row = {key: metadata[key] for key in ("snapshotId", "title", "kind", "language", "durationMs", "createdAt")}
            if progress := metadata.get("provenance", {}).get("progress"):
                row["progress"] = progress
            rows.append(row)
        next_offset = offset + limit
        return {"snapshots": rows, "nextOffset": next_offset if next_offset < len(files) else None}

    def timeline(self, identity, start_ms, end_ms, kind=None, limit=30, after=0):
        """Read interval and point nodes intersecting a window of at most five minutes."""
        page_limit(limit, after)
        if not 0 <= start_ms < end_ms or end_ms - start_ms > MAX_WINDOW_MS:
            raise ValueError("Интервал должен иметь длину до пяти минут.")
        with self.database(identity) as db:
            anchor = -1
            if after:
                cursor = db.execute("SELECT start_ms FROM nodes WHERE rowid = ?", (after,)).fetchone()
                if cursor is None or cursor[0] is None:
                    raise ValueError("Для этой шкалы нужен курсор nextAfter из предыдущей страницы.")
                anchor = cursor[0]
            predicate, arguments = node_kind_filter(kind)
            rows = db.execute(f"""
                SELECT n.rowid AS cursor, n.* FROM nodes n WHERE start_ms < ? AND end_ms > ?
                  AND {predicate} AND (start_ms, n.rowid) > (?, ?)
                ORDER BY start_ms, n.rowid LIMIT ?
            """, (end_ms, start_ms, *arguments, anchor, after, limit + 1)).fetchall()
        return {"snapshotId": identity, **page(rows, limit)}

    def search(self, identity, query, kind=None, limit=25, after=0):
        """Find a literal word or contiguous phrase in labels, with no raw FTS syntax."""
        page_limit(limit, after)
        words = re.findall(r"[^\W_]+", query[:300], flags=re.UNICODE)
        if not words:
            raise ValueError("Введи слово или фразу для поиска.")
        phrase = '"' + " ".join(words) + '"'
        with self.database(identity) as db:
            predicate, arguments = node_kind_filter(kind)
            rows = db.execute(f"""
                SELECT n.rowid AS cursor, n.* FROM search JOIN nodes n ON n.rowid = search.rowid
                WHERE search MATCH ? AND {predicate} AND n.rowid > ?
                ORDER BY n.rowid LIMIT ?
            """, (phrase, *arguments, after, limit + 1)).fetchall()
        return {"snapshotId": identity, **page(rows, limit)}

    def node(self, identity, node_id, limit=30, offset=0):
        """Return one graph node and paginated incoming and outgoing references."""
        page_limit(limit, offset)
        with self.database(identity) as db:
            row = db.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row is None:
                raise ValueError("Узел графа не найден.")
            neighbors = neighbors_of(db, node_id, limit, offset)
            data = json.loads(row["data"])
            if row["kind"] == "object_track" and "representative" not in data:
                stamp = row["start_ms"]
                sample = object_sample_at(db, node_id, stamp)
                if sample:
                    facts = json.loads(sample["data"])
                    data["representative"] = {"observationId": sample["id"], "timeMs": stamp,
                                              "box": facts["box"], "score": facts.get("score")}
            if box := data.get("box"):
                x, y, w, h = box
                if row["kind"] == "object_observation":
                    data["nearFrameEdge"] = min(x, y, 1 - x - w, 1 - y - h) < .01
                if size := self.manifest(db).get("frameSize"):
                    width, height = size
                    left, top = round(x * width), round(y * height)
                    data["boxPx"] = [left, top, round((x + w) * width) - left, round((y + h) * height) - top]
        return {"snapshotId": identity, **brief(row), "label": row["label"], "data": data, **neighbors}

    def scene_evidence(self, identity, digest, language):
        """Reconstruct exact-source scene facts, including measured identity witnesses."""
        with self.database(identity) as db:
            metadata = self.manifest(db)
            if (metadata["source"]["sha256"] != digest or metadata["language"] != language
                    or metadata["kind"] != "video"):
                raise ValueError("Снимок сцены относится к другому исходнику или языку.")
            coverage = metadata.get("coverage", {}).get("scene", [])
            successful = {"ok", "sampled", "analyzed"}
            if not any(window.get("status") in successful for window in coverage):
                raise ValueError("В выбранном снимке нет измеренного покрытия сцены.")
            if any(window.get("status") not in successful | {"uncovered"}
                   or any(stage != "ok" for stage in window.get("statuses", {}).values())
                   for window in coverage):
                raise ValueError("В выбранном снимке есть незавершённые стадии сцены.")
            rows = db.execute("""
                SELECT id, kind, start_ms, label, data FROM nodes
                WHERE kind IN ('object_observation', 'object_alias', 'text_observation', ?, ?, ?)
                   OR (kind = 'frame' AND json_extract(data, '$.panelChecked') = 1)
                ORDER BY start_ms, rowid
            """, INTERFACE_KINDS)
            frames, regions, panels, aliases = {}, {}, [], []
            for row in rows:
                stamp = row["start_ms"]
                frame = frames.setdefault(stamp, {"timeMs": stamp, "objects": [], "text": []})
                data = json.loads(row["data"])
                if row["kind"] == "frame":
                    frame["panelChecked"] = True
                elif row["kind"] in ("object_observation", "object_alias"):
                    if data.get("source") == "interface-layout" and row["label"] == "mobile-ui":
                        panels.append((frame, data["box"]))
                        continue
                    evidence = {name: data[name] for name in ("box", "score", "basis", "detailBasis", "objectPalette",
                                                             "objectMatches", "objectPartMatches") if name in data}
                    item = {"label": row["label"], **evidence}
                    if row["kind"] == "object_alias":
                        aliases.append((frame, data["sourceIndex"], item))
                    else:
                        frame["objects"].append(item)
                elif row["kind"] == "text_observation":
                    frame["text"].append({"text": row["label"], "box": data["box"]})
                else:
                    restore_interface_observation(frame, regions, row["id"], row["kind"], data)
            for frame, index, item in aliases:
                frame["objects"].insert(index, item)
            for frame, box in panels:
                restore_verified_panel(frame, box)
            shot_colors, moodbook = [], []
            for shot in db.execute("SELECT start_ms, data FROM nodes WHERE kind='shot' ORDER BY start_ms, seq"):
                data = json.loads(shot["data"])
                if color := data.get("color"):
                    shot_colors.append({"startMs": shot["start_ms"], **color})
                if mood := data.get("moodbook"):
                    moodbook.append(mood)
            camera_motion = [
                {**json.loads(row["data"]), "startMs": row["start_ms"], "endMs": row["end_ms"]}
                for row in db.execute("SELECT start_ms, end_ms, data FROM nodes "
                                      "WHERE kind='camera_motion' ORDER BY start_ms, seq")
            ]
            ui_activity = {**(metadata.get('uiActivity') or {}),
                           "version": (metadata.get('uiActivity') or {}).get('version', 1),
                           "truncated": (metadata.get("uiActivity") or {}).get("truncated") is True,
                           "panels": [], "samples": []}
            for row in db.execute("SELECT kind, data FROM nodes WHERE kind IN ('ui_panel', 'ui_activity') ORDER BY start_ms, rowid"):
                key = "panels" if row["kind"] == "ui_panel" else "samples"
                facts = json.loads(row["data"])
                ui_activity["truncated"] = ui_activity["truncated"] or facts.get("truncated") is True
                ui_activity[key].append(facts)
        return {"status": "ok", "uiActivity": ui_activity, "frames": list(frames.values()), "shotColors": shot_colors, "moodbook": moodbook,
                "cameraMotion": camera_motion}, coverage

    def frame_context(self, identity, at_ms=None, node_id=None):
        """Resolve a timestamp and fetch only evidence valid at that time."""
        with self.database(identity) as db:
            metadata = self.manifest(db)
            target_id = node_id
            if at_ms is None and node_id:
                row = db.execute("SELECT start_ms, kind, data FROM nodes WHERE id = ?", (node_id,)).fetchone()
                if row:
                    at_ms = row["start_ms"]
                    if row["kind"] == "object_track":
                        representative = json.loads(row["data"]).get("representative", {})
                        at_ms = representative.get("timeMs", at_ms)
                    elif row["kind"] == "object_profile":
                        primary = json.loads(row["data"])["labels"][0]
                        at_ms, target_id = primary["timeMs"], primary["observationId"]
            if metadata["kind"] != "video":
                raise ValueError("У звуковой дорожки нет видеокадров.")
            if type(at_ms) is not int or not 0 <= at_ms < metadata["durationMs"]:
                raise ValueError("Укажи таймкод в границах исходника или ID узла с таймкодом.")
            context = context_at(db, metadata, at_ms)
            if target_id:
                mark_frame_target(db, context, target_id, at_ms)
        return metadata, at_ms, context

    def placement_context(self, identity, node_id, at_ms=None):
        """Resolve one real object observation by ID or by an exact sample on its track."""
        with self.database(identity) as db:
            metadata = self.manifest(db)
            track = db.execute("""
                SELECT start_ms, kind FROM nodes WHERE id=? AND kind IN ('object_track', 'object_observation')
            """, (node_id,)).fetchone()
            if metadata["kind"] != "video" or track is None:
                raise ValueError("Укажи ID трека предмета или его измерения.")
            stamp = track["start_ms"] if at_ms is None else at_ms
            if type(stamp) is not int or not 0 <= stamp < metadata["durationMs"]:
                raise ValueError("Таймкод выходит за пределы исходника.")
            sample = object_sample_at(db, node_id, stamp)
            if sample is None:
                raise ValueError("У предмета нет измерения на этом таймкоде. Открой связи sample_of.")
            return metadata, stamp, sample["id"], json.loads(sample["data"])["box"]

    def audio_context(self, identity, at_ms, node_id, duration_ms, track_id):
        """Resolve a short excerpt from a timed node or source-clock position."""
        with self.database(identity) as db:
            metadata = self.manifest(db)
            if node_id is not None:
                row = db.execute("SELECT start_ms, end_ms FROM nodes WHERE id = ?", (node_id,)).fetchone()
                if row is None or row[0] is None:
                    raise ValueError("Узел не связан с таймкодом.")
                at_ms = row[0]
                if duration_ms is None:
                    duration_ms = min(10000, row[1] - row[0])
            track = db.execute("""SELECT data FROM nodes WHERE kind = 'audio_track'
                                  AND (? IS NULL OR id = ?) ORDER BY rowid LIMIT 1""",
                               (track_id, track_id)).fetchone()
        if track is None:
            raise ValueError("Звуковая дорожка не найдена в снимке.")
        if type(at_ms) is not int or not 0 <= at_ms < metadata["durationMs"]:
            raise ValueError("Таймкод выходит за пределы исходника.")
        if duration_ms is not None and (type(duration_ms) is not int or not 1 <= duration_ms <= 10000):
            raise ValueError("Вырезка звука должна укладываться в 10 секунд.")
        length = min(duration_ms or 8000, metadata["durationMs"] - at_ms)
        return metadata, at_ms, length, json.loads(track["data"]).get("index")


def neighbors_of(db, node_id, limit, offset):
    """Read the local neighborhood in both directions without scanning entire edges."""
    rows = db.execute("""
        SELECT n.*, e.relation, 'out' AS direction FROM edge_refs e JOIN nodes n ON n.id = e.target
            WHERE e.source = ?
        UNION ALL
        SELECT n.*, e.relation, 'in' AS direction FROM edge_refs e JOIN nodes n ON n.id = e.source
            WHERE e.target = ?
        ORDER BY direction, relation, id LIMIT ? OFFSET ?
    """, (node_id, node_id, limit + 1, offset)).fetchall()
    return {"neighbors": [{**brief(row), "relation": row["relation"], "direction": row["direction"]}
                          for row in rows[:limit]],
            "nextOffset": offset + limit if len(rows) > limit else None}


def faces_at(db, metadata, stamp):
    """Return face records for the containing sample, including in older snapshots."""
    step = metadata.get("faceSampleMs") or 40
    sample = stamp - stamp % step
    row = db.execute("SELECT payload FROM face_chunks WHERE bucket = ?", (sample // 1000,)).fetchone()
    samples = json.loads(zlib.decompress(row[0])).get(str(sample), []) if row else []
    return [hit for hit in samples if isinstance(hit, dict) and "track" in hit and "faceId" in hit]


def caption_at(db, language, stamp):
    """Return the current caption and the only speaker IDs eligible for highlights."""
    row = db.execute("""
        SELECT * FROM nodes WHERE kind='utterance' AND start_ms <= ? AND end_ms > ?
        ORDER BY start_ms DESC, rowid LIMIT 1
    """, (stamp, stamp)).fetchone()
    if row is None:
        return None, set()
    overlap = db.execute("""
        SELECT n.id FROM edge_refs e JOIN nodes n ON n.id=e.source
        WHERE e.target=? AND e.relation='overlap_of' AND n.start_ms <= ? AND n.end_ms > ?
        ORDER BY n.start_ms DESC LIMIT 1
    """, (row["id"], stamp, stamp)).fetchone()
    if overlap or json.loads(row["data"]).get("overlapUnresolved"):
        participants = db.execute("SELECT target FROM edge_refs WHERE source=? AND relation='participant'",
                                  (overlap[0],)).fetchall() if overlap else []
        prefix = "Overlapping speech" if language == "en" else "Голоса пересекаются"
        return f"{prefix}: {row['label']}", {participant[0] for participant in participants}
    speaker = db.execute("""
        SELECT n.id, n.label FROM edge_refs e JOIN nodes n ON n.id=e.target
        WHERE e.source=? AND e.relation='spoken_by' ORDER BY n.id LIMIT 1
    """, (row["id"],)).fetchone()
    if speaker:
        return f"{speaker[1]}: {row['label']}", {speaker[0]}
    return row["label"], set()


def object_sample_at(db, node_id, stamp):
    """Read one exact object observation, directly or via a track or profile."""
    row = db.execute("""
        SELECT id, label, data FROM nodes WHERE id=? AND kind='object_observation' AND start_ms=?
        UNION ALL
        SELECT n.id, n.label, n.data FROM edge_refs e JOIN nodes n ON n.id=e.source
        WHERE e.target=? AND e.relation='sample_of' AND n.start_ms=?
        LIMIT 1
    """, (node_id, stamp, node_id, stamp)).fetchone()
    if row is not None:
        return row
    return db.execute("""
        SELECT n.id, n.label, n.data FROM edge_refs p
        JOIN edge_refs s ON s.target=p.source AND s.relation='sample_of'
        JOIN nodes n ON n.id=s.source AND n.start_ms=?
        WHERE p.target=? AND p.relation='part_of_profile'
        ORDER BY json_extract(n.data, '$.score') DESC, n.id LIMIT 1
    """, (stamp, node_id)).fetchone()


def mark_frame_target(db, context, node_id, stamp):
    """Highlight the selected timed object or UI evidence distinctly in the source frame."""
    item = db.execute("""
        SELECT id, label, data FROM nodes WHERE id=? AND start_ms=? AND kind IN ('object_alias', ?, ?, ?)
    """, (node_id, stamp, *INTERFACE_KINDS)).fetchone()
    if item is None:
        item = object_sample_at(db, node_id, stamp)
    if item is None:
        return
    data = json.loads(item["data"])
    box = data["box"]
    target = {"kind": "target", "label": item["label"], "box": box}
    if segment := data.get("segment"):
        target["segment"] = segment
    context["boxes"] = [target] + [
        mark for mark in context["boxes"] if not (mark["kind"] == "object" and mark["box"] == box)]
    context["targetObservationId"] = item["id"]



def person_samples_at(db, stamp, faces):
    """Read the nearest available body, head and gaze only for exact visible track IDs."""
    tracks = {face["track"] for face in faces}
    if not tracks:
        return {}
    rows = db.execute("""
        SELECT start_ms, data FROM nodes WHERE kind='person_observation' AND start_ms BETWEEN ? AND ?
        ORDER BY ABS(start_ms - ?), id
    """, (stamp - 300, stamp + 300, stamp))
    marks = {}
    for row in rows:
        facts = json.loads(row["data"])
        track = facts.get("track")
        if track not in tracks:
            continue
        sample = marks.setdefault(track, {"track": track, "faceId": facts.get("faceId")})
        distance = abs(row["start_ms"] - stamp)
        for key, window in (("box", 150), ("headPose", 250), ("gaze", 300)):
            if key in facts and distance <= window:
                sample.setdefault(key, facts[key])
    visible = {(face["track"], face["faceId"]) for face in faces}
    for row in db.execute("""
        SELECT data FROM nodes WHERE kind='person_motion' AND start_ms <= ? AND end_ms > ?
        ORDER BY start_ms DESC, seq DESC
    """, (stamp, stamp)):
        facts = json.loads(row["data"])
        key = facts.get("track"), facts.get("faceId")
        if key in visible and facts.get("depth") in {"foreground", "midground", "background"}:
            sample = marks.setdefault(key[0], {"track": key[0], "faceId": key[1]})
            if "depth" not in sample:
                sample.update(depth=facts["depth"], depthBasis=facts.get("depthBasis"))
    return marks


def person_label(label, person, language):
    """Label head and eye directions independently on the current face rectangle."""
    arrows = {"left": "←", "right": "→", "up": "↑", "down": "↓", "center": "•"}
    gaze = arrows.get(person.get("gaze", {}).get("horizontal"))
    if gaze:
        label += f" · {'Gaze' if language == 'en' else 'Взгляд'} {gaze}"
    pose = person.get("headPose", {})
    head = arrows.get(pose.get("horizontal"), "")
    tilt = arrows.get(pose.get("vertical"), "") if pose.get("vertical") in ("up", "down") else ""
    if head or tilt:
        direction = tilt if head == "•" and tilt else head + tilt
        label += f" · {'Head' if language == 'en' else 'Голова'} {direction}"
    depth = {"foreground": ("Передний план", "Foreground"),
             "midground": ("Средний план", "Midground"),
             "background": ("Задний план", "Background")}.get(person.get("depth"))
    if depth:
        label += " · " + depth[language == "en"]
        if person.get("depthBasis") == "relative-face-scale":
            label += " ≈"
    return label


def context_at(db, metadata, stamp):
    """Collect matching boxes, caption and shot label for one original frame."""
    caption, active_speakers = caption_at(db, metadata["language"], stamp)
    boxes = []
    face_label = "Face" if metadata["language"] == "en" else "Лицо"
    faces = faces_at(db, metadata, stamp)
    people = person_samples_at(db, stamp, faces)
    for face in faces:
        speaker_id = face.get("speakerId")
        binding = db.execute("SELECT label FROM nodes WHERE id = ?", (speaker_id,)).fetchone()
        active = face["active"] and speaker_id in active_speakers
        title = binding[0] if binding else f"{face_label} {face['faceId']}"
        sample = people.get(face["track"], {})
        boxes.append({"box": face["box"], "kind": "speaker" if active else "face",
                      "label": person_label(title, sample, metadata["language"])})
        if "box" in sample:
            boxes.append({"box": sample["box"], "kind": "body", "label": title})
    step = metadata.get("faceSampleMs")
    exact = stamp - stamp % (step or 40)
    kinds = {"object_observation": "object", "text_observation": "text", "person_observation": "body"}
    marks = db.execute("""
        SELECT * FROM nodes WHERE start_ms IN (?, ?) AND
          (kind IN ('object_observation', 'text_observation') OR (? AND kind = 'person_observation'))
        ORDER BY CASE kind WHEN 'text_observation' THEN 0 WHEN 'object_observation' THEN 1 ELSE 2 END, id LIMIT 24
    """, (exact, stamp, step is None)).fetchall()
    for row in marks:
        detail = json.loads(row["data"])
        if "box" in detail:
            boxes.append({"box": detail["box"], "label": row["label"], "kind": kinds[row["kind"]]})
    shot = db.execute("""
        SELECT label, data FROM nodes WHERE kind='shot' AND start_ms <= ? AND end_ms > ?
        ORDER BY start_ms DESC, seq DESC LIMIT 1
    """, (stamp, stamp)).fetchone()
    return {"boxes": boxes, "caption": caption, "shot": shot[0] if shot else None,
            "shotColor": json.loads(shot["data"]).get("color") if shot else None,
            "people": list(people.values())}
