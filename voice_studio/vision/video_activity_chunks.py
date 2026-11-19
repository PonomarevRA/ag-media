"""Accumulate independent Light-ASD windows on the video playback clock."""

from collections import Counter, defaultdict
from itertools import combinations

from voice_studio.vision.face_reid import FaceIdentityGallery, read_face_references

STEP = 40
CONTEXT_MS = 800


def activity_windows(duration_ms):
    """Use 5 s cores on short clips and 10 s cores on longer clips, with context."""
    size = 5000 if duration_ms <= 30_000 else 10_000
    if duration_ms <= 10_000:
        size = 10_000
    for start in range(0, duration_ms, size):
        end = min(duration_ms, start + size)
        yield start, end, max(0, start - CONTEXT_MS), min(duration_ms, end + CONTEXT_MS)


def iou(left, right):
    """Measure overlap of normalized boxes without native dependencies in the app."""
    x, y, w, h = left
    a, b, u, v = right
    intersection = max(0, min(x + w, a + u) - max(x, a)) * max(0, min(y + h, b + v) - max(y, b))
    union = w * h + u * v - intersection
    return intersection / union if union else 0


def match_overlap(previous, current, id_index):
    """Link IDs only on multiple overlapping frames with a unique mutual match."""
    votes = Counter()
    for before, after in zip(previous, current):
        frame_pairs = set()
        for left in before:
            for right in after:
                if iou(left[1:5], right[1:5]) >= .65:
                    frame_pairs.add((left[id_index], right[id_index]))
        votes.update(frame_pairs)
    old = defaultdict(set)
    new = defaultdict(set)
    for (left, right), count in votes.items():
        if count >= 3:
            old[left].add(right)
            new[right].add(left)
    linked = {}
    for right, candidates in new.items():
        if len(candidates) == 1:
            left = next(iter(candidates))
            if len(old[left]) == 1:
                linked[right] = left
    return linked


class ActivityTimeline:
    """Keep core frames once; align tracks by overlap and faces by source-local evidence."""

    def __init__(self, duration_ms):
        self.frames = [[] for _ in range((duration_ms + STEP - 1) // STEP)]
        self.attributes = {}
        self.outfits = {}
        self.photos = {}
        self.shots = []
        self.cuts = []
        self.matches = []
        self.distinct = set()
        self.estimated = []
        self.samples = 0
        self.next_track = 1
        self.next_face = 1
        self.status = "unavailable"
        self.spans = {}
        self.person_samples = []
        self.person_windows = []
        self.expression_samples = []
        self.expression_statuses = []
        self.face_gallery = FaceIdentityGallery()

    def add(self, report, start, end, scan_start, cue_starts):
        """Add core frames, attributes and per-cue counts in source coordinates."""
        local = report["timeline"]["frames"]
        first = (start - scan_start) // STEP
        stop = min(len(local), first + (end - start + STEP - 1) // STEP)
        prior = self.frames[scan_start // STEP:start // STEP]
        tracks = match_overlap(prior, local[:first], 0)
        faces = match_overlap(prior, local[:first], 6)
        core = local[first:stop]
        references, quality = read_face_references(report.pop("_faceSignatures", {}), with_quality=True)
        present = {hit[6] for hits in core for hit in hits}
        references = {face: vectors for face, vectors in references.items() if face in present}
        # Conservatively keep co-visible visual instances separate, including
        # live faces and printed portraits. Similar descriptors alone do not
        # justify collapsing those instances into one identity hypothesis.
        distinct = set()
        for hits in local:
            visible = sorted({hit[6] for hit in hits})
            distinct.update(combinations(visible, 2))
        distinct.update(tuple(sorted(pair)) for pair in report["faceTracking"].get("distinctFaces", ()))
        returning = self.face_gallery.match(references, faces, distinct, quality)
        faces.update({local_face: face for local_face, (face, _) in returning.items()})
        for index, hits in enumerate(core, start // STEP):
            remapped = []
            for track, x, y, w, h, active, face in hits:
                if track not in tracks:
                    tracks[track] = self.next_track
                    self.next_track += 1
                if face not in faces:
                    faces[face] = self.next_face
                    self.next_face += 1
                remapped.append([tracks[track], x, y, w, h, active, faces[face]])
            self.frames[index] = remapped
        self.face_gallery.remember(faces, references, quality)
        for local_face, (face, score) in returning.items():
            track = next((hit[0] for hits in core for hit in hits if hit[6] == local_face), None)
            if track in tracks and len(self.matches) < 512:
                self.matches.append({"track": tracks[track], "face": face,
                                     "cosine": round(score, 3), "basis": "cross-window-face"})
        self._expressions(report.get("facialExpressions"), start, end, scan_start, tracks, faces)
        self._editor(report, start, end, scan_start, tracks, faces)
        for span in report["spans"]:
            original = cue_starts.get(span["start_ms"])
            if original is not None:
                self._span(span, original, scan_start, tracks, faces)
        return min(end, start + len(core) * STEP)

    def _expressions(self, layer, start, end, offset, tracks, faces):
        """Discard context samples and apply exactly the geometry's track/face ID mapping."""
        if not isinstance(layer, dict):
            self.expression_statuses.append("unavailable")
            return
        self.expression_statuses.append(layer.get("status", "unavailable"))
        for row in layer.get("samples", []):
            time = row["timeMs"] + offset
            track, face = tracks.get(row["track"]), faces.get(row["faceId"])
            if start <= time < end and track is not None and face is not None:
                self.expression_samples.append({**row, "timeMs": time, "track": track, "faceId": face})

    def _span(self, span, original, offset, tracks, faces):
        """Merge counts for the exact same track across non-overlapping parts of a cue."""
        start, end = original
        merged = self.spans.setdefault(start, {"start_ms": start, "end_ms": end, "candidates": {}})
        for row in span["candidates"]:
            track = tracks.get(row["track"])
            face = faces.get(row["face"])
            if track is None or face is None:
                continue
            sample = row["representative"]
            if sample:
                sample = {**sample, "time_ms": sample["time_ms"] + offset}
            before = merged["candidates"].get(track)
            if before is None:
                merged["candidates"][track] = {**row, "track": track, "face": face,
                    "track_start_ms": row["track_start_ms"] + offset, "track_end_ms": row["track_end_ms"] + offset,
                    "representative": sample}
            else:
                if sample and (not before["representative"] or row["mean_logit"] > before["mean_logit"]):
                    before["representative"] = sample
                    before["mean_logit"] = row["mean_logit"]
                before["frames"] += row["frames"]
                before["positive_frames"] += row["positive_frames"]
                before["track_end_ms"] = max(before["track_end_ms"], row["track_end_ms"] + offset)

    def _editor(self, report, start, end, offset, tracks, faces):
        """Keep only real cuts within this core and apply the same mapping to profiles."""
        editor = report["editorTimeline"]
        self._people(editor.get("personPilot"), start, end, offset)
        tracking = report["faceTracking"]
        if tracking.get("status") == "ok":
            self.status = "ok"
        self.samples += tracking.get("samples", 0)
        for shot in editor["shots"]:
            time = shot["startMs"] + offset
            if start <= time < end and (shot["startMs"] or not offset):
                saved = {**shot, "startMs": time}
                if len(self.shots) >= 60:
                    saved.pop("jpeg", None)
                self.shots.append(saved)
        self.cuts.extend(cut + offset for cut in tracking.get("identityCutTimesMs", ())
                         if start <= cut + offset < end and cut)
        for key, target in (("facePhotos", self.photos), ("faceAttributes", self.attributes),
                            ("clothingByFace", self.outfits)):
            for face, value in editor.get(key, {}).items():
                mapped = faces.get(int(face))
                if mapped is None:
                    continue
                if isinstance(value, dict):
                    shifted = dict(value)
                    if "timeMs" in value:
                        shifted["timeMs"] += offset
                    if "sampleTimesMs" in value:
                        shifted["sampleTimesMs"] = [time + offset for time in value["sampleTimesMs"]]
                    target[mapped] = {**shifted, **target.get(mapped, {})}
                else:
                    target.setdefault(mapped, value)
        for left, right in tracking.get("distinctFaces", ()):
            if left in faces and right in faces and faces[left] != faces[right]:
                self.distinct.add(tuple(sorted((faces[left], faces[right]))))
        for match in tracking.get("matches", ()):
            if match["track"] in tracks and match["face"] in faces and len(self.matches) < 512:
                self.matches.append({**match, "track": tracks[match["track"]], "face": faces[match["face"]]})
        for frame in report["timeline"].get("estimatedFrames", ()):
            time = offset + frame * STEP
            if start <= time < end:
                self.estimated.append(time // STEP)

    def _people(self, people, start, end, offset):
        """Expose only core person observations and keep diagnostics per window."""
        if people is None:
            return
        for row in people["samples"]:
            time = row["timeMs"] + offset
            if start <= time < end:
                self.person_samples.append({**row, "timeMs": time})
        summary = {key: value for key, value in people.items() if key != "samples"}
        self.person_windows.append({**summary, "startMs": start, "endMs": end})

    def data(self, duration_ms, *, available_end=None):
        """Publish the timeline and remapped metadata in the existing overlay schema."""
        count = (available_end + STEP - 1) // STEP if available_end is not None else len(self.frames)
        timeline = {"stepMs": STEP, "frames": self.frames[:count]}
        if self.estimated:
            timeline["estimatedFrames"] = list(self.estimated)
        tracking = {"status": self.status, "tracks": self.next_track - 1, "faces": self.next_face - 1,
                    "samples": self.samples, "matches": list(self.matches), "identityCuts": len(self.cuts),
                    "identityCutTimesMs": list(self.cuts), "distinctFaces": sorted(self.distinct)}
        editor = {"durationMs": duration_ms, "shots": list(self.shots), "facePhotos": dict(self.photos),
                  "faceAttributes": dict(self.attributes), "clothingByFace": dict(self.outfits)}
        if self.person_windows:
            editor["personPilot"] = {"samples": list(self.person_samples), "windows": list(self.person_windows)}
        return {"timeline": timeline, "faceTracking": tracking, "editorTimeline": editor,
                "facialExpressions": {"samples": list(self.expression_samples),
                                      "windowStatuses": list(self.expression_statuses)}}
