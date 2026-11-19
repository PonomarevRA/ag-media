"""Match separate visual tracks within one upload using bounded face signatures."""

from voice_studio.mod_runtime import model_resource, is_replacement

from math import isfinite, sqrt
from itertools import combinations

MIN_MATCH = 0.54
MIN_MARGIN = 0.10
MIN_SAMPLES = 2
MAX_SAMPLES_PER_TRACK = 12
MAX_SAMPLES = 384
DISTINCT_FACE = 0.34
MAX_GALLERY_FACES = 128
MAX_FACE_REFERENCES = 12
REFERENCE_QUALITY_RATIO = .5


def face_references(aliases, signatures, qualities=None):
    """Export bounded track references backed by at least two qualified observations.

    Optional qualities come from retained track previews. These descriptors and
    qualities are private worker transport data, never public analysis metadata.
    """
    grouped = {}
    for track, samples in signatures.items():
        vector = reference(samples)
        if vector and track in aliases:
            quality = qualities.get(track, 1.) if qualities is not None else 1.
            grouped.setdefault(str(aliases[track]), []).append((vector, quality))
    for face, rows in grouped.items():
        if qualities is None:
            grouped[face] = [vector for vector, _ in rows]
            continue
        strongest = max(quality for _, quality in rows)
        grouped[face] = [{"vector": vector, "quality": quality} for vector, quality in rows
                         if quality >= strongest * REFERENCE_QUALITY_RATIO]
    return grouped


def read_face_references(payload, *, with_quality=False):
    """Normalize private vectors; optionally return their per-vector quality maps too.

    Legacy vector lists remain valid. Quality records carry the retained preview's
    pixel/confidence proxy, not a measurement of each individual signature frame.
    """
    references = {}
    qualities = {}
    if not isinstance(payload, dict):
        return (references, qualities) if with_quality else references
    remaining = MAX_SAMPLES
    for face, rows in list(payload.items())[:MAX_SAMPLES]:
        try:
            number = int(face)
        except (TypeError, ValueError):
            continue
        if number < 1 or not isinstance(rows, (list, tuple)):
            continue
        qualified = []
        quality_by_vector = {}
        for row in rows[:MAX_SAMPLES]:
            if remaining == 0:
                break
            remaining -= 1
            values = row.get("vector") if isinstance(row, dict) else row
            quality = row.get("quality", 1.) if isinstance(row, dict) else 1.
            if type(quality) not in (float, int) or not isfinite(quality) or quality <= 0:
                continue
            if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 512:
                continue
            if any(type(value) not in (float, int) or not isfinite(value) for value in values):
                continue
            norm = sqrt(sum(value * value for value in values))
            if isfinite(norm) and norm > 0:
                vector = tuple(value / norm for value in values)
                qualified.append(vector)
                quality_by_vector[vector] = max(quality, quality_by_vector.get(vector, 0))
        if qualified:
            references[number] = qualified
            qualities[number] = quality_by_vector
    return (references, qualities) if with_quality else references


def qualified_references(vectors, qualities):
    """Exclude views below half the strongest pixel/confidence quality from reference memory.

    Quality is the retained preview's minimum face-box side in proxy pixels times
    detector confidence. Preview replacement requires a substantial improvement,
    so this approximates the best track view rather than measuring every signature.
    Weak detections remain observations and may match; this retention heuristic
    keeps them from displacing clearer references without lowering cosine thresholds.
    """
    strongest = max((qualities.get(vector, 1.) for vector in vectors), default=0)
    return [vector for vector in vectors
            if qualities.get(vector, 1.) >= strongest * REFERENCE_QUALITY_RATIO]


def diverse_references(vectors):
    """Keep the first anchor and up to twelve diverse views using farthest-first selection.

    Matching remains a heuristic over retained prototypes, not a proof about every
    historical frame. A full gallery continues matching instead of fragmenting returns.
    """
    remaining = list(dict.fromkeys(vectors))
    if len(remaining) <= MAX_FACE_REFERENCES:
        return remaining
    selected = [remaining.pop(0)]
    while remaining and len(selected) < MAX_FACE_REFERENCES:
        index = min(range(len(remaining)),
                    key=lambda item: max(cosine(remaining[item], prior) for prior in selected))
        selected.append(remaining.pop(index))
    return selected


class FaceIdentityGallery:
    """Retain a bounded identity gallery for one upload, without persisting embeddings."""

    def __init__(self):
        self.references = {}
        self.qualities = {}

    def match(self, references, mapped, distinct, qualities=None):
        """Recover unique prior faces while preserving local co-visibility exclusions.

        Geometry-linked mappings are authoritative. A blocked best match abstains;
        it never falls through to a less similar person merely to allocate an ID.
        """
        qualities = qualities or {}
        resolved = dict(mapped)
        batch_references = {face: list(vectors) for face, vectors in self.references.items()}
        batch_quality = {face: dict(values) for face, values in self.qualities.items()}
        for local, face in mapped.items():
            if local in references:
                batch_references.setdefault(face, []).extend(references[local])
                quality = batch_quality.setdefault(face, {})
                for vector, value in qualities.get(local, {}).items():
                    quality[vector] = max(value, quality.get(vector, 0))
                batch_references[face] = qualified_references(batch_references[face], batch_quality[face])
        matches = {}
        for local, vectors in references.items():
            if local in resolved:
                continue
            local_quality = qualities.get(local, {})
            vectors = qualified_references(vectors, local_quality)
            ranked = []
            for face, prior in batch_references.items():
                if any(len(vector) != len(previous) for vector in vectors for previous in prior):
                    continue
                score = min(cosine(vector, previous) for vector in vectors for previous in prior)
                ranked.append((score, face))
            ranked.sort(reverse=True)
            if not ranked:
                continue
            score, face = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else -1
            if score < MIN_MATCH or score - runner_up < MIN_MARGIN:
                continue
            blocked = any(other != local and mapped_face == face
                          and tuple(sorted((local, other))) in distinct
                          for other, mapped_face in resolved.items())
            if not blocked:
                resolved[local] = face
                matches[local] = (face, score)
                # Later returns must agree with all retained qualified views in this
                # batch; low-resolution views cannot displace a clearer anchor.
                batch_references[face].extend(vectors)
                quality = batch_quality.setdefault(face, {})
                for vector, value in local_quality.items():
                    quality[vector] = max(value, quality.get(vector, 0))
                batch_references[face] = qualified_references(batch_references[face], batch_quality[face])
        return matches

    def remember(self, mapped, references, qualities=None):
        """Retain diverse qualified views and evict the least recently observed identity."""
        qualities = qualities or {}
        for local, vectors in references.items():
            if local not in mapped:
                continue
            face = mapped[local]
            prior = self.references.pop(face, [])
            quality = self.qualities.pop(face, {})
            for vector, value in qualities.get(local, {}).items():
                quality[vector] = max(value, quality.get(vector, 0))
            qualified = qualified_references([*prior, *vectors], quality)
            retained = diverse_references(qualified)
            self.references[face] = retained
            self.qualities[face] = {vector: quality.get(vector, 1.) for vector in retained}
            while len(self.references) > MAX_GALLERY_FACES:
                oldest = next(iter(self.references))
                self.references.pop(oldest)
                self.qualities.pop(oldest, None)


def unit_vector(values):
    """Normalize a feature vector once before cosine comparisons."""
    norm = sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values) if norm else ()


def cosine(left, right):
    """Compare already-normalized feature vectors."""
    return sum(a * b for a, b in zip(left, right))


def reference(samples):
    """Return a normalized track reference only when supported by repeated observations."""
    return unit_vector([sum(values) for values in zip(*samples)]) if len(samples) >= MIN_SAMPLES else ()


def distinct_faces(aliases, signatures):
    """Record strong non-matches between established groups without persisting embeddings."""
    refs = {}
    for track, samples in signatures.items():
        if vector := reference(samples):
            refs.setdefault(aliases[track], []).append(vector)
    return [[left, right] for left, right in combinations(sorted(refs), 2)
            if max(cosine(a, b) for a in refs[left] for b in refs[right]) < DISTINCT_FACE]


def group_tracks(tracks, signatures, *, qualities=None):
    """Associate disjoint returns, comparing all retained quality-qualified track references."""
    faces, aliases, matches = {}, {}, []
    for track_id, track in enumerate(tracks, 1):
        start, end = track[0][0], track[-1][0]
        samples = signatures.get(track_id, [])
        profile = reference(samples)
        quality = qualities.get(track_id, 1.) if qualities is not None else 1.
        eligible = []
        if profile:
            for face_id, history in faces.items():
                if history[-1][0] >= start or not history[0][1]:
                    continue
                prior = [vector for _, vector, _ in history if vector]
                quality_by_vector = {}
                for _, vector, value in history:
                    if vector:
                        quality_by_vector[vector] = max(value, quality_by_vector.get(vector, 0))
                retained = qualified_references(prior, quality_by_vector)
                score = min(cosine(profile, vector) for vector in retained)
                eligible.append((score, face_id))
        eligible.sort(reverse=True)
        face_id = len(faces) + 1
        if eligible:
            score, best = eligible[0]
            runner_up = eligible[1][0] if len(eligible) > 1 else -1
            if score >= MIN_MATCH and score - runner_up >= MIN_MARGIN:
                face_id = best
                matches.append({"track": track_id, "face": face_id, "cosine": round(score, 3)})
        aliases[track_id] = face_id
        faces.setdefault(face_id, []).append((end, profile, quality))
    return aliases, matches


class Signatures:
    """Collect a few aligned SFace vectors while YuNet walks the frames."""

    def __init__(self, weights):
        weights = model_resource('vision.face-embedding', 'builtin.sface', weights)
        import cv2
        self.cv2 = cv2
        self.recognizer = None
        self.status = "unavailable"
        self.samples = {}
        self.last_sample = {}
        self.count = 0
        if weights and weights.is_file():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(str(weights), "")
                self.status = "ok"
            except cv2.error:
                self.status = "failed"

    def feature(self, frame, detection):
        """Extract one qualified vector, sharing the normal sampling budget."""
        if (self.recognizer is None or self.count >= MAX_SAMPLES
                or min(detection[2:4]) < 30 or detection[-1] < .79):
            return ()
        self.count += 1
        try:
            aligned = self.recognizer.alignCrop(frame, detection)
            return unit_vector(self.recognizer.feature(aligned).reshape(-1).tolist())
        except self.cv2.error:
            self.recognizer = None
            self.status = "failed"
            return ()

    def check_cut(self, track, frame, detection):
        """Reject a spatial continuation on a suspected cut when recent face samples disagree."""
        prior = self.samples.get(track, [])
        if len(prior) < MIN_SAMPLES:
            return False, ()
        vector = self.feature(frame, detection)
        return bool(vector and max(cosine(vector, sample) for sample in prior[-3:]) < DISTINCT_FACE), vector

    def observe(self, track_id, frame_number, frame, detection, vector=()):
        """Sample early track appearances and occasional later frames without saving features."""
        if self.recognizer is None or (not vector and self.count >= MAX_SAMPLES):
            return
        samples = self.samples.setdefault(track_id, [])
        interval = 5 if len(samples) < 3 else 30
        if len(samples) >= MAX_SAMPLES_PER_TRACK or frame_number - self.last_sample.get(track_id, -30) < interval:
            return
        if min(detection[2:4]) < 30 or detection[-1] < .79:
            return
        vector = vector or self.feature(frame, detection)
        if vector:
            self.last_sample[track_id] = frame_number
            samples.append(vector)
