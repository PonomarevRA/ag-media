"""Track sampled tab faces in one bounded, warm local OpenCV process."""

from voice_studio.mod_runtime import model_resource

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import monotonic

import cv2
import numpy as np

from voice_studio.vision.face_reid import MIN_MARGIN, MIN_MATCH, cosine, unit_vector
from voice_studio.vision.video_revolver import MODEL as YUNET, SFACE

MAX_IDENTITIES = 32
MAX_SESSIONS = 4
SESSION_TTL = 600
GEOMETRY_GAP_MS = 700


def iou(left, right):
    """Compare normalized boxes within a continuous sequence of frames."""
    x, y, w, h = left
    a, b, c, d = right
    intersection = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
    return intersection / max(1e-9, w * h + c * d - intersection)


@dataclass
class Face:
    box: list
    last_seen: int
    signature: tuple


class Gallery:
    """Hold only recent geometry and one normalized vector per face for a capture session."""

    def __init__(self, first_number):
        self.faces = {}
        self.next_number = first_number
        self.previous = None
        self.last_frame = -1
        self.touched = monotonic()

    def track(self, image, time_ms, models):
        """Match geometry in a continuous shot and use appearance after a cut or a gap."""
        width, height = image.shape[1], image.shape[0]
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
        continuous = self.previous is not None and 0 < time_ms - self.last_frame <= GEOMETRY_GAP_MS
        if continuous:
            delta = cv2.absdiff(self.previous, small)
            continuous = np.mean(delta > 45) < .36
        self.previous, self.last_frame = small, time_ms
        models.detector.setInputSize((width, height))
        _, rows = models.detector.detect(image)
        if rows is None:
            return []
        already_used = set()
        hits = []
        for detection in sorted(rows, key=lambda row: -row[-1])[:12]:
            box = normalized_box(detection, width, height)
            if not box:
                continue
            available = {number: face for number, face in self.faces.items() if number not in already_used}
            matched = geometry_match(available, box, time_ms) if continuous else None
            if matched:
                signature = self.faces[matched].signature
                if not signature:
                    signature = models.signature(image, detection)
            else:
                signature = models.signature(image, detection)
                matched = signature_match(available, signature)
            if matched is None:
                matched = self.next_number
                self.next_number += 1
            already_used.add(matched)
            self.faces[matched] = Face(box, time_ms, signature)
            hits.append({"id": matched, "box": box})
        trim_gallery(self.faces)
        return hits


def normalized_box(detection, width, height):
    """Clip the detector rectangle to the captured tab and normalize it to that space."""
    x, y, w, h = detection[:4]
    left, top = max(0.0, x) / width, max(0.0, y) / height
    right, bottom = min(width, x + w) / width, min(height, y + h) / height
    if right <= left or bottom <= top:
        return None
    return [round(float(value), 4) for value in (left, top, right - left, bottom - top)]


def geometry_match(faces, box, time_ms):
    """Maintain a recent single-shot track when its box overlaps sufficiently."""
    scores = [(iou(face.box, box), number) for number, face in faces.items()
              if 0 < time_ms - face.last_seen <= GEOMETRY_GAP_MS]
    if not scores:
        return None
    score, number = max(scores)
    return number if score >= .35 else None


def signature_match(faces, signature):
    """Recover a previously seen face only when a unique confident match exists."""
    if not signature:
        return None
    scores = sorted(((cosine(signature, face.signature), number) for number, face in faces.items()
                     if face.signature), reverse=True)
    if not scores:
        return None
    score, number = scores[0]
    next_score = scores[1][0] if len(scores) > 1 else -1
    return number if score >= MIN_MATCH and score - next_score >= MIN_MARGIN else None


def trim_gallery(faces):
    """Bound descriptors without recycling participant IDs during a session."""
    if len(faces) > MAX_IDENTITIES:
        oldest = sorted(faces, key=lambda number: faces[number].last_seen)
        for number in oldest[:len(faces) - MAX_IDENTITIES]:
            del faces[number]


class Models:
    """Initialize only the inexpensive local vision models once."""

    def __init__(self):
        cv2.setNumThreads(1)
        self.detector = cv2.FaceDetectorYN.create(str(model_resource('vision.faces', 'builtin.yunet', YUNET)), "", (320, 240), .8, .3, 200)
        sface = model_resource('vision.face-embedding', 'builtin.sface', SFACE)
        self.restore = sface.is_file()
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "") if self.restore else None

    def signature(self, image, detection):
        """Use aligned crops only for sufficiently clear faces."""
        if self.recognizer is None or min(detection[2:4]) < 28:
            return ()
        try:
            aligned = self.recognizer.alignCrop(image, detection)
            return unit_vector(self.recognizer.feature(aligned).reshape(-1).tolist())
        except cv2.error:
            return ()


def process_frames(models, sessions, command):
    """Process a temporary frame batch supplied by the parent, using absolute tab time."""
    now = monotonic()
    for key in list(sessions):
        if now - sessions[key].touched > SESSION_TTL:
            del sessions[key]
    key = command["session"]
    first_number = command["firstNumber"]
    tracking_reset = key not in sessions and first_number > 1
    if key not in sessions and len(sessions) >= MAX_SESSIONS:
        del sessions[min(sessions, key=lambda value: sessions[value].touched)]
    if key not in sessions:
        sessions[key] = Gallery(first_number)
    gallery = sessions[key]
    gallery.touched = now
    frames = []
    base, step = command["startMs"], command["stepMs"]
    for index, file in enumerate(sorted(Path(command["folder"]).glob("frame-*.jpg"))):
        time_ms = base + index * step
        image = cv2.imread(str(file))
        faces = gallery.track(image, time_ms, models) if image is not None else []
        frames.append({"timeMs": time_ms, "faces": faces})
    return {
        "faceFrames": frames, "sface": models.restore, "nextNumber": gallery.next_number,
        "trackingReset": tracking_reset,
    }


def main():
    """Serve newline-delimited parent requests without persisting media or embeddings."""
    models = Models()
    sessions = {}
    for line in sys.stdin:
        try:
            result = process_frames(models, sessions, json.loads(line))
        except (ValueError, KeyError, cv2.error):
            result = {"error": "Не удалось обработать кадры."}
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
