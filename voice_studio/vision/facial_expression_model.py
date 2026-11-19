"""Classify visible facial expression using the pinned OpenCV Zoo model locally."""

from voice_studio.mod_runtime import model_resource, is_replacement

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from voice_studio.vision.static_face_evidence import face_crop_signature, MAX_SAMPLES, MAX_TRACKS
from voice_studio.vision.facial_expressions import LABELS, MODEL_ID, MODEL_SHA256, SAMPLE_STEP_MS, THRESHOLD


class ExpressionClassifier:
    """Reuse one CPU network and align YuNet's five measured facial landmarks."""

    def __init__(self, path):
        path = model_resource('vision.expressions', 'builtin.mobilefacenet-expression', path)
        self.samples = []
        self.last_sample = {}
        self.status = "unavailable"
        self.network = None
        if path and Path(path).is_file():
            try:
                self.network = load_network(str(path))
                self.status = "ok"
            except (OSError, ValueError, cv2.error):
                self.status = "failed"

    def observe(self, frame_number, frame, detection, track):
        """Measure at most about two frames per second per track, retaining abstentions."""
        time_ms = frame_number * 40
        if len(self.samples) >= MAX_SAMPLES or (track not in self.last_sample and len(self.last_sample) >= MAX_TRACKS):
            self.status = "partial"
            return
        if self.network is None or time_ms - self.last_sample.get(track, -SAMPLE_STEP_MS) < SAMPLE_STEP_MS:
            return
        self.last_sample[track] = time_ms
        row = {"timeMs": time_ms, "track": track, "label": "unknown", "confidence": 0,
               "status": "unknown", "probabilities": {}}
        try:
            if len(detection) < 15 or min(detection[2:4]) < 32 or detection[-1] < .85:
                row["reason"] = "small-or-low-quality-face"
            else:
                signature = face_crop_signature(frame, detection)
                if signature is not None:
                    row["faceCropSignature"] = signature
                blob = aligned_blob(frame, detection)
                self.network.setInput(blob, "data")
                logits = self.network.forward("label").reshape(-1)
                if logits.size != len(LABELS) or not np.isfinite(logits).all():
                    raise ValueError("Invalid facial expression model output")
                scores = np.exp(logits.astype(np.float64) - np.max(logits))
                probabilities = scores / scores.sum()
                winner = int(np.argmax(probabilities))
                confidence = float(probabilities[winner])
                row.update(probabilities={label: round(float(value), 6) for label, value in zip(LABELS, probabilities)},
                           confidence=round(confidence, 6), rawLabel=LABELS[winner])
                if confidence >= THRESHOLD:
                    row.update(label=LABELS[winner], status="ok")
                else:
                    row["reason"] = "low-confidence"
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            row["reason"] = "inference-failed"
            self.status = "partial"
        self.samples.append(row)

    def data(self, aliases):
        """Attach only the final preexisting face IDs; classification never changes identity."""
        return {"status": self.status, "model": MODEL_ID, "modelSha256": MODEL_SHA256,
                "samples": [{**row, "faceId": aliases[row["track"]]} for row in self.samples
                            if row["track"] in aliases]}


@lru_cache(maxsize=1)
def load_network(path):
    """Keep weights loaded between bounded visual windows in the native worker."""
    if (not is_replacement('vision.expressions', 'builtin.mobilefacenet-expression')
            and sha256(Path(path).read_bytes()).hexdigest() != MODEL_SHA256):
        raise ValueError("Facial expression model checksum differs from the pinned checkpoint")
    network = cv2.dnn.readNet(path)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return network


def aligned_blob(frame, detection):
    """Fit a nonreflective similarity to the publisher's 112 px landmark template."""
    source = np.asarray(detection[4:14], dtype=np.float64).reshape(5, 2)
    if not np.isfinite(source).all():
        raise ValueError("Nonfinite facial landmarks")
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
    # Solve the four similarity parameters directly; robust estimators would introduce randomness.
    matrix = np.zeros((10, 4))
    values = source.reshape(-1)
    for index, (x, y) in enumerate(target):
        matrix[index * 2] = (x, -y, 1, 0)
        matrix[index * 2 + 1] = (y, x, 0, 1)
    a, b, tx, ty = np.linalg.lstsq(matrix, values, rcond=None)[0]
    # The publisher fits template-to-detection then inverts, preserving its least-squares convention.
    inverse = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])
    transform = np.linalg.inv(inverse)[:2]
    aligned = cv2.warpAffine(frame, transform, (112, 112))
    return cv2.dnn.blobFromImage(aligned, scalefactor=1 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=True)
