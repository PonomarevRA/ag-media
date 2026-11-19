"""Use the scene detector for one still image, without creating video tracks or speaker identities."""

from time import perf_counter

import cv2

from voice_paths import model_path
from voice_studio.mod_runtime import is_replacement
from voice_studio.vision.scene_objects import ObjectDetector


def detect_image_objects(image, mode):
    """Return normalized COCO proposals and an explicit unavailable/failed status; never download weights."""
    started = perf_counter()
    weights = model_path("vision", "nanodet.onnx")
    status, objects, runs = "unavailable", [], 0
    if weights.is_file() or is_replacement('vision.objects', 'builtin.nanodet'):
        try:
            detector = ObjectDetector(weights)
            objects = detector.detect(image, sectors=True if mode == "full" else None)
            runs, status = detector.model_runs, "ok"
        except (cv2.error, OSError, ValueError):
            status = "failed"
    return {"objects": objects, "objectStatus": status, "objectModel": "vision.objects",
            "objectModelRuns": runs, "objectMs": round((perf_counter() - started) * 1000)}
