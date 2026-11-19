"""Extract small clothing descriptors in the existing MediaPipe CPU environment."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from voice_studio.vision.face_appearance import appearance_model, apparent_age_and_sex
from voice_studio.workers.person_layer_worker import boxes_for_faces, clothing_mask, open_models, pixel_crop


def garment_pattern(image, mask):
    """Describe broad visible texture inside the clothing mask at a fixed crop scale."""
    height, width = image.shape[:2]
    scale = 160 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    pixels = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    interior = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    interior = cv2.erode(interior, np.ones((7, 7), np.uint8)).astype(bool)
    if np.count_nonzero(interior) < 240:
        return None
    # Local color edges reveal patterns while suppressing ordinary shading on a solid fabric.
    smooth = cv2.GaussianBlur(pixels, (5, 5), 0)
    edges = np.max(cv2.absdiff(pixels, smooth), axis=2)
    density = float(np.mean(edges[interior] >= 13))
    if density >= .095:
        return "patterned"
    if density <= .025:
        return "plain"
    return None


def descriptor(image, box, face, segmenter, mp):
    """Describe masked pixels near the upper body and avoid loose crop boundaries."""
    crop, (_, top, _, bottom) = pixel_crop(image, box)
    mask = clothing_mask(segmenter, mp, crop)
    if mask is None:
        return None
    height = image.shape[0]
    face_bottom = (face[2] + face[4]) * height
    body_start = max(0, round(face_bottom - top))
    body_end = round(body_start + (bottom - top - body_start) * .75)
    mask[:body_start] = 0
    mask[body_end:] = 0
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    coordinates = mask.astype(bool)
    if coordinates.sum() < 300:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[coordinates].astype(int)
    luminance, a, b = lab.T
    chroma_a = np.clip((a - 68) // 24, 0, 4)
    chroma_b = np.clip((b - 68) // 24, 0, 4)
    indices = chroma_a + 5 * chroma_b + 25 * (luminance // 64)
    hist = np.bincount(indices, minlength=100) / len(indices)
    red, green, blue = np.median(crop[coordinates][:, ::-1], axis=0).astype(int)
    return {"hist": hist.tolist(), "swatch": f"#{red:02x}{green:02x}{blue:02x}",
            "pattern": garment_pattern(crop, mask)}


def measure_hair(image, hit, neighbors, segmenter, mp):
    """Measure the visible hair swatch only inside a single person's head crop."""
    track, x, y, w, h = hit[:5]
    head = (x - .34 * w, y - .65 * h, 1.68 * w, 1.62 * h)
    left, top, width, height = head
    for other, a, b, c, d, *_ in neighbors:
        if other != track and left <= a + c / 2 <= left + width and top <= b + d / 2 <= top + height:
            return None
    crop, _ = pixel_crop(image, head)
    if min(crop.shape[:2]) < 48:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    output = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    mask = np.ascontiguousarray(output.category_mask.numpy_view() == 1, dtype=np.uint8)
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
    if np.count_nonzero(interior) < max(120, .04 * interior.size):
        return None
    return np.median(rgb[interior], axis=0).astype(int).tolist()


def read_samples(request_path):
    """Reuse model instances and process only selected frames with an unambiguous torso."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    plan = {"body": True, "masks": True, "gaze": False}
    model_path = request["models"].get("appearance")
    appearance = appearance_model(Path(model_path) if model_path else None)
    samples = []
    with ExitStack() as stack:
        try:
            detector, mp, _, segmenter = open_models(stack, plan, request["models"])
        except (ImportError, OSError, RuntimeError, ValueError):
            detector = mp = segmenter = None
        for row in request["frames"]:
            image = cv2.imread(str(request_path.parent / row["file"]))
            if image is None:
                continue
            outfit_targets = any(len(target) == 2 for target in row["targets"])
            boxes = boxes_for_faces(detector.detect_view(image, people=True), row["faces"]) if detector and outfit_targets else {}
            for face, track, *kind in row["targets"]:
                hit = next(item for item in row["faces"] if item[0] == track)
                crop, _ = pixel_crop(image, hit[1:5])
                result = apparent_age_and_sex(appearance, crop)
                hair = measure_hair(image, hit, row["faces"], segmenter, mp) if segmenter else None
                if hair is not None:
                    result["hairRgb"] = hair
                if not kind and track in boxes:
                    result.update(descriptor(image, boxes[track], hit, segmenter, mp) or {})
                if result:
                    samples.append({**result, "face": face, "timeMs": row["timeMs"], "appearanceOnly": bool(kind)})
    return samples


if __name__ == "__main__":
    print(json.dumps(read_samples(Path(sys.argv[1])), allow_nan=False))
