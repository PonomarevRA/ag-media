"""Run body detection, head and gaze estimation, and clothing masks on decoded crops."""

from voice_studio.core.process_usage import process_usage
from voice_studio.mod_runtime import model_resource, is_replacement

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from voice_studio.vision.head_pose import head_pose
from voice_studio.vision.scene_objects import ObjectDetector

cv2.setNumThreads(1)

BODY_PREVIEW_INTERVAL_MS = 500
BODY_MASK_INTERVAL_MS = 145
FACE_INTERVAL_MS = 500
CLOSE_FACE_AREA = .08


def matches_face(box, face):
    """A person's detected rectangle must contain the face in its upper portion."""
    x, y, w, h = box
    a, b, p, q = face[1:5]
    center_x, center_y = a + p / 2, b + q / 2
    overlap = max(0, min(x + w, a + p) - max(x, a)) * max(0, min(y + h, b + q) - max(y, b))
    return (x <= center_x <= x + w and y <= center_y <= y + h * .62
            and w >= p * 1.15 and h >= q * 1.55 and overlap >= p * q * .60)


def boxes_for_faces(detections, faces):
    """Keep only one-to-one assignments between person detections and face tracks."""
    matches = {}
    for detection in detections:
        box = detection["box"]
        possible = [face for face in faces if matches_face(box, face)]
        if len(possible) == 1 and detection["score"] >= .36:
            face = possible[0]
            matches.setdefault(face[0], box)
    return matches


def shifted_box(box, before, after):
    """Translate a recently verified body rectangle with its short-lived face track."""
    x, y, w, h = box
    a, b, p, q = before[1:5]
    u, v, r, s = after[1:5]
    if min(p, q, r, s) <= 0 or not .8 <= r / p <= 1.25 or not .8 <= s / q <= 1.25:
        return None
    dx, dy = u + (r - p) / 2 - a, v + (s - q) / 2 - b
    if abs(dx) > .08 or abs(dy) > .08:
        return None
    left, top = max(0, x + dx), max(0, y + dy)
    return [left, top, min(w, 1 - left), min(h, 1 - top)]


def pixel_crop(image, box, expand=0):
    """Crop in the same oriented coordinate space as the original video."""
    height, width = image.shape[:2]
    x, y, w, h = box
    left = max(0, int((x - w * expand) * width))
    top = max(0, int((y - h * expand) * height))
    right = min(width, round((x + w * (1 + expand)) * width))
    bottom = min(height, round((y + h * (1 + expand)) * height))
    return image[top:bottom, left:right], (left, top, right, bottom)


def eye_fraction(points, indices, shape):
    """Measure iris position along the line between open eyelid corners."""
    left, right, upper, lower, iris = indices
    height, width = shape[:2]

    def location(index):
        """Convert normalized landmarks into crop pixels before projecting."""
        return np.array((points[index].x * width, points[index].y * height))

    start, end = location(left), location(right)
    axis = end - start
    size = float(np.linalg.norm(axis))
    if size < 9 or np.linalg.norm(location(upper) - location(lower)) / size < .12:
        return None
    offset = float((location(iris) - start) @ axis) / (size * size)
    return offset if .17 <= offset <= .83 else None


def measure_gaze(points, shape):
    """Return a coarse screen-relative horizontal direction from two visible irises."""
    if len(points) < 478:
        return None
    left = eye_fraction(points, (33, 133, 159, 145, 468), shape)
    right = eye_fraction(points, (362, 263, 386, 374, 473), shape)
    if left is None or right is None or abs(left - right) > .18:
        return None
    average = (left + right) / 2
    direction = "left" if average < .40 else "right" if average > .60 else "center"
    return {"horizontal": direction, "irisFraction": round(average, 3)}


def face_observation(model, mp, image, face):
    """Read head rotation and optional iris direction from one tracked-face inference."""
    crop, (left, top, right, bottom) = pixel_crop(image, face[1:5], .32)
    if min(crop.shape[:2]) < 58:
        return {}
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    output = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not output.face_landmarks or len(output.face_landmarks[0]) < 2:
        return {}

    points = output.face_landmarks[0]
    nose = points[1]
    image_h, image_w = image.shape[:2]
    x, y, w, h = face[1:5]
    nose_x = (left + nose.x * (right - left)) / image_w
    nose_y = (top + nose.y * (bottom - top)) / image_h
    # The expanded crop can expose a neighboring face; keep only the requested track.
    if not (x <= nose_x <= x + w and y <= nose_y <= y + h):
        return {}

    observation = {}
    if output.facial_transformation_matrixes:
        pose = head_pose(output.facial_transformation_matrixes[0])
        if pose:
            observation["headPose"] = pose
    gaze = measure_gaze(points, crop.shape)
    if gaze:
        observation["gaze"] = gaze
    return observation


def clothing_mask(segmenter, mp, crop):
    """Return pixels classified as clothing when the mask covers a useful area."""
    if min(crop.shape[:2]) < 35:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    mask = np.ascontiguousarray(result.category_mask.numpy_view() == 4, dtype=np.uint8)
    return mask if .015 <= float(mask.mean()) <= .84 else None


def mask_for_person(segmenter, mp, image, box, neighbors, face, output):
    """Segment clothing in one qualified crop and reject crop overlap with another face."""
    if any(other[0] != face[0] and matches_face(box, other) for other in neighbors):
        return None
    crop, (left, top, right, bottom) = pixel_crop(image, box, .015)
    mask = clothing_mask(segmenter, mp, crop)
    if mask is None:
        return None
    mask *= 255
    if not cv2.imwrite(str(output), mask):
        return None
    height, width = image.shape[:2]
    return {"mask": output.name, "roi": [round(n, 6) for n in
            (left / width, top / height, (right - left) / width, (bottom - top) / height)],
            "maskWidth": int(mask.shape[1]), "maskHeight": int(mask.shape[0])}


def face_tracks(timeline, time_ms):
    """Read the current face tracks without bridging cuts or unobserved gaps."""
    index = time_ms // timeline["stepMs"]
    return timeline["frames"][index] if 0 <= index < len(timeline["frames"]) else []


def open_models(stack, plan, paths):
    """Load the selected workers and enable the existing face model’s pose matrix."""
    paths = dict(paths)
    if plan['gaze']:
        paths['gaze'] = str(model_resource('vision.gaze', 'builtin.mediapipe-face', paths['gaze']))
    if plan['masks']:
        paths['masks'] = str(model_resource('vision.segmentation', 'builtin.mediapipe-selfie', paths['masks']))
    mp = segmenter = eyes = None
    if plan["gaze"] or plan["masks"]:
        import mediapipe as mp

    if mp:
        cpu = mp.tasks.BaseOptions.Delegate.CPU
    if plan["gaze"]:
        try:
            eyes = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=paths["gaze"], delegate=cpu),
                    output_facial_transformation_matrixes=True)))
        except (OSError, ValueError, RuntimeError):
            eyes = None
    if plan["masks"]:
        segmenter = stack.enter_context(mp.tasks.vision.ImageSegmenter.create_from_options(
            mp.tasks.vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=paths["masks"], delegate=cpu),
                                                 output_category_mask=True, output_confidence_masks=False)))
    detector = ObjectDetector(paths["body"]) if plan["body"] else None
    return detector, mp, eyes, segmenter


def body_box(cache, face):
    """Reuse a body rectangle only while the same face still fits its shifted position."""
    before, box = cache.get(face[0], (None, None))
    if box is None:
        return None
    shifted = shifted_box(box, before, face)
    return shifted if shifted and matches_face(shifted, face) else None


def refresh_body(time_ms, refreshed, eligible, cache, masks):
    """Refresh preview geometry after movement or a new track, and keep native mask cadence."""
    interval = BODY_MASK_INTERVAL_MS if masks else BODY_PREVIEW_INTERVAL_MS
    if time_ms - refreshed >= interval or any(face[0] not in cache for face in eligible):
        return True
    return not masks and any(cache[face[0]][1] is not None and not body_box(cache, face) for face in eligible)


def needs_face_sample(face, time_ms, prior_time):
    """Use each crop sample for close-ups and half the rate for smaller faces."""
    close = face[3] * face[4] >= CLOSE_FACE_AREA
    interval = 0 if close else FACE_INTERVAL_MS
    return prior_time is None or time_ms - prior_time >= interval


def analyze(work, request, detector, mp, eyes, segmenter):
    """Associate detections with qualified tracks and emit bounded preview or native mask data."""
    target = request["target"]
    timeline = request["timeline"]
    results = []
    cache = {}
    face_times = {}
    refreshed = -1_000
    sample_width = sample_height = 0
    body_passes = face_samples = 0
    for index, (filename, time_ms) in enumerate(request["frames"]):
        faces = face_tracks(timeline, time_ms)
        eligible = [face for face in faces if target is None or face[6] == target]
        eligible = sorted(eligible, key=lambda hit: -(hit[3] * hit[4]))[:4]
        picked = []
        frame = {"timeMs": time_ms, "index": index, "people": picked}
        results.append(frame)
        if not eligible and target is None:
            cache.clear()
            face_times.clear()
            continue
        image = cv2.imread(str(work / filename))
        if image is None:
            continue
        sample_height, sample_width = image.shape[:2]
        if detector and eligible and refresh_body(time_ms, refreshed, eligible, cache, segmenter is not None):
            detected = boxes_for_faces(detector.detect_view(image, people=True), faces)
            cache = {face[0]: (face, detected.get(face[0])) for face in faces}
            refreshed = time_ms
            body_passes += 1
        current_times = {}
        for face in eligible:
            track = face[0]
            person = {"faceId": face[6], "track": track}
            box = body_box(cache, face)
            if box:
                person["box"] = [round(value, 4) for value in box]
                if segmenter:
                    mask = mask_for_person(segmenter, mp, image, box, faces, face, work / f"mask-{index:06}.png")
                    if mask:
                        frame.update(mask, bodyBox=person["box"], track=track)
            prior = face_times.get(track)
            if eyes and needs_face_sample(face, time_ms, prior):
                person.update(face_observation(eyes, mp, image, face))
                prior = time_ms
                face_samples += 1
            if prior is not None:
                current_times[track] = prior
            if len(person) > 2:
                picked.append(person)
        face_times = current_times
    return {"frames": results, "sampleWidth": sample_width, "sampleHeight": sample_height,
            "bodyPasses": body_passes, "faceSamples": face_samples}


def main(path):
    """Run an isolated task and report native time and memory for the selected models."""
    started = perf_counter()
    request = json.loads(path.read_text(encoding="utf-8"))
    with ExitStack() as stack:
        models = open_models(stack, request["plan"], request["models"])
        result = analyze(path.parent, request, *models)
    cpu_ms, peak_mib = process_usage()
    result.update(plan=request["plan"], workerMs=round((perf_counter() - started) * 1000),
                  gazeUnavailable=request["plan"]["gaze"] and models[2] is None,
                  workerCpuMs=cpu_ms,
                  workerPeakMiB=peak_mib)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
