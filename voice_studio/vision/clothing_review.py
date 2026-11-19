"""Sample a small set of outfit references between the face and audio stages."""

from voice_studio.mod_runtime import model_resource, worker_environment

from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

from voice_studio.vision.clothing_match import borderline_pairs, choose_sample_tracks, corroborate
from voice_studio.vision.face_appearance import describe_appearance

from voice_paths import PROJECT_ROOT as ROOT, dependency_path, model_path, temporary_directory

MP_PYTHON = ROOT / ".engines/mediapipe/bin/python"
BODY_MODEL = model_path("vision", "nanodet.onnx")
MASK_MODEL = model_path("mediapipe", "selfie_multiclass_256x256.tflite")
APPEARANCE_MODEL = model_path("vision", "openvino-age-gender", "model.xml")
APPEARANCE_RUNTIME = dependency_path("face-attributes")


def sample_requests(tracks, selected, width, height, appearance_faces=()):
    """Retain two outfit frames and add up to two separated face-only samples for priority speakers."""
    requests = defaultdict(list)
    for face, index in selected.items():
        track = tracks[index - 1]
        picked = [track[len(track) // 3][0], track[len(track) * 2 // 3][0]]
        for frame in picked:
            requests[frame].append((face, index))
        if face in appearance_faces:
            for offset in (len(track) // 6, len(track) * 5 // 6):
                frame = track[offset][0]
                if all(abs(frame - prior) >= 3 for prior in picked):
                    requests[frame].append((face, index, "appearance"))
                    picked.append(frame)
    visible = defaultdict(list)
    for track_id, track in enumerate(tracks, 1):
        for frame, (x, y, w, h) in track:
            if frame in requests:
                box = [max(0., min(1., float(value / axis))) for value, axis in
                       ((x, width), (y, height), (w, width), (h, height))]
                visible[frame].append([track_id, *box])
    return requests, visible


def save_references(source, folder, requests, visible):
    """Decode the existing video proxy once and write only requested sample frames."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    saved = []
    try:
        for index in range(max(requests) + 1):
            if not capture.grab():
                break
            if index not in requests:
                continue
            ok, frame = capture.retrieve()
            filename = f"{index}.jpg"
            try:
                if ok and cv2.imwrite(str(folder / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    saved.append({"file": filename, "timeMs": index * 40, "targets": requests[index],
                                  "faces": visible[index]})
            except cv2.error:
                continue
    finally:
        capture.release()
    return saved


def collect_samples(source, tracks, selected, width, height, appearance_faces=()):
    """Read a bounded sidecar of descriptors from the MediaPipe interpreter."""
    requests, visible = sample_requests(tracks, selected, width, height, appearance_faces)
    with temporary_directory() as raw:
        folder = Path(raw)
        frames = save_references(source, folder, requests, visible)
        request = folder / "clothing.json"
        request.write_text(json.dumps({"frames": frames, "models": {
            "body": str(BODY_MODEL), "masks": str(MASK_MODEL), "appearance": str(APPEARANCE_MODEL),
        }}), encoding="utf-8")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONPATH": str(APPEARANCE_RUNTIME)}
        response = subprocess.check_output([str(MP_PYTHON), str(ROOT / "clothing_sample_worker.py"), str(request)],
                                           env=worker_environment(env), stderr=subprocess.DEVNULL, timeout=22)
        return json.loads(response)


def review_clothing(source, tracks, signatures, aliases, width, height, active_faces=()):
    """Keep baseline face groups when optional outfit evidence is missing or inconclusive."""
    report = {"status": "unavailable", "proposals": 0, "samples": 0, "matches": []}
    if os.environ.get("VOICE_CLOTHING_PROFILE", "1") == "0":
        return aliases, {}, {**report, "status": "disabled"}
    if not MP_PYTHON.is_file():
        return aliases, {}, report
    body_model = model_resource('vision.objects', 'builtin.nanodet', BODY_MODEL)
    mask_model = model_resource('vision.segmentation', 'builtin.mediapipe-selfie', MASK_MODEL)
    if not body_model.is_file() or not mask_model.is_file():
        return aliases, {}, report
    pairs = borderline_pairs(tracks, signatures, aliases)
    selected = choose_sample_tracks(tracks, aliases, pairs, active_faces)
    report["proposals"] = len(pairs)
    if not selected:
        return aliases, {}, {**report, "status": "skipped"}
    started = perf_counter()
    try:
        # Bound the additional work to the first four activity candidates with selected tracks.
        first_speakers = [face for face in dict.fromkeys(active_faces) if face in selected][:4]
        samples = collect_samples(source, tracks, selected, width, height, first_speakers)
    except (OSError, ValueError, subprocess.SubprocessError):
        return aliases, {}, {**report, "status": "failed"}
    outfit_samples = [sample for sample in samples if "hist" in sample]
    revised, matches, descriptions = corroborate(pairs, outfit_samples, aliases)
    face_aliases = {face: revised[index] for face, index in selected.items()}
    original_samples = [sample for sample in samples if not sample.get("appearanceOnly")]
    appearance = describe_appearance(original_samples, face_aliases)
    extra_appearance = describe_appearance(samples, face_aliases, allow_age_consensus=True)
    for face, traits in extra_appearance.items():
        known = appearance.get(face, {})
        if any(key in traits and key not in known for key in ("age", "appearanceClass", "hairSwatch")):
            appearance[face] = {**traits, **known, "sampleTimesMs": traits["sampleTimesMs"]}
    return revised, descriptions, {**report, "status": "ok" if samples else "unavailable",
                                   "samples": len(samples), "matches": matches,
                                   "appearance": appearance,
                                   "elapsedMs": round((perf_counter() - started) * 1000)}
