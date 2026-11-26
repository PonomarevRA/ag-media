"""Confirm a short, isolated speaking-face change from original frames."""

from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
import subprocess
import sys

from voice_paths import temporary_directory
from voice_studio.vision.video_activity_chunks import STEP, iou

MAX_PAIRS = 3
MIN_SIMILARITY = .68


def head_samples(cue, activity, timeline):
    """Take two separated frame boxes from the exact measured speaking track."""
    first = (cue["startMs"] + STEP - 1) // STEP
    last = (cue["endMs"] + STEP - 1) // STEP
    center = activity["sample"]["time_ms"]
    samples = []
    for index, hits in enumerate(timeline[first:last], first):
        for hit in hits:
            if hit[0] == activity["track"] and hit[6] == activity["face"]:
                samples.append({"timeMs": index * STEP, "box": hit[1:5]})
                break
    samples.sort(key=lambda row: abs(row["timeMs"] - center))
    if not samples:
        return []
    second = next((row for row in samples[1:] if abs(row["timeMs"] - samples[0]["timeMs"]) >= 120), None)
    return [samples[0], second] if second else []


def boundary_pairs(cues, observations, visual):
    """Bound checks to one-off face IDs between adjacent cues of the same acoustic cluster."""
    from voice_studio.vision.face_speakers import co_visible_pairs

    confirmed = Counter(row["face"] for row in observations.values() if row.get("state") == "visible-active")
    distinct = {tuple(sorted(pair)) for pair in visual.get("faceTracking", {}).get("distinctFaces", ())}
    cuts = visual.get("faceTracking", {}).get("identityCutTimesMs", ())
    timeline = (visual.get("timeline") or {}).get("frames", ())
    pairs = []
    seen = set()
    for left, right in pairwise(cues):
        if (left.get("speaker") is None or left["speaker"] != right.get("speaker")
                or abs(right["startMs"] - left["endMs"]) > 240
                or any(cue.get("overlap") or cue.get("overlaps") or cue.get("speakerNumber")
                       for cue in (left, right))
                or any(left["startMs"] < cut < right["endMs"] for cut in cuts)):
            continue
        links = [observations.get(str(cue["startMs"]), {}) for cue in (left, right)]
        if any(row.get("state") != "visible-active" or not row.get("sample") for row in links):
            continue
        faces = tuple(sorted(row["face"] for row in links))
        if faces[0] == faces[1] or faces in distinct:
            continue
        weak, strong = sorted(range(2), key=lambda index: confirmed[links[index]["face"]])
        first, second = (left, right)[weak], (left, right)[strong]
        if (confirmed[links[weak]["face"]] != 1 or confirmed[links[strong]["face"]] < 3
                or first["endMs"] - first["startMs"] > 2800 or first["startMs"] in seen):
            continue
        samples = [head_samples(cue, links[index], timeline) for index, cue in enumerate((left, right))]
        if not all(samples):
            continue
        if faces in co_visible_pairs({"frames": timeline[left["startMs"] // STEP:(right["endMs"] + STEP - 1) // STEP]},
                                     set(faces)):
            continue
        pairs.append({"cueMs": first["startMs"], "face": links[strong]["face"], "samples": samples})
        seen.add(first["startMs"])
        if len(pairs) == MAX_PAIRS:
            break
    return pairs


def run_native_review(source, formats, pairs):
    """Verify a bounded set of candidate pairs with the existing original-crop worker."""
    from voice_studio.vision.video_revolver import MODEL, SFACE, VISION, run_worker

    if not pairs:
        return []
    try:
        with temporary_directory() as folder:
            request = Path(folder) / "speaker-face-pairs.json"
            request.write_text(json.dumps({"source": str(source), "formats": formats,
                                           "yunet": str(MODEL), "sface": str(SFACE), "pairs": pairs}), encoding="utf-8")
            return run_worker("voice_studio.vision.speaker_face_recheck", [request], [VISION], timeout=25)["matches"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return []


def measure_pairs(request):
    """Recompute private SFace measurements on source crops, exposing only agreement scores."""
    import cv2
    from voice_studio.mod_runtime import model_resource
    from voice_studio.speech.speaker_photos import frame_at
    from voice_studio.vision.face_reid import Signatures, cosine
    from voice_studio.vision.scene_ocr import pixel_view

    cv2.setNumThreads(1)
    yunet = model_resource("vision.faces", "builtin.yunet", Path(request["yunet"]))
    detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), .79, .3, 50)
    signatures = Signatures(Path(request["sface"]))

    def feature(sample):
        """Find and align the head overlapping the track box in a bounded native crop."""
        frame = frame_at(request["source"], request["formats"], sample["timeMs"], native=True)
        if frame is None:
            return ()
        x, y, w, h = sample["box"]
        left, top = max(0, x - w * .28), max(0, y - h * .28)
        right, bottom = min(1, x + 1.28 * w), min(1, y + 1.28 * h)
        crop, bounds = pixel_view(frame, (left, top, right - left, bottom - top))
        height, width = crop.shape[:2]
        if min(height, width) < 32:
            return ()
        detector.setInputSize((width, height))
        _, rows = detector.detect(crop)
        if rows is None:
            return ()
        a, b, cw, ch = bounds
        target = [(x - a) / cw * width, (y - b) / ch * height, w / cw * width, h / ch * height]
        options = [row for row in rows if iou(target, row[:4]) >= .5]
        return signatures.feature(crop, options[0]) if len(options) == 1 else ()

    output = []
    for pair in request["pairs"]:
        try:
            left, right = [[feature(sample) for sample in side] for side in pair["samples"]]
        except (OSError, ValueError, subprocess.SubprocessError, cv2.error):
            continue
        if not all((*left, *right)):
            continue
        score = min(cosine(a, b) for a in left for b in right)
        if score >= MIN_SIMILARITY:
            output.append({"cueMs": pair["cueMs"], "face": pair["face"], "similarity": round(score, 3)})
    return {"matches": output}


if __name__ == "__main__":
    print(json.dumps(measure_pairs(json.loads(Path(sys.argv[1]).read_bytes()))))
