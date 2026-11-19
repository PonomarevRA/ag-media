"""Sample visible faces from local video using YuNet in the vision runtime."""

from voice_studio.mod_runtime import model_resource, is_replacement

from base64 import b64encode
import json
import subprocess
import sys

import cv2
import numpy as np


def frame_at(source, formats, time_ms, *, native=False):
    """Decode a frame at the trim point as a proxy JPEG or a full-resolution PNG."""
    encoding = ["-c:v", "png"] if native else [
        "-vf", "scale=640:640:force_original_aspect_ratio=decrease", "-c:v", "mjpeg"]
    encoded = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-protocol_whitelist", "file,pipe",
         "-format_whitelist", formats, "-ss", f"{time_ms / 1000:.3f}", "-i", source,
         "-map", "0:V:0" if native else "0:v:0", "-an", "-frames:v", "1",
         *encoding, "-f", "image2pipe", "pipe:1"],
        stderr=subprocess.DEVNULL, timeout=4,
    )
    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR) if encoded else None


def thumbnail(frame, box):
    """Encode a small square crop, including a margin around the detected face."""
    x, y, width, height = box[:4]
    radius = max(width, height) * .63
    cx, cy = x + width / 2, y + height / 2
    matrix = np.array([[2 * radius / 128, 0, cx - radius],
                       [0, 2 * radius / 128, cy - radius]], dtype=np.float32)
    crop = cv2.warpAffine(frame, matrix, (128, 128), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    ok, image = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return b64encode(image).decode("ascii") if ok else None


def collect(source, model, formats, phrases):
    """Return visible photo candidates by phrase start; their identities are unknown."""
    cv2.setNumThreads(1)
    model = str(model_resource('vision.faces', 'builtin.yunet', model))
    detector = cv2.FaceDetectorYN.create(model, "", (320, 320), .78, .30, 500)
    result = {}
    for start, end in phrases:
        key = str(start)
        at = (start + end) // 2
        evidence = {}
        try:
            frame = frame_at(source, formats, at)
            if frame is not None:
                detector.setInputSize((frame.shape[1], frame.shape[0]))
                _, boxes = detector.detect(frame)
                if boxes is not None:
                    faces = [thumbnail(frame, box) for box in sorted(boxes, key=lambda box: box[0])[:3]]
                    evidence.update(photoTimeMs=at, photos=[photo for photo in faces if photo])
        except subprocess.SubprocessError:
            pass
        if evidence:
            result[key] = evidence
    return result


if __name__ == "__main__":
    source, model, formats, phrases = sys.argv[1:]
    print(json.dumps(collect(source, model, formats, json.loads(phrases))))
