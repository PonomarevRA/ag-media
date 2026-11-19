"""Sample visible people and retry YuNet on source-resolution head crops."""

from voice_studio.mod_runtime import model_resource, is_replacement

from collections import Counter
import subprocess

import cv2
import numpy as np

from voice_studio.speech.speaker_photos import frame_at
from voice_studio.vision.scene_objects import ObjectDetector, containing_fraction, iou

BODY_STEP = 25
RETRY_STEP = 5
MAX_SOURCE_FRAMES = 16
PERSON_SCORE = .45


def head_region(box):
    """Approximate the upper body region in normalized frame coordinates."""
    x, y, width, height = box
    return (max(0, x - width * .10), max(0, y - height * .08),
            min(1, x + width * 1.10), min(1, y + height * .48))


def face_in_region(face, region, size):
    """Match a face center to a person's expected head region."""
    width, height = size
    x, y, w, h = face[:4]
    left, top, right, bottom = region
    return left <= (x + w / 2) / width <= right and top <= (y + h / 2) / height <= bottom


def unmatched_heads(regions, faces, size):
    """Claim a baseline face at most once so overlapping people still get a retry."""
    pairs = []
    for index, region in enumerate(regions):
        left, top, right, bottom = region
        for number, face in enumerate(faces):
            if face_in_region(face, region, size):
                x, y, w, h = face[:4]
                dx = ((x + w / 2) / size[0] - (left + right) / 2) / max(.01, right - left)
                dy = ((y + h / 2) / size[1] - (top + bottom) / 2) / max(.01, bottom - top)
                pairs.append((dx * dx + dy * dy, index, number))
    claimed_regions, claimed_faces = set(), set()
    for _, index, number in sorted(pairs):
        if index not in claimed_regions and number not in claimed_faces:
            claimed_regions.add(index)
            claimed_faces.add(number)
    return [region for index, region in enumerate(regions) if index not in claimed_regions]


def crop_region(frame, region):
    """Slice a normalized region from an image and return its pixel origin."""
    height, width = frame.shape[:2]
    left, top, right, bottom = (round(value * scale)
                                for value, scale in zip(region, (width, height, width, height)))
    return frame[top:bottom, left:right], left, top


def source_face(detector, frame, region, proxy):
    """Confirm local alignment, detect on native pixels, and project onto the proxy."""
    height, width = frame.shape[:2]
    proxy_height, proxy_width = proxy.shape[:2]
    crop, left, top = crop_region(frame, region)
    reference, _, _ = crop_region(proxy, region)
    if min(crop.shape[:2]) < 24 or not reference.size or not similar_pixels(crop, reference):
        return None
    detector.setInputSize((crop.shape[1], crop.shape[0]))
    _, faces = detector.detect(crop)
    if faces is None:
        return None
    # A head candidate from an overlapping body crop is ambiguous if it has several faces.
    if len(faces) != 1:
        return None
    face = faces[0].copy()
    face[0] += left
    face[1] += top
    face[4:14:2] += left
    face[5:14:2] += top
    face[:14:2] *= proxy_width / width
    face[1:14:2] *= proxy_height / height
    return face


def duplicate_face(left, right):
    """Reject overlaps and nested boxes centered on the same small face."""
    if iou(left, right) >= .35:
        return True
    if left[2] * left[3] > right[2] * right[3]:
        left, right = right, left
    if min(left[2:]) <= 0 or containing_fraction(right, left) < .70:
        return False
    x, y, w, h = left
    a, b, u, v = right
    # A tight retry can be centered higher than YuNet's full-frame box.
    return abs(x + w / 2 - a - u / 2) <= w * .35 and abs(y + h / 2 - b - v / 2) <= h * .70


def similar_pixels(original, proxy):
    """Compare downscaled pixels while allowing normal proxy encoding differences."""
    left = cv2.resize(original, (32, 32), interpolation=cv2.INTER_AREA)
    right = cv2.resize(proxy, (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.norm(left, right, cv2.NORM_L1) / left.size <= 12


def same_view(original, proxy):
    """Reject a source sample with a different orientation or visibly changed scene."""
    height, width = original.shape[:2]
    proxy_height, proxy_width = proxy.shape[:2]
    return abs(width / height / (proxy_width / proxy_height) - 1) <= .025 and similar_pixels(original, proxy)


class PersonFacePilot:
    """Keep a one-second person proposal cache and bounded original-frame retries."""

    def __init__(self, source, formats, offset, person_model, yunet, *, count_only=False):
        yunet = model_resource('vision.faces', 'builtin.yunet', yunet)
        self.source = source
        self.formats = formats
        self.offset = offset
        self.samples = []
        self.stats = Counter()
        self.regions = []
        self.next_sample = 0
        self.last_recovery = -RETRY_STEP - 1
        self.last_source_attempt = -BODY_STEP
        try:
            self.people = ObjectDetector(person_model)
            self.faces = None if count_only else cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), .78, .30, 500)
        except cv2.error:
            self.people = None

    def sample_people(self, sample):
        """Measure people once per second and refresh immediately on a cut."""
        self.regions = []
        self.next_sample = sample.number + BODY_STEP
        try:
            self.stats["bodyPasses"] += 1
            hits = self.people.detect_view(sample.color, people=True)
        except cv2.error:
            self.stats["bodyErrors"] += 1
            return None
        hits = [hit for hit in hits if hit["score"] >= PERSON_SCORE]
        self.regions = [head_region(hit["box"]) for hit in hits]
        row = {"timeMs": sample.number * 40, "count": len(hits), "people": hits}
        self.samples.append(row)
        return row

    def rescue_faces(self, sample, regions, faces, proxy_size):
        """Add source-resolution faces only if the retrieved frame matches the proxy."""
        self.stats["sourceAttempts"] += 1
        try:
            original = frame_at(str(self.source), self.formats, self.offset + sample.number * 40, native=True)
            if original is None or not same_view(original, sample.color):
                self.stats["sourceRejected"] += 1
                return
            for region in regions[:3]:
                candidate = source_face(self.faces, original, region, sample.color)
                if candidate is None or not face_in_region(candidate, region, proxy_size):
                    continue
                if not any(duplicate_face(candidate[:4], other[:4]) for other in faces):
                    faces.append(candidate)
                    self.stats["recoveredFaces"] += 1
                    self.last_recovery = sample.number
        except (subprocess.SubprocessError, OSError, cv2.error):
            self.stats["sourceErrors"] += 1

    def augment(self, sample, detections, measured):
        """Preserve baseline measurements and add confirmed crops before tracking."""
        if self.people is None:
            return detections
        row = None
        if sample.cut or sample.suspect_cut or sample.number >= self.next_sample:
            row = self.sample_people(sample)
        if sample.cut or sample.suspect_cut:
            self.last_recovery = -RETRY_STEP - 1
        # Retry every five frames while a crop is productive, otherwise once a second.
        retry_stride = RETRY_STEP if sample.number - self.last_recovery <= RETRY_STEP else BODY_STEP
        eligible = sample.number - self.last_source_attempt >= retry_stride
        if (self.faces is None or not measured or not eligible or not self.regions
                or self.stats["sourceAttempts"] >= MAX_SOURCE_FRAMES):
            return detections
        faces = list(detections) if detections is not None else []
        proxy_size = sample.color.shape[1], sample.color.shape[0]
        missing = unmatched_heads(self.regions, faces, proxy_size)
        if not missing:
            return detections
        before = len(faces)
        self.last_source_attempt = sample.number
        self.rescue_faces(sample, missing, faces, proxy_size)
        if row is not None:
            row["recoveredFaces"] = len(faces) - before
        return faces if len(faces) > before else detections

    def data(self):
        """Expose sampled counts and retry diagnostics without implying continuous coverage."""
        return {"status": "ok" if self.samples else "unavailable", "detector": "NanoDet",
                "samples": self.samples, **self.stats}
