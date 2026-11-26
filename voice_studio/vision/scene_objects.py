"""Decode COCO NanoDet-Plus predictions using the existing OpenCV DNN runtime."""

from voice_studio.mod_runtime import model_resource, is_replacement
from voice_studio.vision.object_tracking_contract import TRACKING_PROPOSAL_SCORE

from copy import deepcopy

import cv2
import numpy as np

SIDE = 416
PROPOSAL_SCORE = .32
STRONG_SCORE = .5
LABELS = (
    "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|"
    "stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|"
    "backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|"
    "baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|"
    "banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|"
    "potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|"
    "oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush"
).split("|")
MEAN = (103.53, 116.28, 123.675)
STD = (57.375, 57.12, 58.395)
# Keep the original double-precision normalization and float32 rounding for every channel value.
CHANNEL_VALUES = np.arange(256, dtype=np.float32).reshape(1, 256, 1)
PIXEL_LUT = np.asarray((CHANNEL_VALUES - MEAN) / STD, dtype=np.float32)


class ObjectDetector:
    """Detect prominent objects from a 416px letterboxed view of a frame."""

    def __init__(self, weights, *, reuse_views=False):
        """Load fixed weights and optionally retain one exact previous model view."""
        weights = model_resource('vision.objects', 'builtin.nanodet', weights)
        self.reuse_views = reuse_views
        self.previous_view = None
        self.cache_hits = 0
        self.model_runs = 0
        self.net = cv2.dnn.readNet(str(weights))
        self.outputs = self.net.getUnconnectedOutLayersNames()

    def clear_view_cache(self):
        """Drop input pixels and proposals at a shot boundary while keeping run counters."""
        self.previous_view = None

    def detect(self, image, *, sectors=None, minimum_score=None):
        """Find objects in the full view and optional overlapping sector crops."""
        height, width = image.shape[:2]
        detector_options = {} if minimum_score is None else {"minimum_score": minimum_score}
        candidates = self.detect_view(image, **detector_options)
        if sectors is None:
            sectors = height >= width * 1.25
        if sectors:
            for bounds in object_sectors(width, height):
                candidates.extend(self.detect_region(image, bounds, **detector_options))
        return unique_proposals(candidates)

    def detect_with_tracking(self, image, *, sectors=None):
        """Return unchanged standard proposals and separate low-score tracking candidates."""
        proposals = self.detect(image, sectors=sectors, minimum_score=TRACKING_PROPOSAL_SCORE)
        ordinary = [item for item in proposals if not item.get("trackingProposalOnly")]
        weak = [item for item in proposals if item.get("trackingProposalOnly")]
        return ordinary, weak

    def detect_region(self, image, bounds, *, minimum_score=None):
        """Project detections from one crop back to source coordinates, excluding clipped edges."""
        height, width = image.shape[:2]
        left, top, right, bottom = bounds
        span_x, span_y = right - left, bottom - top
        candidates = []
        options = {} if minimum_score is None else {"minimum_score": minimum_score}
        for item in self.detect_view(image[top:bottom, left:right], **options):
            x, y, w, h = item["box"]
            if ((left and x <= .01) or (right < width and x + w >= .99)
                    or (top and y <= .01) or (bottom < height and y + h >= .99)):
                continue
            box = [(left + x * span_x) / width, (top + y * span_y) / height,
                   w * span_x / width, h * span_y / height]
            candidates.append({**item, "box": box})
        return candidates

    def detect_view(self, image, *, people=False, minimum_score=None):
        """Return person or object proposals in the view's normalized coordinates."""
        height, width = image.shape[:2]
        scale = SIDE / max(width, height)
        scaled_w, scaled_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (scaled_w, scaled_h))
        options = {} if minimum_score is None else {"minimum_score": minimum_score}
        if not self.reuse_views:
            return self.detect_resized_view(resized, people, **options)
        previous = self.previous_view
        if previous is not None:
            previous_image, previous_people, previous_options, previous_objects = previous
            # A zero maximum difference means every resized byte matches.
            if (people == previous_people and options == previous_options and resized.shape == previous_image.shape
                    and cv2.norm(resized, previous_image, cv2.NORM_INF) == 0):
                self.cache_hits += 1
                return deepcopy(previous_objects)
        objects = self.detect_resized_view(resized, people, **options)
        # Sector projection mutates returned boxes, so retain independent coordinates.
        self.previous_view = resized, people, options, deepcopy(objects)
        return objects

    def detect_resized_view(self, resized, people, *, minimum_score=PROPOSAL_SCORE):
        """Letterbox a resized BGR image and decode one inference in its normalized coordinates."""
        scaled_h, scaled_w = resized.shape[:2]
        left, top = (SIDE - scaled_w) // 2, (SIDE - scaled_h) // 2
        padded = np.full((SIDE, SIDE, 3), 114, dtype=np.uint8)
        padded[top:top + scaled_h, left:left + scaled_w] = resized
        pixels = cv2.LUT(padded, PIXEL_LUT)
        self.net.setInput(cv2.dnn.blobFromImage(pixels))
        self.model_runs += 1
        boxes, scores, classes = decode_boxes(self.net.forward(self.outputs), people=people, minimum_score=minimum_score)
        picked = cv2.dnn.NMSBoxesBatched(boxes, scores, classes, minimum_score, 0.45)
        objects = []
        for index in np.asarray(picked).reshape(-1)[:20]:
            x, y, w, h = boxes[index]
            x1, x2 = np.clip([(x - left) / scaled_w, (x + w - left) / scaled_w], 0, 1)
            y1, y2 = np.clip([(y - top) / scaled_h, (y + h - top) / scaled_h], 0, 1)
            if (x2 - x1) * (y2 - y1) >= 0.002:
                box = [round(float(value), 4) for value in (x1, y1, x2 - x1, y2 - y1)]
                item = {"label": LABELS[classes[index]], "box": box, "score": round(scores[index], 3)}
                if scores[index] < PROPOSAL_SCORE:
                    item["trackingProposalOnly"] = True
                objects.append(item)
        return objects


def object_sectors(width, height):
    """Overlap the long axis or four quadrants to retain detail without cutting objects."""
    span_x, span_y = round(width * .64), round(height * .64)
    if height >= width * 1.25:
        return [(0, 0, width, span_y), (0, height - span_y, width, height)]
    if width >= height * 1.25:
        return [(0, 0, span_x, height), (width - span_x, 0, width, height)]
    return [(left, top, left + span_x, top + span_y)
            for top in (0, height - span_y) for left in (0, width - span_x)]


def iou(left, right):
    """Compare two normalized boxes on the same full-frame coordinate system."""
    x, y, w, h = left
    u, v, p, q = right
    area = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    return area / max(1e-9, w * h + p * q - area)


def unique_proposals(candidates):
    """Keep the existing score-ordered same-class deduplication for an ordinary frame."""
    kept = []
    for item in sorted(candidates, key=lambda hit: (not hit.get("trackingProposalOnly"), hit["score"]), reverse=True):
        if not any(item["label"] == existing["label"] and iou(item["box"], existing["box"]) >= .4
                   for existing in kept):
            kept.append(item)
    return kept[:20]


def containing_fraction(outer, inner):
    """Return the share of the smaller measured box covered by the first."""
    overlap = iou(outer, inner)
    inner_area = inner[2] * inner[3]
    return overlap * (outer[2] * outer[3] + inner_area) / ((1 + overlap) * inner_area)


def calibrate_tiles(detector, plan, folder):
    """Enable tiles for shots where one anchor shows a new confident small object."""
    enabled = set()
    for index, sample in enumerate(plan):
        if not sample["sectors"]:
            continue
        image = cv2.imread(str(folder / f"{index:04}.jpg"))
        if image is None:
            continue
        full = detector.detect(image, sectors=False)
        combined = detector.detect(image, sectors=True)
        for hit in combined:
            box = hit["box"]
            if hit["score"] < STRONG_SCORE or box[2] * box[3] > .025:
                continue
            if not any(item["label"] == hit["label"] and iou(box, item["box"]) >= .35 for item in full):
                enabled.add(sample["shot"])
                break
    return enabled


def drinkware_crop(image, box, expansion):
    """Build one bounded crop whose edges are stable under small detector jitter."""
    x, y, w, h = box
    height, width = image.shape[:2]
    before_x, before_y, after_x, after_y = expansion
    left = round(round(max(0, x - before_x * w), 2) * width)
    top = round(round(max(0, y - before_y * h), 2) * height)
    right = round(round(min(1, x + after_x * w), 2) * width)
    bottom = round(round(min(1, y + after_y * h), 2) * height)
    return left, top, right, bottom


def scan_drinkware(detector, image, bounds):
    """Return exact source boxes for confident cups, or veto if the laptop is confirmed."""
    left, top, right, bottom = bounds
    span_x, span_y = right - left, bottom - top
    if min(span_x, span_y) < 48:
        return []
    height, width = image.shape[:2]
    proposals = detector.detect_view(image[top:bottom, left:right])
    if any(hit["label"] == "laptop" and hit["score"] >= .45 for hit in proposals):
        return None
    cups = []
    for hit in sorted(proposals, key=lambda row: -row["score"]):
        if hit["label"] not in {"cup", "wine glass"} or hit["score"] < STRONG_SCORE:
            continue
        x, y, w, h = hit["box"]
        if min(x, y, 1 - x - w, 1 - y - h) < .015:
            continue
        box = [(left + x * span_x) / width, (top + y * span_y) / height,
               w * span_x / width, h * span_y / height]
        if not any(iou(box, other["box"]) >= .25 for other in cups):
            cups.append({**hit, "box": box, "detailBasis": "local-drinkware"})
    return cups


def refine_drinkware(detector, image, objects):
    """Retry one weak laptop footprint before replacing it with distinct, evidenced cups."""
    candidates = [obj for obj in objects if obj["label"] == "laptop" and obj["score"] < .45]
    if not candidates:
        return objects, None
    outer = min(candidates, key=lambda obj: obj["box"][2] * obj["box"][3])
    for expansion in ((.74, .52, 2.10, 1.69), (.90, .85, 2.37, 1.95)):
        cups = scan_drinkware(detector, image, drinkware_crop(image, outer["box"], expansion))
        if cups is None:
            return objects, None
        if len(cups) >= 2 and any(containing_fraction(outer["box"], cup["box"]) >= .8 for cup in cups):
            break
    else:
        return objects, None
    kept, removed = [], []
    for obj in objects:
        if obj in candidates and (obj is outer or iou(obj["box"], outer["box"]) >= .35):
            removed.append(obj)
        else:
            kept.append(obj)
    for cup in cups:
        if not any(obj["label"] in {"cup", "wine glass"} and iou(cup["box"], obj["box"]) >= .55 for obj in kept):
            kept.append(cup)
    kept.sort(key=lambda row: -row["score"])
    return kept, {"proposal": outer, "suppressed": removed, "basis": "local-drinkware", "verifiedCups": len(cups)}


def confirm_objects(frames):
    """Keep strong boxes and promote lower scores only with a nearby matching observation."""
    sampled = [frame for frame in frames if frame["objects"]]
    raw = [frame["objects"] for frame in sampled]
    for index, frame in enumerate(sampled):
        nearby = []
        for other in (index - 1, index + 1):
            if not 0 <= other < len(sampled):
                continue
            adjacent = sampled[other]
            if frame.get("shot") == adjacent.get("shot") and abs(frame["timeMs"] - adjacent["timeMs"]) <= 1400:
                nearby.extend(raw[other])
        confirmed = []
        for item in raw[index]:
            strong = item["score"] >= STRONG_SCORE
            repeated = any(item["label"] == other["label"] and iou(item["box"], other["box"]) >= .30
                           for other in nearby)
            if strong or repeated:
                box = [round(value, 4) for value in item["box"]]
                confirmed.append({**item, "box": box, "basis": "score" if strong else "repeated"})
        frame["objects"] = confirmed[:12]


def decode_boxes(outputs, *, people=False, minimum_score=PROPOSAL_SCORE):
    """Pair output heads by grid size so inference works across OpenCV output orderings."""
    heads = {(out.shape[1], out.shape[2]): out[0] for out in outputs}
    boxes, scores, classes = [], [], []
    for stride in (8, 16, 32):
        side = SIDE // stride
        count = side ** 2
        probabilities = heads[count, len(LABELS)]
        class_ids = probabilities.argmax(axis=1)
        confidence = probabilities[np.arange(count), class_ids]
        anchors = np.flatnonzero(((class_ids == 0) if people else (class_ids != 0)) & (confidence >= minimum_score))
        logits = heads[count, 32][anchors].reshape(-1, 4, 8)
        exp = np.exp(logits - logits.max(axis=2, keepdims=True))
        offsets = (exp / exp.sum(axis=2, keepdims=True)) @ np.arange(8) * stride
        for anchor, (left, top, right, bottom) in zip(anchors, offsets):
            x = (anchor % side) * stride + (stride - 1) / 2
            y = (anchor // side) * stride + (stride - 1) / 2
            boxes.append([float(x - left), float(y - top), float(left + right), float(top + bottom)])
            scores.append(float(confidence[anchor]))
            classes.append(int(class_ids[anchor]))
    return boxes, scores, classes
