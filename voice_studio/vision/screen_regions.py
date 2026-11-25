"""Select large, persistent visual regions for bounded original-frame OCR."""

import cv2
import numpy as np

from voice_studio.vision.scene_objects import iou
from voice_studio.vision.scene_ocr import pixel_view
from voice_studio.vision.scene_sampling import evenly_spaced

MAX_DETAIL_STATES = 12
DETAIL_STATES_PER_REGION = 5


def graphic_regions(image):
    """Find large inserted cards and compact diagrams on a light canvas."""
    small = cv2.resize(image, (240, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    light = cv2.inRange(small, (175, 175, 175), (255, 255, 255))
    if (light != 0).mean() >= .65:
        # Local contrast isolates legible diagram details from blurred duplicate artwork.
        contrast = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 1.4))
        ys, xs = np.nonzero(contrast >= 13)
        if len(xs) >= 90:
            x, right = np.quantile(xs, [.025, .975]) / 240
            y, bottom = np.quantile(ys, [.025, .975]) / 160
            if right - x >= .24 and bottom - y >= .20:
                left, top = max(0, x - .10), max(0, y - .09)
                return [("diagram-candidate", [float(left), float(top),
                                               float(min(1, right + .10) - left),
                                               float(min(1, bottom + .09) - top)])]
        return []
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((5, 7), dtype=np.uint8))
    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.55 <= w / 240 <= .99 and .13 <= h / 160 <= .65):
            continue
        if cv2.contourArea(contour) >= .80 * w * h:
            boxes.append([x / 240, y / 160, w / 240, h / 160])
    return [("card-candidate", max(boxes, key=lambda box: box[2] * box[3]))] if boxes else []


def materially_changed(previous, current):
    """Detect a sustained pixel-state change with thresholds tolerant of codec noise."""
    return float((cv2.absdiff(previous, current) >= 16).mean()) >= .018


class ScreenRegionStates:
    """Track at most one large region per category and schedule only distinct stable states."""

    def __init__(self):
        self.active = {}
        self.states = []
        self.shot = None

    def sample(self, image, sample, panel=None):
        """Consume a sampled source frame, using the existing motion-assisted split panel."""
        if sample["shot"] != self.shot:
            self.active.clear()
            self.shot = sample["shot"]
        boxes = [("panel", panel["box"])] if panel else graphic_regions(image)
        visible = {kind for kind, _ in boxes}
        for kind in tuple(self.active):
            if kind not in visible:
                self.active.pop(kind)
        for kind, box in boxes:
            if kind != "panel" and any(iou(box, face) >= .18 for face in sample.get("faces", ())):
                continue
            crop, _ = pixel_view(image, box)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            signature = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)
            self.observe(sample, kind, box, signature, panel)

    def observe(self, sample, kind, box, signature, panel):
        """Confirm geometry twice and commit content once it stabilizes after a change."""
        stamp = sample["timeMs"]
        prior = self.active.get(kind)
        if prior is None or iou(box, prior["box"]) < .66 or stamp - prior["timeMs"] > 2200:
            self.active[kind] = {"box": box, "timeMs": stamp, "signature": signature, "state": None}
            return
        state = prior["state"]
        settled = not materially_changed(prior["signature"], signature)
        # Animated interstitials may never be still at the sampled cadence.
        # Once geometry is corroborated, take a reading and allow a bounded
        # refresh during sustained changes as well as a settled new state.
        changed = False
        if state is not None:
            gap = stamp - state["timeMs"]
            changed = gap >= (400 if settled else 900) and materially_changed(prior["stateSignature"], signature)
        if state is None or changed:
            previous = state
            state = {"sourceFrame": sample["sourceFrame"], "timeMs": stamp, "shot": sample["shot"],
                     "box": box, "regionKind": kind, "lastObservedMs": stamp}
            if panel and panel.get("candidateType"):
                state["panelType"] = panel["candidateType"]
            if previous is not None:
                state["previousStateMs"] = previous["timeMs"]
            self.states.append(state)
            if len(self.states) > MAX_DETAIL_STATES * 4:
                self.states = self.select(MAX_DETAIL_STATES * 2, MAX_DETAIL_STATES)
            prior["state"] = state
            prior["stateSignature"] = signature
        elif state is not None and not materially_changed(prior["stateSignature"], signature):
            state["lastObservedMs"] = stamp
        prior.update(box=box, timeMs=stamp, signature=signature)

    def plan(self):
        """Take a fixed per-analysis budget while covering different visual categories."""
        return self.select(MAX_DETAIL_STATES, DETAIL_STATES_PER_REGION)

    def select(self, maximum, per_category):
        """Distribute state samples through the timeline, including late appearances."""
        by_kind = {}
        for state in self.states:
            by_kind.setdefault(state["regionKind"], []).append(state)
        quotas = dict.fromkeys(by_kind, 0)
        remaining = maximum
        while remaining:
            eligible = [kind for kind, rows in by_kind.items() if quotas[kind] < min(per_category, len(rows))]
            if not eligible:
                break
            for kind in eligible[:remaining]:
                quotas[kind] += 1
                remaining -= 1
        selected = []
        for kind, rows in by_kind.items():
            count = quotas[kind]
            if count == 1:
                selected.append(rows[len(rows) // 2])
            else:
                selected.extend(evenly_spaced(rows, count))
        return sorted(selected, key=lambda row: row["sourceFrame"])
