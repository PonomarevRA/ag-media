"""Add detail from production anchor crops without replacing the baseline observations."""

from collections import defaultdict
from time import perf_counter

import cv2

from voice_studio.vision.object_appearance import ObjectAppearance
from voice_studio.vision.scene_objects import STRONG_SCORE, containing_fraction, iou, object_sectors, unique_proposals
from voice_studio.vision.scene_sampling import SMALL_OBJECT_DETAIL

MAX_SMALL_AREA = .025
MAX_ADDITIONS_PER_FRAME = 4


def small_enough(item):
    """Apply the validated threshold in full-frame area, independent of crop size."""
    width, height = item["box"][2:]
    return 0 < width * height <= MAX_SMALL_AREA


def already_detected(item, existing):
    """Avoid a second same-label box for one object while keeping adjacent objects separate."""
    box = item["box"]
    for candidate in existing:
        if candidate["label"] != item["label"]:
            continue
        other = candidate["box"]
        if iou(box, other) >= .32 or max(containing_fraction(box, other), containing_fraction(other, box)) >= .85:
            return True
    return False


class SmallObjectPass:
    """Learn crop targets at the ordinary anchor and revisit only selected regions in that shot."""

    def __init__(self):
        self.targets = {}
        self.samples = []

    def observe_anchor(self, detector, image, shot):
        """Reuse the ordinary full/sector inference to identify confident, newly visible small objects."""
        height, width = image.shape[:2]
        full = detector.detect_view(image)
        candidates = list(full)
        regions = []
        for index, bounds in enumerate(object_sectors(width, height)):
            proposals = detector.detect_region(image, bounds)
            candidates.extend(proposals)
            regions.append((index, proposals))
        baseline = unique_proposals(candidates)
        # Each selected region must contribute a proposal that survived the ordinary anchor NMS.
        retained = {id(item) for item in baseline}
        targets = defaultdict(set)
        for index, proposals in regions:
            for item in proposals:
                if (id(item) in retained and item["score"] >= STRONG_SCORE and small_enough(item)
                        and not already_detected(item, full)):
                    targets[index].add(item["label"])
        if targets:
            self.targets[shot] = dict(targets)
        return baseline

    def add_sample(self, path, frame):
        """Retain a reference to a measured baseline frame, without keeping decoded pixels."""
        self.samples.append((path, frame))

    def run(self, detector):
        """Revisit the baseline clock in activated shots, appending only confident new detail."""
        started = perf_counter()
        runs = detector.model_runs
        checked = added = 0
        status = "ok"
        appearance = ObjectAppearance()
        active_shot = None
        for path, frame in self.samples:
            shot = frame["shot"]
            targets = self.targets.get(shot)
            if not targets:
                continue
            if shot != active_shot:
                detector.clear_view_cache()
                active_shot = shot
            try:
                image = cv2.imread(str(path))
                if image is None:
                    status = "partial"
                    continue
                detail = self._new_objects(detector, image, frame["objects"], targets)
                appearance.add(image, {"timeMs": frame["timeMs"], "shot": shot, "objects": detail})
                frame["objects"].extend(detail)
                checked += 1
                added += len(detail)
            except (cv2.error, KeyError, ValueError):
                status = "partial"
        return {"status": status, "shots": sorted(self.targets), "frames": checked, "added": added,
                "modelRuns": detector.model_runs - runs, "elapsedMs": round((perf_counter() - started) * 1000)}

    def _new_objects(self, detector, image, baseline, targets):
        """Search the witness regions only and retain small proposals for their supported labels."""
        height, width = image.shape[:2]
        candidates = []
        for index, bounds in enumerate(object_sectors(width, height)):
            labels = targets.get(index)
            if not labels:
                continue
            candidates.extend(item for item in detector.detect_region(image, bounds)
                              if item["label"] in labels and item["score"] >= STRONG_SCORE and small_enough(item))
        additions = []
        for item in unique_proposals(candidates):
            if already_detected(item, baseline) or already_detected(item, additions):
                continue
            # Keep a conflicting crop label from duplicating an almost identical baseline box.
            if any(iou(item["box"], candidate["box"]) >= .72 for candidate in baseline):
                continue
            box = [round(value, 4) for value in item["box"]]
            additions.append({**item, "box": box, "basis": "score", "detailBasis": SMALL_OBJECT_DETAIL})
            if len(additions) == MAX_ADDITIONS_PER_FRAME:
                break
        return additions
