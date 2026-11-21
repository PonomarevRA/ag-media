"""Use measured change distribution to locate a persistent split-screen panel."""

from bisect import bisect_right

WIDTH = 192
BANDS = 20
MAX_SAMPLE_GAP = 1800


def assign_motion(plan, rows, cuts):
    """Summarize local change pixels between successive OCR frames in each shot."""
    next_row = 0
    for sample in plan:
        if not sample["ocr"]:
            continue
        stamp = sample["timeMs"]
        cut = cuts[max(0, bisect_right(cuts, stamp) - 1)] if cuts else 0
        bands = [0.0] * BANDS
        while next_row < len(rows) and rows[next_row]["startMs"] <= stamp:
            row = rows[next_row]
            next_row += 1
            if row["startMs"] < max(cut, stamp - MAX_SAMPLE_GAP):
                continue
            for region in row["regions"]:
                left, _, width, _ = region["box"]
                if width <= 0:
                    continue
                for band in range(int(left * BANDS), min(BANDS, int((left + width) * BANDS) + 1)):
                    overlap = min((band + 1) / BANDS, left + width) - max(band / BANDS, left)
                    bands[band] += max(0, overlap) * region["changedPixels"] / width
        sample["changeBands"] = [round(value) for value in bands]


def gray_view(image):
    """Use a small raster for seam geometry and change comparison."""
    import cv2

    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (WIDTH, 96), interpolation=cv2.INTER_AREA)


def seams(gray):
    """Examine the strongest vertical divisions through the interior of the frame."""
    import numpy as np

    interior = gray[8:-8].astype(np.int16)
    coverage = (np.abs(np.diff(interior, axis=1)) >= 21).mean(axis=0)
    selected = []
    for index in np.argsort(-coverage[42:149], kind="stable")[:14] + 42:
        column = int(index) + 1
        if coverage[index] < .42 or any(abs(column - prior) < 5 for prior, _ in selected):
            continue
        selected.append((column, float(coverage[index])))
        if len(selected) == 3:
            break
    return selected


def panel_region(column, right, aspect):
    """Preserve source coordinates and distinguish phone-shaped from wider UI."""
    split = column / WIDTH
    x, width = (split, 1 - split) if right else (0, split)
    return {"box": [x, 0, width, 1], "basis": "frame-layout", "changeAssisted": True,
            "candidateType": "mobile-ui" if width * aspect <= .8 else "ui"}


def matching_panel(image, hint):
    """Verify the same seam before sampling a dense scrolling frame."""
    if not hint.get("changeAssisted"):
        return None
    x, _, width, _ = hint["box"]
    expected = (x or width) * WIDTH
    if any(abs(column - expected) <= 4 for column, _ in seams(gray_view(image))):
        return hint
    return None


class SplitPanelProbe:
    """Retain geometry across small UI changes, dropping it at cuts or visual gaps."""

    def __init__(self):
        self.previous = None
        self.panel = None

    def sample(self, image, sample):
        """Use the seam, face exclusion and asymmetric motion to propose a panel for OCR."""
        import cv2
        from voice_studio.vision.scene_objects import iou

        stamp, shot = sample["timeMs"], sample["shot"]
        gray = gray_view(image)
        prior = self.previous
        self.previous = (shot, stamp, gray)
        if prior is None or prior[0] != shot or not 0 < stamp - prior[1] <= MAX_SAMPLE_GAP:
            prior = None
            self.panel = None
        height, width = image.shape[:2]
        if width < height * 1.15:
            self.panel = None
            return None
        delta = cv2.absdiff(prior[2], gray) if prior else None
        bands = sample["changeBands"]
        faces = [a + b / 2 for a, _, b, h in sample["faces"] if b * h >= .005]
        candidates = []
        for column, strength in seams(gray):
            band = round(column * BANDS / WIDTH)
            left_activity = sum(bands[:band]) / band
            right_activity = sum(bands[band:]) / (BANDS - band)
            for right in (False, True):
                panel = panel_region(column, right, width / height)
                x, _, span, _ = panel["box"]
                if span < .21 or span > .68:
                    continue
                if any(x + .02 <= center <= x + span - .02 for center in faces):
                    continue
                inside, outside = (right_activity, left_activity) if right else (left_activity, right_activity)
                content = gray[:, column + 3:] if right else gray[:, :column - 3]
                if distinct_text_edges(content) < .026:
                    continue
                stable = self.panel is not None and iou(panel["box"], self.panel["box"]) >= .92
                if delta is not None:
                    body = delta[9:-9, :column - 3] if right else delta[9:-9, column + 3:]
                    screen = delta[9:-9, column + 3:] if right else delta[9:-9, :column - 3]
                    slower = body.mean() >= 3 and screen.mean() <= body.mean() * .6
                else:
                    slower = False
                asymmetric = outside > max(30, inside * 2.5)
                if stable or (faces and strength >= .52) or (slower and asymmetric and strength >= .58):
                    candidates.append((3 * stable + 1.5 * bool(faces) + slower + asymmetric + strength, panel))
        self.panel = max(candidates, key=lambda item: item[0])[1] if candidates else None
        return self.panel


def distinct_text_edges(gray):
    """Discard flat padding before using the criterion to route OCR."""
    import cv2

    return float((cv2.absdiff(gray[:, 1:], gray[:, :-1]) >= 24).mean())
