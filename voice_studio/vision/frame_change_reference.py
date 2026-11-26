"""Detect local changes relative to adjacent and last stable frame tiles."""

import cv2
import numpy as np

from voice_studio.vision.frame_change_events import REFERENCE_BASIS, overlap


class StableRegionChanges:
    """Maintain one native scan-sized reference and short-lived per-tile evidence."""

    def __init__(self, side, tile, threshold):
        self.tile = tile
        self.threshold = threshold
        self.shape = (side // tile, side // tile)
        self.reference = None
        self.credit = np.zeros(self.shape, dtype=np.float32)
        self.still = np.zeros(self.shape, dtype=np.uint8)
        self.weak_steps = np.zeros(self.shape, dtype=np.uint8)
        self.reported = np.zeros(self.shape, dtype=bool)
        self.strong_tiles = 0
        self.confirmed_tiles = 0

    def tiles(self, mask):
        """Return the fraction of affected pixels in each fixed spatial tile."""
        rows, columns = self.shape
        return cv2.resize(mask, (columns, rows), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    def pixels(self, mask):
        """Project one boolean tile decision back to the scan grid."""
        return mask.repeat(self.tile, axis=0).repeat(self.tile, axis=1)

    def reset(self, current):
        """Start a new reference after a wide visual change."""
        self.reference = current.copy()
        self.credit.fill(0)
        self.still.fill(0)
        self.weak_steps.fill(0)
        self.reported.fill(False)

    def observe(self, previous, current, *, broad):
        """Produce a sparse pixel mask for sharp jumps or corroborated weak changes."""
        if self.reference is None:
            self.reset(previous)
        if broad:
            self.reset(current)
            return None
        weak_threshold = max(7, self.threshold // 2)
        adjacent = cv2.absdiff(previous, current)
        weak_bound = (weak_threshold - 1,) * 3
        strong_bound = (self.threshold - 1,) * 3
        adjacent_fraction = 1 - self.tiles(cv2.inRange(adjacent, (0, 0, 0), weak_bound))
        reference_quiet = cv2.inRange(cv2.absdiff(self.reference, current), (0, 0, 0), weak_bound)
        reference_fraction = 1 - self.tiles(reference_quiet)
        strong_fraction = 1 - self.tiles(cv2.inRange(adjacent, (0, 0, 0), strong_bound))
        strong = (strong_fraction >= .22) & (reference_fraction >= .22)
        weak = (adjacent_fraction >= .06) & (reference_fraction >= .22)

        # Accumulate only local changes with both kinds of evidence. Isolated
        # mild differences decay, and a settled patch replaces its own reference.
        self.credit = np.clip(self.credit + np.where(weak, reference_fraction - .19, -.24), 0, 1)
        self.weak_steps = np.where(weak, np.minimum(self.weak_steps + 1, 8), 0)
        self.still = np.where(adjacent_fraction < .06, np.minimum(self.still + 1, 8), 0)
        confirmed = (self.weak_steps >= 2) & (self.credit >= .36) & ~self.reported
        settled = self.still == 4
        if settled.any():
            settled_pixels = self.pixels(settled)
            self.reference[settled_pixels] = current[settled_pixels]
            self.credit[settled] = 0
            self.reported[settled] = False
        self.reported |= confirmed
        if not (strong | confirmed).any():
            return None
        self.strong_tiles += int(strong.sum())
        self.confirmed_tiles += int((confirmed & ~strong).sum())
        return cv2.bitwise_or(reference_quiet, (~self.pixels(strong | confirmed)).astype(np.uint8) * 255)

    def summary(self):
        """Count emitted tile observations by the type of evidence."""
        return {"strongTiles": self.strong_tiles, "confirmedWeakTiles": self.confirmed_tiles}


def mark_reference_regions(image, regions):
    """Attach a scale-local focus measurement to candidate boxes on the scan."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    height, width = gray.shape
    for region in regions:
        x, y, w, h = region["box"]
        patch = laplacian[round(y * height):round((y + h) * height),
                          round(x * width):round((x + w) * width)]
        region["sharpness"] = round(float(patch.var()), 2) if patch.size else 0.0
        region["detectionBasis"] = REFERENCE_BASIS


def merge_reference_regions(adjacent, stable, limit):
    """Reserve a spatially distinct reference observation before filling adjacent slots."""
    adjacent = [dict(region) for region in adjacent or ()]
    distinct = []
    for region in stable:
        nearest = max(adjacent, key=lambda item: overlap(region["box"], item["box"]), default=None)
        if nearest is not None and overlap(region["box"], nearest["box"]) >= .45:
            nearest.update({key: region[key] for key in ("detectionBasis", "sharpness")})
        else:
            distinct.append({**region, "referenceOnly": True})
    if not distinct:
        return adjacent[:limit]
    # Time-distributed thinning later decides which witnesses can reach OCR.
    return (distinct[:2] + adjacent + distinct[2:])[:limit]
