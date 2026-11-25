"""Measure spatially coherent vehicle color in original source frames."""

import cv2
import numpy as np

from voice_studio.vision.vehicle_evidence import VEHICLE_LABELS, valid_box

CROP_WIDTH, CROP_HEIGHT = 96, 64
GRID_X, GRID_Y = np.meshgrid(
    (np.arange(CROP_WIDTH, dtype=np.float32) + .5) / CROP_WIDTH,
    (np.arange(CROP_HEIGHT, dtype=np.float32) + .5) / CROP_HEIGHT,
)
BORDER_X, BORDER_Y = np.meshgrid((np.arange(160) + .5) / 160, (np.arange(90) + .5) / 90)
KERNEL = np.ones((3, 5), dtype=np.uint8)


def inside_box(x, y, box):
    """Return the mask of points inside a normalized detector box."""
    left, top, width, height = box
    return (x >= left) & (x <= left + width) & (y >= top) & (y <= top + height)


def distance_from_center(x, y, box):
    """Compare competing box centers in units of each box's width and height."""
    left, top, width, height = box
    return ((x - left) / width - .5) ** 2 + ((y - top) / height - .5) ** 2


def crop_mask(box, neighbors, x, y):
    """Limit measurements to the body and the closer vehicle in shared pixels."""
    center_distance = (GRID_X - .5) ** 2 + (GRID_Y - .5) ** 2
    inner = (center_distance < .23) & (GRID_Y >= .26) & (GRID_Y < .91)
    left, top, width, height = box
    for other in neighbors:
        u, v, w, h = other
        area = max(0, min(left + width, u + w) - max(left, u)) * max(0, min(top + height, v + h) - max(top, v))
        if .18 * (width * height + w * h - area) < area < .85 * min(width * height, w * h):
            inner &= ~inside_box(x, y, other) | (center_distance < distance_from_center(x, y, other))
    return inner


def hue_mask(hue, center):
    """Select a hue range that also works across the circular red boundary."""
    distance = np.abs(hue - center)
    return (distance <= 8) | (distance >= 172)


def body_color(image, frame_hsv, item, neighbors):
    """Return the strongest coherent chromatic patch and nearby background share."""
    box = item["box"]
    left, top, width, height = box
    x, y = left + GRID_X * width, top + GRID_Y * height
    inner = crop_mask(box, neighbors, x, y)
    pixel_count = np.count_nonzero(inner)
    if pixel_count < 800:
        return None
    crop = cv2.remap(image, x * image.shape[1], y * image.shape[0], cv2.INTER_LINEAR)
    hue, sat, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    foreground = inner & (sat >= 85) & (value >= 28)
    if np.count_nonzero(foreground) < .20 * pixel_count:
        return None

    # Sample above the center of the detection, away from other vehicle boxes.
    outside = (BORDER_Y > .16) & (BORDER_Y < top + .36 * height)
    for other in [box, *neighbors]:
        outside &= ~inside_box(BORDER_X, BORDER_Y, other)
    outside_count = np.count_nonzero(outside)
    if not outside_count:
        return None
    outside &= (frame_hsv[:, :, 1] >= 85) & (frame_hsv[:, :, 2] >= 28)
    hue = hue.astype(np.int16)
    outside_hue = frame_hsv[:, :, 0].astype(np.int16)
    strongest, best_score = None, 0
    for center in range(0, 180, 10):
        pixels = foreground & hue_mask(hue, center)
        if np.count_nonzero(pixels) < .20 * pixel_count:
            continue
        filled = cv2.morphologyEx(pixels.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL) & inner
        _, components, stats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
        if len(stats) < 2:
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        component = pixels & (components == largest)
        coverage = int(np.count_nonzero(component)) / int(pixel_count)
        background = int(np.count_nonzero(outside & hue_mask(outside_hue, center))) / int(outside_count)
        if (coverage < .20 or coverage - background < .10
                or stats[largest, cv2.CC_STAT_WIDTH] < 52
                or stats[largest, cv2.CC_STAT_AREA] < .08 * CROP_WIDTH * CROP_HEIGHT):
            continue
        saturation = float(np.median(sat[component])) / 255
        score = .45 * coverage + .55 * saturation - .4 * background
        if score > best_score:
            angles = hue[component] * (np.pi / 90)
            angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2 * np.pi))
            strongest = {"hue": round(angle / (2 * np.pi), 4), "coverage": round(coverage, 3),
                         "saturation": round(saturation, 3), "background": round(background, 3)}
            best_score = score
    return strongest


def measure_vehicle_bodies(image, items, siblings):
    """Measure selected vehicle boxes while excluding nearby vehicles from background."""
    frame_hsv = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2HSV)
    vehicles = [item for item in siblings if item.get("label") in VEHICLE_LABELS and valid_box(item.get("box"))]
    for item in items:
        if not valid_box(item.get("box")):
            continue
        neighbors = [other["box"] for other in vehicles if other is not item]
        evidence = body_color(image, frame_hsv, item, neighbors)
        if evidence:
            item["vehicleBodyColor"] = evidence
