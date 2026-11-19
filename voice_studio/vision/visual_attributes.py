"""Estimate head turn and a dominant clothing patch from a clear detector sample."""

import cv2
import numpy as np

HUES = ("красный", "оранжевый", "жёлтый", "зелёный", "голубой", "синий", "фиолетовый", "розовый")


def head_turn(landmarks):
    """Estimate horizontal head orientation in image coordinates from YuNet landmarks."""
    right, left, nose = np.asarray(landmarks[:6]).reshape((3, 2))
    axis = left - right
    length_sq = float(axis @ axis)
    if length_sq < 15 ** 2:
        return None
    fraction = float((nose - right) @ axis) / length_sq
    if not -.2 <= fraction <= 1.2:
        return None
    if abs(fraction - .5) <= .16:
        return "прямо"
    return "вправо" if (fraction > .5) == (axis[0] > 0) else "влево"


def palette(patch):
    """Return a color only when a single broad HSV class dominates the patch."""
    if min(patch.shape[:2]) < 10:
        return None
    hsv = cv2.cvtColor(cv2.resize(patch, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h, s, v = hsv.reshape(-1, 3).astype(int).T
    labels = np.digitize(h, [10, 23, 38, 85, 102, 130, 158, 171]) % 8
    labels[s < 38] = 8
    labels[(s < 42) & (v >= 182)] = 9
    labels[v < 55] = 10
    strength = np.bincount(labels, minlength=11)
    index = int(strength.argmax())
    if strength[index] < .65 * len(labels):
        return None
    return (*HUES, "серый", "белый", "тёмный")[index]


def visible_attributes(frame, detection, time_ms):
    """Report clothing color only when both shoulder samples agree."""
    if min(detection[2:4]) < 44 or detection[-1] < .86:
        return {}
    result = {}
    direction = head_turn(detection[4:14])
    if direction:
        result["headTurn"] = direction
    x, y, w, h = map(float, detection[:4])
    frame_height, frame_width = frame.shape[:2]
    left, mid_left, mid_right, right = [round(x + w * fraction) for fraction in (-.12, .28, .72, 1.12)]
    top, bottom = [round(y + h * fraction) for fraction in (1.15, 1.55)]
    if 0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height:
        color = palette(frame[top:bottom, left:mid_left])
        if color and color == palette(frame[top:bottom, mid_right:right]):
            result["clothingColor"] = color
    if result:
        result["timeMs"] = time_ms
    return result
