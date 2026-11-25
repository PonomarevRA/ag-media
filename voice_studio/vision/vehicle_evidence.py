"""Shared contract for optional vehicle evidence from original detector crops."""

from math import isfinite

VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})


def valid_box(box):
    """Validate a normalized detector box before sampling source pixels."""
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(value) in (int, float) and 0 <= value <= 1 for value in box)
            and min(box[2:]) > 0)


def validated_body_color(value):
    """Accept only compact, finite normalized measurements from a source crop."""
    fields = ("hue", "coverage", "saturation", "background")
    if not isinstance(value, dict):
        return None
    for field in fields:
        number = value.get(field)
        if type(number) not in (int, float) or not isfinite(number) or not 0 <= number <= 1:
            return None
    if value["coverage"] < .20 or value["saturation"] < .33 or value["coverage"] - value["background"] < .099:
        return None
    return {field: value[field] for field in fields}


def hue_distance(first, second):
    """Measure the shorter distance around a normalized hue circle."""
    difference = abs(first - second)
    return min(difference, 1 - difference)
