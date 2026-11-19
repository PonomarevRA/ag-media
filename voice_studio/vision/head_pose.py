"""Estimate screen-relative head orientation from a tracked Face Landmarker crop."""

from math import atan2, degrees, hypot, isfinite

TURN_THRESHOLD_DEG = 12
TILT_THRESHOLD_DEG = 15


def head_pose(matrix):
    """Convert the face's forward axis into screen-relative yaw and pitch, in degrees."""
    if matrix.shape != (4, 4):
        return None
    right, up, toward_camera = map(float, matrix[:3, 2])
    if not all(map(isfinite, (right, up, toward_camera))) or toward_camera <= 0:
        return None
    yaw = degrees(atan2(right, hypot(up, toward_camera)))
    pitch = degrees(atan2(up, toward_camera))
    horizontal = "center"
    vertical = "center"
    if abs(yaw) >= TURN_THRESHOLD_DEG:
        horizontal = "right" if yaw > 0 else "left"
    if abs(pitch) >= TILT_THRESHOLD_DEG:
        vertical = "up" if pitch > 0 else "down"
    return {"yawDeg": round(yaw, 1), "pitchDeg": round(pitch, 1),
            "horizontal": horizontal, "vertical": vertical}
