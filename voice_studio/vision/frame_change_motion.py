"""Optional exposure and camera-motion compensation for the frame-change raster."""

import cv2
import numpy as np

from voice_studio.vision.camera_motion import estimate_transform

NOISE_SAMPLE_STEP = 16
MOTION_CANDIDATE_FRACTION = .12
MIN_MOTION_CONFIDENCE = .70
MAX_RESIDUAL_RATIO = .60
MIN_SHIFT_PX = 1.25


def adaptive_quiet_pixels(previous, current, base):
    """Preserve local contrast when the background drifts by a few 8-bit levels."""
    sample = np.s_[::NOISE_SAMPLE_STEP, ::NOISE_SAMPLE_STEP]
    delta = current[sample].astype(np.int16) - previous[sample]
    shift = np.median(delta, axis=(0, 1))
    deviations = np.max(np.abs(delta - shift), axis=2)
    noise = float(np.median(deviations))
    magnitude = float(np.max(np.abs(shift)))
    # Apply a color correction only if most samples agree and the change is modest.
    if 4 <= magnitude <= 22 and noise <= 4:
        previous = cv2.add(previous, tuple(float(value) for value in shift) + (0.,))
    threshold = max(round(.8 * base), min(base, round(3 * noise)))
    return quiet_pixels(previous, current, threshold)


def quiet_pixels(previous, current, threshold):
    """Mark pixels whose three color channels all stay below the threshold."""
    return cv2.inRange(cv2.absdiff(previous, current), (0, 0, 0), (threshold - 1,) * 3)


class CameraCompensator:
    """Try alignment for broad changes and keep it only if it explains the majority."""

    def __init__(self):
        self.allowed = None
        self.compensated_frames = 0

    def quiet_mask(self, previous, current, threshold, quiet):
        """Return a mask in current-frame coordinates, excluding newly exposed edges."""
        height, width = quiet.shape
        changed = quiet.size - cv2.countNonZero(quiet)
        if changed < MOTION_CANDIDATE_FRACTION * quiet.size:
            return quiet
        first = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        second = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        if self.allowed is None:
            self.allowed = np.full(quiet.shape, 255, dtype=np.uint8)
        try:
            motion = estimate_transform((first, self.allowed), (second, self.allowed))
        except cv2.error:
            return quiet
        if not motion or motion["confidence"] < MIN_MOTION_CONFIDENCE:
            return quiet
        scale = motion["scale"]
        dx, dy = motion["shiftX"] * width, motion["shiftY"] * height
        zoom_pixels = abs(scale - 1) * max(width, height) / 2
        # A pure vertical translation is ambiguous with scrolling on a full-screen UI.
        if max(abs(dx), zoom_pixels) < MIN_SHIFT_PX:
            return quiet
        affine = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
        affine[:, 2] += (dx, dy)
        # A constant border identifies uncovered pixels. Interpolated boundary pixels are
        # excluded too, so panning past the edge does not look like a new subject.
        aligned = cv2.warpAffine(previous, affine, (width, height))
        covered = cv2.warpAffine(self.allowed, affine, (width, height))
        uncovered = covered < 255
        aligned_quiet = quiet_pixels(aligned, current, threshold)
        aligned_quiet[uncovered] = 255
        changed -= np.count_nonzero((quiet == 0) & uncovered)
        residual = quiet.size - cv2.countNonZero(aligned_quiet)
        if residual > MAX_RESIDUAL_RATIO * changed or not changed:
            return quiet
        self.compensated_frames += 1
        return aligned_quiet
