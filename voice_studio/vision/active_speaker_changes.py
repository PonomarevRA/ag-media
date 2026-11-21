"""Route localized head motion to a bounded second audiovisual model pass."""

import numpy as np

MAX_RECHECKS = 4
MAX_FRAMES = 60
CONTEXT_FRAMES = 8


def head_change_window(video, scores, threshold):
    """Select a short interval with repeated lower-face changes and uncertain activity."""
    if len(scores) < 14 or not .18 <= np.mean(scores >= 0) < .8:
        return None
    # The normalized Light-ASD crop cancels most translation. Contrast the lower
    # part of the same face against upper-face motion to reject broad camera changes.
    pixels = video[:, ::4, ::4].astype(np.int16)
    delta = np.abs(np.diff(pixels, axis=0)) >= max(8, threshold // 2)
    lower = delta[:, 14:25, 7:21].mean(axis=(1, 2))
    upper = delta[:, 4:14, 7:21].mean(axis=(1, 2))
    motion = np.flatnonzero((lower >= .045) & (lower >= 1.35 * upper + .015)) + 1
    if len(motion) < 3:
        return None
    # Pick a measured burst, rather than joining changes far apart in a long cue.
    groups = np.split(motion, np.flatnonzero(np.diff(motion) > 7) + 1)
    burst = max(groups, key=len)
    if len(burst) < 3:
        return None
    center = int(np.median(burst))
    left = max(0, center - MAX_FRAMES // 2)
    right = min(len(scores), left + MAX_FRAMES)
    left = max(0, right - MAX_FRAMES)
    return left, right


def recheck_activity(video, audio, scores, start, end, threshold, infer):
    """Accept only substantially stronger independent votes from the same A/V track.

    All indices use the 25 fps proxy. MFCC windows use four samples per frame.
    Motion selects where to infer and never counts as a positive speech vote.
    """
    window = head_change_window(video[start:end], scores[start:end], threshold)
    if window is None:
        return scores, 0
    left, right = (start + index for index in window)
    first = max(0, left - CONTEXT_FRAMES)
    last = min(len(scores), right + CONTEXT_FRAMES)
    measured = infer(video[first:last], audio[first * 4:last * 4])[left - first:right - first]
    baseline = scores[left:right]
    if (len(measured) != len(baseline) or np.mean(measured >= 0) < .88
            or np.count_nonzero(measured >= 0) - np.count_nonzero(baseline >= 0) < 4):
        return scores, 1
    # Average independent model votes to preserve conflicting baseline evidence.
    fused = (baseline + measured) / 2
    if np.count_nonzero(fused >= 0) - np.count_nonzero(baseline >= 0) < 3:
        return scores, 1
    improved = scores.copy()
    improved[left:right] = fused
    return improved, 1
