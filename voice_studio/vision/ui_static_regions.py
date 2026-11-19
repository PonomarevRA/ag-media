"""Cheap, color-independent geometry proposals from an already decoded BGR frame.

A rectangle is only a visual anchor candidate. This module never classifies its
contents as a person, application, speech, or emotion.
"""

MAX_CANDIDATES = 64
MAX_CONTOURS = 512


def _color_mask(pixels, hsv, color):
    """Match observed color, tolerating subpixel brightness changes for any saturated hue."""
    import cv2
    import numpy as np
    if color.max() - color.min() <= 35:
        return np.max(np.abs(pixels.astype(np.int16) - color), axis=-1) <= 28
    color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue_delta = np.abs(hsv[..., 0].astype(np.int16) - int(color_hsv[0]))
    hue_delta = np.minimum(hue_delta, 180-hue_delta)
    return (hue_delta <= 10) & (hsv[..., 1] >= 65) & (hsv[..., 2] >= 50)


def _outline_facts(crop, bands):
    """Inspect perimeter pixels and sparse interior samples, rejecting filled panels and grids."""
    import cv2
    import numpy as np
    height, width = crop.shape[:2]
    if height <= 4*bands or width <= 4*bands:
        return None
    ring = np.concatenate((crop[:bands].reshape(-1, 3), crop[-bands:].reshape(-1, 3),
                           crop[:, :bands].reshape(-1, 3), crop[:, -bands:].reshape(-1, 3)))
    quantized = (ring // 24).astype(np.int16)
    color_keys = quantized[:, 0]*121 + quantized[:, 1]*11 + quantized[:, 2]
    colors, counts = np.unique(color_keys, return_counts=True)
    interior = crop[bands:-bands:4, bands:-bands:4]
    center_color = np.median(interior.reshape(-1, 3), axis=0)
    ring_hsv = cv2.cvtColor(ring.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    interior_hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
    # Background colors dominate many rings. Inspect a few modes, then demand
    # three long sides and very little of that same color inside the rectangle.
    for index in np.argsort(counts)[-12:][::-1]:
        color = np.median(ring[color_keys == colors[index]], axis=0)
        contrast = float(np.max(np.abs(color - center_color)))
        if contrast < 18:
            continue
        mask = _color_mask(ring, ring_hsv, color)
        top_end, bottom_end, left_end = bands*width, 2*bands*width, 2*bands*width+height*bands
        top = mask[:top_end].reshape(bands,width)
        bottom = mask[top_end:bottom_end].reshape(bands,width)
        left = mask[bottom_end:left_end].reshape(height,bands)
        right = mask[left_end:].reshape(height,bands)
        outside = [top[0].mean(), bottom[-1].mean(), left[:, 0].mean(), right[:, -1].mean()]
        if sum(side >= .8 for side in outside) >= 3:
            continue
        sides = [top.max(axis=0).mean(), bottom.max(axis=0).mean(),
                 left.max(axis=1).mean(), right.max(axis=1).mean()]
        if sum(side >= .8 for side in sides) < 3:
            continue
        if _color_mask(interior, interior_hsv, color).mean() > .06:
            continue
        # Full interior inspection is reserved for a qualifying color. It catches
        # thin grid lines that a sparse interior sample could otherwise miss.
        inside = crop[2*bands:-2*bands, 2*bands:-2*bands]
        inside_hsv = cv2.cvtColor(inside, cv2.COLOR_BGR2HSV)
        inside_mask = _color_mask(inside, inside_hsv, color)
        if (inside_mask.mean(axis=0) >= .8).any() and (inside_mask.mean(axis=1) >= .8).any():
            continue
        return {'outlineCoverage': round(float(sum(sides)) / 4, 3),
                'outlineColorBgr': [int(value) for value in color],
                'outlineContrast': round(contrast, 2), 'outlineType': 'thin-rectangle'}
    return None


def _overlap(left, right):
    """Return intersection over union for normalized rectangle proposals."""
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[0] + left[2], right[0] + right[2]), min(left[1] + left[3], right[1] + right[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    union = left[2]*left[3] + right[2]*right[3] - intersection
    return intersection / union if union else 0


def analysis_view(image):
    """Bound temporary geometry work to 960 pixels per dimension; never retain the view."""
    import cv2
    height, width = image.shape[:2]
    if not height or not width:
        return image
    scale = min(1, 960 / max(height, width))
    if scale < 1:
        return cv2.resize(image, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
    return image


def outline_at_box(view, box):
    """Recheck an existing normalized ROI each frame, without another contour search."""
    height, width = view.shape[:2]
    x, y, w, h = box
    crop = view[round(y*height):round((y+h)*height), round(x*width):round((x+w)*width)]
    facts = _outline_facts(crop, max(6, round(min(width, height)*.018)))
    return {'box': list(box), **facts} if facts else None


def outline_candidates(image):
    """Propose at most 64 large thin rectangles of any color, excluding solid regions.

    Analysis is bounded to 960 pixels per dimension and 512 contours. Border color,
    contrast and geometry are observations, with no activity or identity meaning.
    """
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    if not height or not width:
        return []
    view = analysis_view(image)
    height, width = view.shape[:2]
    # Channel edges detect equally bright colors that grayscale would discard.
    edges = np.zeros((height, width), np.uint8)
    for channel in cv2.split(view):
        edges = cv2.bitwise_or(edges, cv2.Canny(channel, 25, 70))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]
    candidates = []
    evaluated = []
    bands = max(6, round(min(width, height) * .018))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width*.15 or h < height*.12 or w*h < width*height*.035:
            continue
        # Include pixels on either side of Canny's edge so the ring contains the
        # actual line, then reject interior color rather than trusting an edge.
        x0, y0 = max(0, x-2), max(0, y-2)
        x1, y1 = min(width, x+w+2), min(height, y+h+2)
        box = [round(v, 5) for v in (x0/width, y0/height, (x1-x0)/width, (y1-y0)/height)]
        if any(_overlap(box, previous) > .88 for previous in evaluated):
            continue
        evaluated.append(box)
        facts = _outline_facts(view[y0:y1, x0:x1], bands)
        if facts:
            candidates.append({'box': box, **facts})
            if len(candidates) >= MAX_CANDIDATES:
                break
    return _without_grid(candidates)


def _without_grid(candidates):
    """Suppress repeated touching cells of a drawn grid, a common geometric false cue."""
    if len(candidates) < 4:
        return candidates
    grid = set()
    for index, row in enumerate(candidates):
        x, y, w, h = row['box']
        aligned = []
        for other_index, other in enumerate(candidates):
            a, b, c, d = other['box']
            if abs(w-c) > .015 or abs(h-d) > .015:
                continue
            if abs(x-a) <= w*1.1 and abs(y-b) <= h*1.1:
                aligned.append(other_index)
        if len(aligned) >= 4:
            grid.update(aligned)
    return [row for index, row in enumerate(candidates) if index not in grid]
