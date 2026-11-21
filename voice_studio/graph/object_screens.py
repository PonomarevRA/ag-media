"""Infer display context from timed UI evidence without replacing detector observations."""

from bisect import bisect_left
from collections import defaultdict
import re

from voice_studio.graph.media_graph_objects import overlap
from voice_studio.graph.media_graph_interfaces import PANEL_TYPES

DESKTOP_COMMANDS = ("selection pane", "picture layout", "picture effects", "behind text", "in front of text",
                    "edit wrap points", "position wrap", "bring forward", "send backward",
                    "layout options", "remove background", "keyboard shortcuts", "file explorer")
AMBIGUOUS_DISPLAY_LABELS = {"book", "refrigerator", "tv screen", "screen"}
DEVICES = {"cell phone", "laptop", "tv", "monitor"}


PANEL_MATCH_RADIUS_MS = 750


class VerifiedPanels:
    """Locate measured panels at or near a frame without crossing a cut or negative check."""

    def __init__(self, graph, frames, coordinates):
        self.graph = graph
        self.exact = {}
        self.shots = defaultdict(list)
        for frame in frames:
            stamp = frame.get("timeMs")
            if type(stamp) is not int or not 0 <= stamp < graph.duration:
                continue
            boxes = [box for panel in frame.get("interfaces") or ()
                     if isinstance(panel, dict) and panel.get("panelType") in PANEL_TYPES
                     and (box := coordinates(panel.get("box")))]
            if not boxes and not frame.get("panelChecked"):
                continue
            self.exact[stamp] = boxes
            shot = next(graph.intersecting_shots(stamp, stamp + 1), None)
            if shot:
                self.shots[shot].append((stamp, boxes))
        for checks in self.shots.values():
            checks.sort(key=lambda check: check[0])

    def measurement_at(self, stamp):
        """Return the nearest measurement and its source timestamp, including negative checks."""
        if stamp in self.exact:
            return stamp, self.exact[stamp]
        shot = next(self.graph.intersecting_shots(stamp, stamp + 1), None)
        checks = self.shots.get(shot, ())
        index = bisect_left(checks, stamp, key=lambda check: check[0])
        nearest = min(checks[max(0, index - 1):index + 1], key=lambda check: abs(check[0] - stamp), default=None)
        if nearest and abs(nearest[0] - stamp) < PANEL_MATCH_RADIUS_MS:
            return nearest
        return stamp, ()


def contained_elements(entries, box, coordinates):
    """Use only elements whose center and most of their area lie inside the object."""
    x, y, w, h = box
    for entry in entries:
        bounds = coordinates(entry.get("box")) if isinstance(entry, dict) else None
        if not bounds:
            continue
        a, b, u, v = bounds
        common = max(0, min(x + w, a + u) - max(x, a)) * max(0, min(y + h, b + v) - max(y, b))
        if x <= a + u / 2 <= x + w and y <= b + v / 2 <= y + h and common >= .65 * u * v:
            yield entry


def normalized_text(entries):
    """Normalize local OCR strings so phrases survive punctuation and case variation."""
    words = set()
    for entry in entries:
        label = re.sub(r"[^\w]+", " ", str(entry.get("text") or "").casefold()).strip()
        if len(label) >= 3:
            words.add(label)
    return words


def screen_evidence(label, box, frame, coordinates):
    """Combine spatially aligned layout and text signals and abstain on ordinary printed text."""
    own = label.casefold()
    if own not in DEVICES | AMBIGUOUS_DISPLAY_LABELS:
        return None
    words = normalized_text(contained_elements(frame.get("text") or (), box, coordinates))
    structured = False
    device_overlap = False
    mobile = own == "cell phone"
    for layout in frame.get("interfaces") or ():
        if not isinstance(layout, dict) or not (region := coordinates(layout.get("box"))):
            continue
        alignment = overlap(box, region)
        if alignment < .48:
            continue
        local = list(contained_elements(layout.get("elements") or (), box, coordinates))
        local_words = normalized_text(local)
        words.update(local_words)
        controls = sum(1 for element in local if element.get("text") and
                       element.get("role") in {"button_candidate", "field_candidate"})
        structured |= controls >= 2
        if alignment >= .68 and len(local_words) >= 3:
            device_overlap = True
            mobile |= layout.get("deviceLabel") == "cell phone"
    phrases = sorted(command for command in DESKTOP_COMMANDS if any(command in text for text in words))
    desktop = len(phrases) >= 2 and len(words) >= 3 and box[2] * box[3] >= .12
    if desktop and not mobile:
        return {"type": "ui", "displayLabel": "monitor" if own in AMBIGUOUS_DISPLAY_LABELS else label,
                "basis": "desktop-controls", "signals": phrases}
    if len(words) >= 3 and (structured or (device_overlap and own in DEVICES)):
        return {"type": "mobile-ui" if mobile else "ui", "basis": "layout-and-text"}
    return None


def panel_scroll_previews(graph, tracks, pages, coordinates):
    """Attach a temporary page only to the best scene-local profile supported by its measured viewports."""
    output = {}
    sessions = []
    for track_id, shot, samples in tracks:
        panel_samples = [(s["startMs"], s["data"]["box"]) for s in samples
                         if s["data"].get("source") == "interface-layout"]
        if panel_samples:
            track = graph.nodes[track_id]
            sessions.append((track["data"]["profileId"], graph.nodes.get(shot), panel_samples))
    for page in pages:
        if not page.get("previewUrl"):
            continue
        matches = {}
        for view in page.get("viewports", ()):
            clock, box = view.get("timeMs"), coordinates(view.get("box"))
            if not box or type(clock) is not int:
                continue
            for profile, shot, samples in sessions:
                if shot and not shot["startMs"] <= clock < shot["endMs"]:
                    continue
                best = max((overlap(box, bounds) for at, bounds in samples if abs(at - clock) <= 750), default=0)
                if best >= .5:
                    matches[profile] = matches.get(profile, 0) + best
        ranked = sorted(matches, key=matches.get, reverse=True)
        if ranked and (len(ranked) == 1 or matches[ranked[0]] > 1.2 * matches[ranked[1]]):
            output.setdefault(ranked[0], []).append({
                **{field: page[field] for field in ("previewUrl", "startMs", "endMs")},
                **{field: page[field] for field in ("height", "viewportHeight", "addedPixels") if field in page},
                "markdownAvailable": "mosaicText" in page,
            })
    for matches in output.values():
        matches.sort(key=lambda row: (-row.get("addedPixels", 0), row["startMs"]))
    return output
