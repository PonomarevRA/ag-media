"""Group shot-local pixel changes and retain useful source-frame witnesses."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappush, heapreplace
from itertools import count


EVENT_SAMPLING = ("event", "event-ocr-reuse")
MAX_TRANSIENT_MS = 500
MAX_LINK_GAP_MS = 160
MAX_EVENT_MS = 1600
MAX_ACTIVE_EVENTS = 64
MAX_REPORTED_EVENTS = 2048
GRID_SIDE = 6


def overlap(first, second):
    """Return the intersection over union of two normalized boxes."""
    x, y, width, height = first
    a, b, w, h = second
    shared = max(0, min(x + width, a + w) - max(x, a)) * max(0, min(y + height, b + h) - max(y, b))
    return shared / (width * height + w * h - shared) if shared else 0.0


def pixel_signatures(previous, current, regions):
    """Keep two tiny RGB samples per eligible region to verify a later pixel return."""
    import cv2

    height, width = current.shape[:2]
    signatures = []
    for region in regions:
        x, y, w, h = region["box"]
        left, top = round(x * width), round(y * height)
        right, bottom = round((x + w) * width), round((y + h) * height)
        if region["changedPixels"] < 90 or min(right - left, bottom - top) < 6:
            signatures.append(None)
            continue
        section = (slice(top, bottom), slice(left, right))
        size = (GRID_SIDE, GRID_SIDE)
        before = cv2.resize(previous[section], size, interpolation=cv2.INTER_AREA).tobytes()
        after = cv2.resize(current[section], size, interpolation=cv2.INTER_AREA).tobytes()
        signatures.append((before, after))
    return signatures


def signature_distance(first, second):
    """Measure inter-frame color differences in 8-bit levels across a thumbnail patch."""
    return sum(abs(a - b) for a, b in zip(first, second)) / len(first)


@dataclass(slots=True)
class Witness:
    """One measured region and its exact native source-frame timestamp."""

    row: dict
    index: int

    @property
    def region(self):
        return self.row["regions"][self.index]

    @property
    def time(self):
        return self.row["startMs"]

    @property
    def pixels(self):
        return self.region["changedPixels"]

    @property
    def signature(self):
        return self.row["_pixelSignatures"][self.index]


@dataclass(slots=True)
class Event:
    """An onset, strongest visible change and last measured region within one shot."""

    sequence: int
    onset: Witness
    peak: Witness
    last: Witness
    samples: int = 1
    returned_at: int | None = None

    def advance(self, witness):
        """Update the spatial link and keep the strongest frame before disappearance."""
        self.last = witness
        self.samples += 1
        if witness.pixels > self.peak.pixels:
            self.peak = witness

    def returned(self, witness, shared):
        """Require same-area before and after pixels before treating a change as brief."""
        if (shared < .80 or witness.time - self.onset.time > MAX_TRANSIENT_MS
                or not self.onset.signature or not witness.signature):
            return False
        original, appeared = self.onset.signature
        visible, restored = witness.signature
        return (signature_distance(original, restored) <= 7
                and signature_distance(original, appeared) >= 11
                and signature_distance(restored, visible) >= 11)


class EventCollector:
    """Keep a bounded set of events and prefer verified brief changes when saturated."""

    def __init__(self):
        self.active = []
        self.kept = []
        self.sequence = count()
        self.completed = 0

    def finish(self, event):
        """Keep a completed event without unbounded metadata growth."""
        self.completed += 1
        score = (event.returned_at is not None, min(event.samples, 4), event.peak.pixels, -event.sequence)
        entry = (score, event)
        if len(self.kept) < MAX_REPORTED_EVENTS:
            heappush(self.kept, entry)
        elif score > self.kept[0][0]:
            heapreplace(self.kept, entry)

    def prune(self, stamp):
        """Let a recent onset wait for a short stable hold to end."""
        remaining = []
        for event in self.active:
            expired = (stamp - event.onset.time > MAX_EVENT_MS
                       or (stamp - event.last.time > MAX_LINK_GAP_MS and stamp - event.onset.time > MAX_TRANSIENT_MS))
            if expired:
                self.finish(event)
            else:
                remaining.append(event)
        self.active = remaining

    def observe(self, row):
        """Associate at most one current region with each live event."""
        self.prune(row["startMs"])
        prior = list(self.active)
        matched = set()
        for index, _ in enumerate(row["regions"]):
            witness = Witness(row, index)
            options = []
            for event in prior:
                if event.sequence in matched:
                    continue
                first_overlap = overlap(event.onset.region["box"], witness.region["box"])
                returned = event.returned(witness, first_overlap)
                shared = overlap(event.last.region["box"], witness.region["box"])
                if returned or (shared >= .28 and witness.time - event.last.time <= MAX_LINK_GAP_MS):
                    options.append((returned, first_overlap if returned else shared, event))
            options.sort(key=lambda option: option[:2], reverse=True)
            if options and unique_best(options):
                returned, _, event = options[0]
                matched.add(event.sequence)
                if returned:
                    event.returned_at = witness.time
                    event.samples += 1
                    self.active.remove(event)
                    self.finish(event)
                else:
                    event.advance(witness)
            else:
                self.active.append(Event(next(self.sequence), witness, witness, witness))
        while len(self.active) > MAX_ACTIVE_EVENTS:
            self.finish(self.active.pop(0))

    def close(self):
        """End active events at a cut or the end of available measurements."""
        for event in self.active:
            self.finish(event)
        self.active.clear()


def unique_best(options):
    """Avoid linking a region when near-tied events claim the same observation."""
    if len(options) == 1:
        return True
    best, second = options[:2]
    return best[0] != second[0] or best[1] - second[1] >= .08


def group_events(rows, cut_stamps):
    """Annotate measured onset and peak frames before per-second thinning."""
    collector = EventCollector()
    shot = None
    for row in rows:
        next_shot = bisect_right(cut_stamps, row["startMs"])
        if next_shot != shot:
            collector.close()
            shot = next_shot
        collector.observe(row)
    collector.close()
    ordered = sorted((event for _, event in collector.kept), key=lambda event: (event.onset.time, event.sequence))
    events = []
    for event in ordered:
        returned = event.returned_at is not None
        onset, peak = event.onset, event.peak
        use_end = event.returned_at if returned else event.last.row["endMs"]
        facts = {"eventId": f"{onset.row['sourceFrame']}:{onset.index}",
                 "eventStartMs": onset.time, "eventEndMs": use_end, "eventSamples": event.samples,
                 "eventBasis": "roi-pixel-return" if returned else "adjacent-region-overlap",
                 "transient": returned}
        onset.row.setdefault("eventWitnesses", []).append({**facts, "index": onset.index, "role": "onset"})
        if peak is not onset:
            peak.row.setdefault("eventWitnesses", []).append({**facts, "index": peak.index, "role": "peak"})
        events.append({**facts, "onsetFrame": onset.row["sourceFrame"], "peakFrame": peak.row["sourceFrame"]})
    return events, collector.completed - len(events)


def prefer_witness_rows(bucket, limit):
    """Spend the existing frame budget on verified or stronger event witnesses first."""
    from voice_studio.vision.scene_sampling import evenly_spaced

    preferred, remaining = [], []
    for row in bucket:
        if row.get("eventWitnesses"):
            preferred.append(row)
        else:
            remaining.append(row)

    def priority(row):
        """Favor brief appearances, then a stronger change within the same budget."""
        return max((item["transient"], item["eventSamples"] > 1, row["regions"][item["index"]]["changedPixels"])
                   for item in row["eventWitnesses"])

    preferred.sort(key=priority, reverse=True)
    selected = preferred[:limit]
    selected.extend(evenly_spaced(remaining, limit - len(selected)))
    return sorted(selected, key=lambda row: row["startMs"])


def constrain_events(report, cut_stamps):
    """Drop event links that later editor-supplied cuts show to be cross-shot."""
    if not report.get("events"):
        return
    retained = []
    invalid = set()
    cuts = sorted(cut_stamps)
    for event in report["events"]:
        start, end = event["eventStartMs"], event["eventEndMs"]
        after_start = bisect_right(cuts, start)
        # The return frame is part of the evidence for a transient event.
        crosses = after_start < len(cuts) and (cuts[after_start] < end
                   or (event["transient"] and cuts[after_start] == end))
        if crosses or (event["transient"] and after_start and cuts[after_start - 1] == start):
            invalid.add(event["eventId"])
        else:
            retained.append(event)
    if invalid:
        report["events"] = retained
        for row in (*report["rows"], *report.get("objectRows", ())):
            witnesses = row.get("eventWitnesses", ())
            if witnesses:
                row["eventWitnesses"] = [item for item in witnesses if item["eventId"] not in invalid]


def direct_crop_witnesses(rows, side):
    """Choose at most one native crop per transient event and per source-clock second."""
    per_event = {}
    for row in rows:
        for witness in row.get("eventWitnesses", ()):
            if not witness["transient"]:
                continue
            index = witness["index"]
            region = row["regions"][index]
            _, _, width, height = region["box"]
            area = width * height
            pixels = region["changedPixels"]
            if not .0015 <= area <= .18 or min(width, height) < .018 or pixels < 90:
                continue
            density = pixels / (area * side * side)
            if density < .12:
                continue
            priority = (witness["role"] == "peak", density * pixels)
            record = (priority, row, index, witness)
            previous = per_event.get(witness["eventId"])
            if previous is None or priority > previous[0]:
                per_event[witness["eventId"]] = record
    per_second = {}
    for record in per_event.values():
        priority, row, _, _ = record
        second = row["startMs"] // 1000
        previous = per_second.get(second)
        if previous is None or priority[1] > previous[0][1]:
            per_second[second] = record
    return [entry[1:] for _, entry in sorted(per_second.items())]
