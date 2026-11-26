"""Meter OCR within one scene and reuse successful exact crop inputs."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import file_digest
from pathlib import Path
from threading import Lock
from time import perf_counter


ACTIVE_SCENE_OCR = ContextVar("active_scene_ocr", default=None)
MAX_CACHE_ENTRIES = 256
MAX_CACHE_LINES = 128
MAX_CACHE_TEXT = 8192


@contextmanager
def scene_ocr_scope(limit=None, *, reuse=False, reserve=64):
    """Share one optional crop limit and exact-result cache across a scene worker."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("The OCR crop limit must be a non-negative integer.")
    if type(reserve) is not int or reserve < 0:
        raise ValueError("The targeted OCR reserve must be a non-negative integer.")
    scope = SceneOCRWork(limit, reuse, reserve)
    token = ACTIVE_SCENE_OCR.set(scope)
    try:
        yield scope
    finally:
        ACTIVE_SCENE_OCR.reset(token)


def crop_key(job, options):
    """Hash the exact encoded pixels submitted to OCR without hashing time or position."""
    _, path, _, width, height = job
    try:
        with Path(path).open("rb") as image:
            return options, width, height, file_digest(image, "sha256").digest()
    except OSError:
        return None


def merge_statuses(past, current):
    """Keep degraded backend states visible alongside successful batches."""
    for engine, state in current.items():
        if past.get(engine, "ok") == "ok" or state != "ok":
            past[engine] = state


class SceneOCRWork:
    """Account for every request and keep the most recent bounded raw OCR results."""

    def __init__(self, limit, reuse, reserve):
        self.limit = limit
        self.reuse = reuse
        self.reserve = reserve
        self.requested = self.dispatched = self.reused = self.dropped = self.batches = 0
        self.engine_calls = self.normal_dispatched = 0
        self.ocr_seconds = 0.0
        self.cache = OrderedDict()
        self._targeted = False
        self._lock = Lock()

    @contextmanager
    def targeted(self):
        """Allow event-driven OCR to draw on the reserved part of the crop limit."""
        previous = self._targeted
        self._targeted = True
        try:
            yield
        finally:
            self._targeted = previous

    @contextmanager
    def request(self, count):
        """Classify requested crops as dispatched, reused or dropped."""
        self.requested += count
        before = self.dispatched + self.reused
        try:
            yield
        finally:
            self.dropped += count - (self.dispatched + self.reused - before)

    def available(self, count):
        """Allow baseline work within its share and targeted work within the total."""
        if self.limit is None:
            return count
        remaining = self.limit - self.dispatched
        if not self._targeted:
            remaining = min(remaining, self.limit - self.reserve - self.normal_dispatched)
        return max(0, min(count, remaining))

    def engine_started(self):
        """Count backend launches, including Tesseract repairs in pool threads."""
        with self._lock:
            self.engine_calls += 1

    def dispatch(self, jobs, operation):
        """Time actual OCR work and charge each admitted crop once."""
        self.batches += 1
        self.dispatched += len(jobs)
        if not self._targeted:
            self.normal_dispatched += len(jobs)
        started = perf_counter()
        try:
            return operation(jobs)
        finally:
            self.ocr_seconds += perf_counter() - started

    def read_replacement(self, jobs, operation):
        """Preserve the replacement provider's merged output and apply the crop limit."""
        selected = jobs[:self.available(len(jobs))]
        if not selected:
            return {}, {}
        self.engine_started()
        return self.dispatch(selected, operation)

    def read_native(self, jobs, operation, options):
        """Return spatially unprojected readings in the same order as the request."""
        if not self.reuse:
            selected = jobs[:self.available(len(jobs))]
            if not selected:
                return [None] * len(jobs), {}
            lines, statuses = self.dispatch(selected, operation)
            return lines + [None] * (len(jobs) - len(selected)), statuses

        output = [None] * len(jobs)
        statuses, pending, repeats = {}, {}, []
        limit = self.available(len(jobs))
        for index, job in enumerate(jobs):
            key = crop_key(job, options)
            if key in self.cache:
                lines, worked = self.cache[key]
                self.cache.move_to_end(key)
                output[index] = lines
                merge_statuses(statuses, worked)
                self.reused += 1
            elif key is not None and key in pending:
                repeats.append((index, pending[key]))
            elif len(pending) < limit:
                pending[key if key is not None else index] = index

        if not pending:
            return output, statuses
        results, worked = self.dispatch([jobs[index] for index in pending.values()], operation)
        merge_statuses(statuses, worked)
        for (key, index), lines in zip(pending.items(), results):
            output[index] = lines
            if isinstance(key, tuple) and self.can_reuse(lines, worked):
                self.remember(key, lines, worked)

        retry = []
        for index, previous in repeats:
            if self.can_reuse(output[previous], worked):
                output[index] = output[previous]
                self.reused += 1
            else:
                # Keep uncertain repeats eligible for their own backend attempt.
                retry.append(index)
        retry = retry[:self.available(len(retry))]
        if retry:
            results, worked = self.dispatch([jobs[index] for index in retry], operation)
            merge_statuses(statuses, worked)
            for index, lines in zip(retry, results):
                output[index] = lines
        return output, statuses

    @staticmethod
    def can_reuse(lines, statuses):
        """Treat real empty reads as valid, excluding absent or degraded crops."""
        return lines is not None and bool(statuses) and all(value == "ok" for value in statuses.values())

    def remember(self, key, lines, statuses):
        """Retain only bounded entries with complete backend results."""
        if len(lines) > MAX_CACHE_LINES or sum(len(line.get("text", "")) for line in lines) > MAX_CACHE_TEXT:
            return
        if len(self.cache) >= MAX_CACHE_ENTRIES:
            self.cache.popitem(last=False)
        self.cache[key] = lines, statuses

    def summary(self):
        """Return measured OCR work without leaking crop content or file paths."""
        return {"requested": self.requested, "dispatched": self.dispatched, "reused": self.reused,
                "dropped": self.dropped, "batches": self.batches, "engineCalls": self.engine_calls,
                "ocrMs": round(self.ocr_seconds * 1000), "cacheEntries": len(self.cache)}
