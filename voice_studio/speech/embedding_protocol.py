"""Bounded wire values shared by the native embedding parent and child."""

import math
from pathlib import Path

MAX_CLUSTERS = 24
MAX_SAMPLES = 72
MAX_REQUEST_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024


def file_signature(path):
    """Identify the current local artifact without hashing hundreds of MB per request."""
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def validate_spans(spans):
    """Reject oversized or malformed millisecond windows before touching a model."""
    if not isinstance(spans, dict) or len(spans) > MAX_CLUSTERS:
        raise ValueError("Too many or invalid profile references")
    count = 0
    for number, windows in spans.items():
        if isinstance(number, bool) or not str(number).lstrip('-').isdigit():
            raise ValueError("Profile labels must be integers")
        if not isinstance(windows, (list, tuple)):
            raise ValueError("Profile windows must be lists")
        count += len(windows)
        if count > MAX_SAMPLES:
            raise ValueError("Too many profile references")
        for window in windows:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError("Profile window must contain start and end milliseconds")
            start, end = window
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                raise ValueError("Profile windows require ordered nonnegative integer milliseconds")


def validate_vectors(value, spans):
    """Validate finite bounded native vectors and restore the public integer-key ABI."""
    if not isinstance(value, dict) or set(value) != {str(number) for number in spans}:
        raise ValueError("Worker returned different profile labels")
    output = {}
    for number, windows in spans.items():
        vectors = value[str(number)]
        if not isinstance(vectors, list) or len(vectors) > len(windows):
            raise ValueError("Worker returned invalid profile samples")
        for vector in vectors:
            if not isinstance(vector, list) or not 1 <= len(vector) <= 4096:
                raise ValueError("Worker returned an invalid embedding dimension")
            for component in vector:
                if type(component) not in (int, float) or not math.isfinite(component):
                    raise ValueError("Worker returned a non-finite embedding")
        output[int(number)] = vectors
    return output
