"""Collect process timings without requiring Unix-only resource counters."""

import sys
from time import process_time

try:
    import resource
except ImportError:
    resource = None


def process_usage():
    """Return process CPU milliseconds and peak RSS MiB when the OS exposes it."""
    if resource is None:
        return round(1000 * process_time()), None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return round(1000 * (usage.ru_utime + usage.ru_stime)), round(usage.ru_maxrss / rss_scale, 1)
