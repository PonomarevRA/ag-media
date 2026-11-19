"""Use Accelerate to compute the same autocorrelation numerators on macOS."""

from array import array
from ctypes import CDLL, POINTER, c_double, c_long, c_ulong
from functools import lru_cache
from itertools import accumulate
from math import sqrt
import sys


@lru_cache(maxsize=1)
def correlate():
    """Load the optional double-precision Accelerate convolution API."""
    if sys.platform != "darwin":
        return None
    try:
        library = CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        entry = library.vDSP_convD
    except (AttributeError, OSError):
        return None
    doubles = POINTER(c_double)
    entry.argtypes = [doubles, c_long, doubles, c_long, doubles, c_long, c_ulong, c_ulong]
    entry.restype = None
    return entry


def scores(frame, first, last):
    """Compute the normalized positive-lag correlation over the existing frame."""
    width = len(frame)
    coefficients = (c_double * width).from_buffer_copy(array("d", frame))
    padded = (c_double * (width + last)).from_buffer_copy(array("d", [*frame, *([0.0] * last)]))
    dots = (c_double * (last + 1))()
    correlate()(padded, 1, coefficients, 1, dots, 1, len(dots), width)
    power = list(accumulate((sample * sample for sample in frame), initial=0.0))
    values = []
    for lag in range(first, last + 1):
        denominator = sqrt(power[width - lag] * (power[-1] - power[lag]))
        values.append(dots[lag] / denominator if denominator else 0)
    return values
