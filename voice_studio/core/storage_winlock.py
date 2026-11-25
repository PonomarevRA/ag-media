"""Windows shared and exclusive leases on byte zero of the storage lock file."""

from contextlib import contextmanager
from ctypes import POINTER, Structure, WinDLL, WinError, byref, c_size_t, get_last_error
from ctypes.wintypes import BOOL, DWORD, HANDLE
from errno import EAGAIN
from msvcrt import get_osfhandle


class Overlapped(Structure):
    """The LockFileEx byte offset is zero in every Studio lease."""

    _fields_ = [("internal", c_size_t), ("internal_high", c_size_t),
                ("offset", DWORD), ("offset_high", DWORD), ("event", HANDLE)]


kernel32 = WinDLL("kernel32", use_last_error=True)
lock_file = kernel32.LockFileEx
lock_file.argtypes = (HANDLE, DWORD, DWORD, DWORD, DWORD, POINTER(Overlapped))
lock_file.restype = BOOL
unlock_file = kernel32.UnlockFileEx
unlock_file.argtypes = (HANDLE, DWORD, DWORD, DWORD, POINTER(Overlapped))
unlock_file.restype = BOOL

EXCLUSIVE = 2
FAIL_IMMEDIATELY = 1
LOCK_VIOLATION = 33


@contextmanager
def windows_lease(handle, *, exclusive, nonblocking):
    """Hold a synchronous kernel lease for the lifetime of the context."""
    descriptor = get_osfhandle(handle.fileno())
    flags = (EXCLUSIVE if exclusive else 0) | (FAIL_IMMEDIATELY if nonblocking else 0)
    location = Overlapped()
    if not lock_file(descriptor, flags, 0, 1, 0, byref(location)):
        error = get_last_error()
        if error == LOCK_VIOLATION:
            raise BlockingIOError(EAGAIN, "Storage lease is held")
        raise WinError(error)
    try:
        yield
    finally:
        unlock_file(descriptor, 0, 1, 0, byref(location))
