"""Shared source limits; individual native inference windows remain bounded separately."""

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_MS = 2 * 60 * 60 * 1000
MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024
MAX_TRANSCRIPT_CUES = 20000
MAX_RAW_SPEAKER = 100_000_000
