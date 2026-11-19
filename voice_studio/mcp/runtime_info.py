"""Describe the MCP source baseline so clients can detect a long-lived process after updates."""

from hashlib import sha256
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def source_revision():
    """Fingerprint service/graph source and the naming contract, without media or model weights."""
    digest = sha256()
    files = sorted([*(PACKAGE / "mcp").glob("*.py"), *(PACKAGE / "graph").glob("*.py"), PACKAGE / "core/tool_names.py"])
    for path in files:
        digest.update(path.relative_to(PACKAGE).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


STARTUP_SOURCE_REVISION = source_revision()


def runtime_info():
    """Report the startup baseline separately from disk; lazy imports are not a loaded-code attestation."""
    try:
        disk = source_revision()
    except OSError:
        disk = None
    return {"contractVersion": "ag-media-mcp/3", "sourceScope": "mcp-graph-and-tool-names-python",
            "startupSourceRevision": STARTUP_SOURCE_REVISION, "diskSourceRevision": disk,
            "restartRequired": disk != STARTUP_SOURCE_REVISION if disk is not None else None,
            "snapshotReadsPinned": True}
