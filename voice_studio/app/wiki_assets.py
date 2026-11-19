"""Resolve only versioned, generated documentation assets, never workspace data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "wiki" / "site"


def wiki_asset(url_path, locale="ru"):
    """Return an allowlisted (path, MIME) pair; malformed or missing paths stay 404."""
    if url_path in {"/wiki", "/wiki/"}:
        name = "Home.html" if locale == "en" else "Home-ru.html"
    elif url_path.startswith("/wiki/"):
        name = url_path[len("/wiki/"):]
    else:
        return None
    manifest = SITE / "manifest.json"
    if not manifest.is_file():
        return None
    allowed = json.loads(manifest.read_text(encoding="utf-8"))
    mime = allowed.get(name)
    if not isinstance(mime, str):
        return None
    path = SITE / name
    # The manifest cannot authorize a link into media, model or temporary folders.
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(SITE.resolve()):
        return None
    return path, mime
