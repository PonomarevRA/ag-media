"""Reduce screenshot OCR coverage without hiding failed crops behind successful control-label reads."""


def ocr_coverage(statuses, *, has_lines=False):
    """Prefer each group's final fallback, then preserve incomplete but useful observations as partial."""
    groups = []
    for prefix in ("", "controls-"):
        apple = statuses.get(prefix + "apple-vision")
        fallback = statuses.get(prefix + "tesseract")
        if prefix and apple is None and fallback is None:
            continue  # No control-label crops were requested.
        final = fallback if fallback is not None else apple
        if final == "ok":
            groups.append("ok")
        elif final == "partial" or apple == "partial":
            groups.append("partial")
        else:
            groups.append("unavailable")
    if groups and all(value == "ok" for value in groups):
        return "ok"
    if has_lines or any(value in {"ok", "partial"} for value in groups):
        return "partial"
    return "unavailable"
