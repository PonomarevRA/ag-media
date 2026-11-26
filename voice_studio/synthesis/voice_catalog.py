"""Summarize saved voice evidence without loading audio or running models."""

import json
from statistics import median

from voice_studio.synthesis.accent_profiles import saved_accent


def profile_name(value):
    """Validate an optional catalog label. An empty label uses the version number."""
    if value is None:
        return ""
    if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Введите имя голосового профиля длиной до 80 символов.")
    name = value.strip()
    if len(name) > 80:
        raise ValueError("Введите имя голосового профиля длиной до 80 символов.")
    return name


def voice_catalog(profiles, recordings):
    """Project profile rows into a catalog shared by synthesis, dubbing and MCP."""
    available = {row["id"] for row in recordings if not row["evaluation_only"]}
    result = []
    for profile in profiles:
        refs = json.loads(profile["refs"])
        primary = refs[0] if refs and refs[0] in available else None
        report = json.loads(profile["reference_report"]) if profile["reference_report"] else None
        metrics = summarize_review(report, refs) if report else None
        entry = {
            "version": profile["version"], "language": profile["language"], "name": profile["name"],
            "created": profile["created"], "recordingCount": len(refs), "previewId": primary,
            "synthesisReady": bool(primary and all(ref in available for ref in refs)),
            "dubbingReady": bool(primary and metrics),
            "metrics": metrics,
        }
        if style := saved_accent(profile["reference_report"]):
            entry["accentStyle"] = style
        result.append(entry)
    return result


def summarize_review(report, refs):
    """Summarize measured pairs and active speech from samples retained in this version."""
    retained = set(refs)
    takes = [take for take in report["takes"] if take["id"] in retained]
    scores = [pair["cosine"] for pair in report["pairs"] if pair["left"] in retained and pair["right"] in retained]
    primary = next((take for take in takes if refs and take["id"] == refs[0]), None)
    return {
        "metric": report["metric"], "encoderFingerprint": report.get("encoderFingerprint"),
        "medianCosine": round(median(scores), 3) if scores else None,
        "minimumCosine": min(scores) if scores else None, "pairCount": len(scores),
        "activeSeconds": round(sum(take["activeMs"] for take in takes) / 1000, 2),
        "excludedCount": len(report["takes"]) - len(takes),
        "primarySpeechDbfs": primary["speechDbfs"] if primary else None,
    }
