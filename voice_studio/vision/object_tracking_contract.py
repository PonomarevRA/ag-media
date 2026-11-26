"""Wire contract for detector proposals retained only as existing track evidence."""

from voice_studio.vision.vehicle_evidence import valid_box

WEAK_CONTINUATION = "weak-track-continuation"
MAX_TRACKING_GAP_MS = 550
TRACKING_PROPOSAL_SCORE = .24
STRONG_TRACKING_SCORE = .50
TRACKING_BASES = frozenset({"box-iou", "background-lk", "background-dis", "native-edge-view"})
TRACKING_EXCLUDED_LABELS = frozenset({"person", "ui", "mobile-ui", "tv", "tv screen", "monitor", "screen"})


def tracking_witness(item, stamp):
    """Validate weak provenance and its preceding source-clock anchor."""
    witness, score, label = item.get("trackingWitness"), item.get("score"), item.get("label")
    if (item.get("detailBasis") != WEAK_CONTINUATION or not isinstance(witness, dict)
            or not isinstance(label, str) or label.casefold() in TRACKING_EXCLUDED_LABELS
            or type(score) not in (float, int) or not TRACKING_PROPOSAL_SCORE <= score <= 1):
        return None
    start, box, basis = witness.get("timeMs"), witness.get("box"), witness.get("basis")
    if (type(start) is int and start >= 0 and 0 < stamp - start <= MAX_TRACKING_GAP_MS
            and valid_box(box) and isinstance(basis, str) and basis in TRACKING_BASES
            and (score < STRONG_TRACKING_SCORE or basis == "native-edge-view")):
        return {"timeMs": start, "box": [round(value, 4) for value in box], "basis": basis}
    return None
