"""Cache the optional final identity review independently from scene detection."""

from copy import deepcopy
from voice_studio.core.media_event_bus import run_enrichment
from voice_studio.core.analysis_stages import stage


def review_objects(source, report, cache, *, emit):
    """Review existing scene observations; missing scene data stays explicitly pending.

    Only native crop computation uses the current job bus. Checkpoint selection,
    manifest updates and final report mutations remain on the calling thread.
    """
    scene = report.get("sceneEvidence")
    scene_state = (report.get("analysisStages", {}).get("scene") or {}).get("state")
    if not isinstance(scene, dict) or scene_state in ("pending", "invalidated", "failed"):
        cache.selection.mark("object_review", "pending")
        return report
    raw = deepcopy(scene)
    # Derived pair evidence must never become its own next-run input.
    for frame in raw.get("frames", []):
        for item in frame.get("objects", []):
            item.pop("objectPartMatches", None)

    def compute():
        """Run bounded measured-crop matching, without restarting a scene worker."""
        from voice_studio.vision.object_reconciliation import prepare_object_reconciliation
        with stage(emit, "object-review") as progress:
            reviewed = run_enrichment(lambda: prepare_object_reconciliation(source, {"sceneEvidence": raw}))
            status = (reviewed.get("objectReconciliationEvidence") or {}).get("status")
            if status not in ("measured", "no-evidence"):
                progress["state"] = "failed"
        return {key: reviewed[key] for key in ("sceneEvidence", "objectReconciliationEvidence") if key in reviewed}

    result = cache.run("object_review", {}, raw, compute,
                       lambda value: isinstance(value.get("sceneEvidence"), dict)
                       and (value.get("objectReconciliationEvidence") or {}).get("status") in ("measured", "no-evidence"), emit=emit)
    report.update(result)
    status = (result.get("objectReconciliationEvidence") or {}).get("status")
    if status in ("measured", "no-evidence"):
        report["objectReconciliationEnabled"] = True
        from voice_studio.graph.media_graph_model import object_layers
        report.update(object_layers(report))
    return report
