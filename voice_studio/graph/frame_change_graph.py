"""Expose geometric change witnesses as first-class source-clock graph nodes."""

def change_rows(scene):
    """Read one short-video scan or the list of already aligned partition witnesses."""
    layer = scene.get("frameChanges") or {}
    rows = layer.get("rows") if isinstance(layer, dict) else None
    return rows if isinstance(rows, list) else []


def shift_change_rows(scene, start, end):
    """Move a window's measurements to source time without altering cached inputs."""
    for row in change_rows(scene):
        left = start + row["startMs"]
        right = min(end, start + row["endMs"])
        if start <= left < right:
            yield {**row, "startMs": left, "endMs": right}


def change_coverage(scene):
    """Summarize scan and supplemental detector work without copying the witnesses."""
    report = scene.get("frameChanges")
    if not isinstance(report, dict):
        return {}
    allowed = ("status", "mode", "basis", "framesCompared", "globalFrames", "omittedFrames", "cutCandidatesRemoved", "elapsedMs")
    fields = {key: report[key] for key in allowed if type(report.get(key)) in (str, int)}
    if isinstance(objects := scene.get("frameChangeObjects", report.get("objectChecks")), dict):
        fields["objectChecks"] = {key: objects[key] for key in ("status", "frames", "added", "modelRuns", "elapsedMs", "skippedFrames")
                                  if type(objects.get(key)) in (str, int)}
    return fields


def valid_region(region, box_coordinates):
    """Accept only a normalized box with a positive count measured on the analysis raster."""
    if not isinstance(region, dict) or not (box := box_coordinates(region.get("box"))):
        return None
    count = region.get("changedPixels")
    if type(count) is not int or not 0 < count <= 320 * 320:
        return None
    return {"box": box, "changedPixels": count, "basis": "tile-difference"}


def add_frame_changes(graph, scene, track, box_coordinates):
    """Add timed region witnesses with links to their measured frame, shot and track."""
    for row in change_rows(scene):
        if not isinstance(row, dict):
            continue
        start, end = row.get("startMs"), row.get("endMs")
        regions = row.get("regions")
        if (type(start) is not int or type(end) is not int or not 0 <= start < min(end, graph.duration)
                or not isinstance(regions, list)):
            continue
        end = min(end, graph.duration)
        shots = list(graph.intersecting_shots(start, end))
        if len(shots) > 1:
            continue
        for index, region in enumerate(regions[:4]):
            evidence = valid_region(region, box_coordinates)
            if evidence is None:
                continue
            node = f"frame_change:{start}:{index}"
            graph.node(node, "frame_change", graph.names["frame_change"], (start, end), evidence)
            graph.connect(node, "at_frame", graph.frame(start, "tile-difference"))
            if track:
                graph.connect(node, "on_track", track)
            if shots:
                graph.connect(node, "in_shot", shots[0])
