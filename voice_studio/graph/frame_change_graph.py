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
            regions = []
            for region in row["regions"]:
                reading = region.get("recognition")
                if (isinstance(reading, dict) and type(reading.get("sampleTimeMs")) is int
                        and type(reading.get("verifiedUntilMs")) is int):
                    reading = {**reading, "sampleTimeMs": start + reading["sampleTimeMs"],
                               "verifiedUntilMs": min(end, start + reading["verifiedUntilMs"])}
                    region = {**region, "recognition": reading}
                regions.append(region)
            yield {**row, "startMs": left, "endMs": right, "regions": regions}


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
    if isinstance(recognition := report.get("recognition"), dict):
        fields["recognition"] = {key: recognition[key] for key in ("candidatesChecked", "reviewedRegions", "ocrCrops", "detectorRuns", "elapsedMs")
                                 if type(recognition.get(key)) is int}
    return fields


def recognition_rows(value, label_key, box_coordinates):
    """Keep bounded readings with exact-frame normalized boxes from a reviewed crop."""
    rows = []
    if not isinstance(value, list):
        return rows
    for item in value[:6 if label_key == "text" else 3]:
        if not isinstance(item, dict) or not (box := box_coordinates(item.get("box"))):
            continue
        label = item.get(label_key)
        if not isinstance(label, str) or not label.strip():
            continue
        row = {label_key: label.strip()[:160], "box": box}
        if label_key == "label":
            score = item.get("score")
            if type(score) not in (int, float) or not 0 <= score <= 1:
                continue
            row["score"] = score
        rows.append(row)
    return rows


def valid_region(region, box_coordinates):
    """Accept only a normalized box with a positive count measured on the analysis raster."""
    if not isinstance(region, dict) or not (box := box_coordinates(region.get("box"))):
        return None
    count = region.get("changedPixels")
    if type(count) is not int or not 0 < count <= 320 * 320:
        return None
    result = {"box": box, "changedPixels": count, "basis": "tile-difference"}
    if isinstance(reading := region.get("recognition"), dict):
        objects = recognition_rows(reading.get("objects"), "label", box_coordinates)
        text = recognition_rows(reading.get("text"), "text", box_coordinates)
        result["recognition"] = {"status": "recognized" if objects or text else "unclassified",
                                 "objects": objects, "text": text}
    return result


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
            sample_frame = None
            if recognition := evidence.get("recognition"):
                raw = region["recognition"]
                sample, verified = raw.get("sampleTimeMs"), raw.get("verifiedUntilMs")
                if type(sample) is int and type(verified) is int and start <= sample < verified <= graph.duration:
                    recognition.update(sampleTimeMs=sample, verifiedUntilMs=verified)
                    sample_frame = graph.frame(sample, "frame-change-crop")
            graph.node(node, "frame_change", graph.names["frame_change"], (start, end), evidence)
            if sample_frame:
                graph.connect(node, "recognition_from_frame", sample_frame)
            graph.connect(node, "at_frame", graph.frame(start, "tile-difference"))
            if track:
                graph.connect(node, "on_track", track)
            if shots:
                graph.connect(node, "in_shot", shots[0])
