"""Project sampled visual screen evidence onto the source-clock media graph."""

from math import isfinite

from voice_studio.graph.media_graph_objects import overlap

INTERFACE_KINDS = ("screen_region", "ui_element", "ui_connection")
REGION_BASES = {"device-crop", "frame-layout"}
PANEL_TYPES = ("mobile-ui", "ui")
MAX_REGIONS = 2
MAX_ELEMENTS = 48
MAX_CONNECTIONS = 24
NATIVE_SMALL_CONTROLS = {
    "native-small-control-outline": "button_candidate",
    "native-small-control-glyph": "icon_candidate",
}
MAX_NATIVE_DIMENSION = 16_384
MAX_STATE_SUPPORT_MS = 12_001
STATE_EVIDENCE_TIMES = ("recognizedAtMs", "verifiedFromMs", "verifiedUntilMs")


def sequence(value, limit):
    """Take the bounded prefix of an optional report collection."""
    return value[:limit] if isinstance(value, (list, tuple)) else ()


def text_field(value, limit):
    """Return source text without coercing imported non-text objects."""
    return value.strip()[:limit] if isinstance(value, str) else ""


def within(inner, outer):
    """Allow rounding at a sampled screen boundary without accepting outside boxes."""
    x, y, width, height = inner
    left, top, span_x, span_y = outer
    return (left - .002 <= x and top - .002 <= y
            and x + width <= left + span_x + .002 and y + height <= top + span_y + .002)


def line_points(value, screen):
    """Accept one measured two-point segment in normalized source coordinates."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    points = []
    for point in value:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or not all(type(axis) in (int, float) and isfinite(axis) and 0 <= axis <= 1 for axis in point)):
            return None
        x, y = (round(axis, 4) for axis in point)
        if not within((x, y, 0, 0), screen):
            return None
        points.append([x, y])
    return points if points[0] != points[1] else None


def segment_box(points):
    """Provide a tight nonzero highlight even for a purely horizontal or vertical line."""
    left, right = sorted(point[0] for point in points)
    top, bottom = sorted(point[1] for point in points)
    if right - left < .002:
        left, right = max(0, left - .002), min(1, right + .002)
    if bottom - top < .002:
        top, bottom = max(0, top - .002), min(1, bottom + .002)
    return [round(value, 4) for value in (left, top, right - left, bottom - top)]


def clean_native_control(row, role, basis):
    """Preserve exact pixel geometry only for the native small-control detector."""
    if NATIVE_SMALL_CONTROLS.get(basis) != role:
        return {}
    pixel, confidence = row.get("pixelBox"), row.get("confidence")
    if (not isinstance(pixel, (list, tuple)) or len(pixel) != 4
            or any(type(value) is not int or value < 0 for value in pixel)
            or type(confidence) not in (int, float) or not isfinite(confidence) or not 0 <= confidence <= 1):
        return {}
    x, y, width, height = pixel
    if min(width, height) <= 0 or max(x + width, y + height) > MAX_NATIVE_DIMENSION:
        return {}
    return {"pixelBox": list(pixel), "confidence": confidence}


def clean_state_evidence(value, box_coordinates, screen, element, stamp, interval):
    """Retain bounded support metadata only on the exact frame that supplied the OCR."""
    if not isinstance(value, dict) or interval is None:
        return None
    recognized, start, end = (value.get(key) for key in STATE_EVIDENCE_TIMES)
    source, count, kind = (value.get(key) for key in ("sourceFrame", "supportCount", "kind"))
    if any(type(item) is not int for item in (recognized, start, end, source, count)):
        return None
    lower, upper = interval
    if (recognized != stamp or not lower <= start <= recognized < end <= upper
            or end - start > MAX_STATE_SUPPORT_MS or not 0 <= source < 2 ** 31
            or not 1 <= count <= end - start):
        return None
    if kind == "changed":
        if count != 1 or start != recognized or end != recognized + 1:
            return None
    elif kind != "stable" or count < 2:
        return None
    crop = box_coordinates(value.get("cropBox"))
    if not crop or min(crop[2:]) <= 0 or not within(crop, screen):
        return None
    x, y, w, h = element
    if not within((x + w / 2, y + h / 2, 0, 0), crop):
        return None
    evidence = {"recognizedAtMs": recognized, "verifiedFromMs": start, "verifiedUntilMs": end,
                "sourceFrame": source, "supportCount": count, "kind": kind, "cropBox": crop}
    if previous := clean_same_content(value.get("sameContentAs"), box_coordinates, screen, lower, recognized):
        evidence["sameContentAs"] = previous
    return evidence


def clean_same_content(value, box_coordinates, screen, shot_start, stamp):
    """Reference only an earlier observed crop in the same shot and viewport."""
    if not isinstance(value, dict):
        return None
    recognized, source = value.get("recognizedAtMs"), value.get("sourceFrame")
    if (type(recognized) is not int or not shot_start <= recognized < stamp
            or type(source) is not int or not 0 <= source < 2 ** 31):
        return None
    crop = box_coordinates(value.get("box"))
    if not crop or min(crop[2:]) <= 0 or not within(crop, screen):
        return None
    return {"recognizedAtMs": recognized, "sourceFrame": source, "box": crop}


def clean_region(row, box_coordinates, *, stamp=None, interval=None):
    """Normalize one screen and remap connector indices after discarding malformed elements."""
    if not isinstance(row, dict):
        return None
    basis = row.get("basis")
    box = box_coordinates(row.get("box"))
    if not isinstance(basis, str) or basis not in REGION_BASES or not box or min(box[2:]) <= 0:
        return None

    verified = basis == "frame-layout" and row.get("panelType") in PANEL_TYPES
    elements, index_map = [], {}
    for original, element in enumerate(sequence(row.get("elements"), MAX_ELEMENTS)):
        if not isinstance(element, dict):
            continue
        role = text_field(element.get("role"), 48)
        element_basis = text_field(element.get("basis"), 48)
        bounds = box_coordinates(element.get("box"))
        if not role or not element_basis or not bounds or min(bounds[2:]) <= 0 or not within(bounds, box):
            continue
        index_map[original] = len(elements)
        data = {"role": role, "box": bounds, "basis": element_basis}
        if label := text_field(element.get("text"), 180):
            data["text"] = label
            if verified and (evidence := clean_state_evidence(
                    element.get("stateEvidence"), box_coordinates, box, bounds, stamp, interval)):
                data["stateEvidence"] = evidence
        if verified:
            data.update(clean_native_control(element, role, element_basis))
        for field in ("column", "row"):
            if type(element.get(field)) is int and 1 <= element[field] <= 4096:
                data[field] = element[field]
        if mosaic := clean_mosaic(element.get("mosaic"), box_coordinates):
            data["mosaic"] = mosaic
        elements.append(data)
    if not elements and not verified:
        return None

    connections = []
    seen = set()
    for connection in sequence(row.get("connections"), MAX_CONNECTIONS):
        if not isinstance(connection, dict):
            continue
        first, second = connection.get("from"), connection.get("to")
        if (type(first) is not int or type(second) is not int or first == second
                or first not in index_map or second not in index_map):
            continue
        pair = tuple(sorted((index_map[first], index_map[second])))
        if pair in seen or not (points := line_points(connection.get("segment"), box)):
            continue
        seen.add(pair)
        connections.append({"from": index_map[first], "to": index_map[second], "segment": points})

    region = {"box": box, "basis": basis, "cropAvailable": True}
    if row.get("regionKind") in ("panel", "card-candidate", "diagram-candidate"):
        region["regionKind"] = row["regionKind"]
    for field in ("lastObservedMs", "previousStateMs"):
        if type(row.get(field)) is int:
            region[field] = row[field]
    if row.get("detailBasis") == "original-resolution":
        region["detailBasis"] = row["detailBasis"]
    if verified:
        region["panelType"] = row["panelType"]
        if row.get("changeAssisted") is True:
            region["changeAssisted"] = True
        if type(reading := row.get("readingTimeMs")) is int and reading >= 0:
            region["readingTimeMs"] = reading
    if device := text_field(row.get("deviceLabel"), 48):
        region["deviceLabel"] = device
    return region, elements, connections


def clean_mosaic(value, box_coordinates):
    """Retain a bounded address on the stitched PNG in addition to the frame box."""
    if not isinstance(value, dict) or not (box := box_coordinates(value.get("box"))):
        return None
    limits = {"pageStartMs": (0, 7_200_000), "pageIndex": (0, 15), "column": (1, 12), "row": (1, 4096)}
    fields = {}
    for name, (low, high) in limits.items():
        item = value.get(name)
        if type(item) is not int or not low <= item <= high:
            return None
        fields[name] = item
    pixel = value.get("pixelBox")
    if not isinstance(pixel, list) or len(pixel) != 4 or any(type(item) is not int or not 0 <= item <= 4096 for item in pixel):
        return None
    return {**fields, "box": box, "pixelBox": pixel}


def add_interface_observations(graph, stamp, rows, box_coordinates):
    """Attach visible screen regions, their elements and their connectors to a sampled frame."""
    candidates = sequence(rows, MAX_REGIONS)
    if not candidates:
        return
    sampled_shot = next(graph.intersecting_shots(stamp, stamp + 1), None)
    interval = (0, graph.duration)
    if sampled_shot is not None:
        shot = graph.nodes[sampled_shot]
        interval = (shot["startMs"], shot["endMs"])
    for region_index, candidate in enumerate(candidates):
        normalized = clean_region(candidate, box_coordinates, stamp=stamp, interval=interval)
        if normalized is None:
            continue
        region, elements, connections = normalized
        end = candidate.get("screenStateEndMs")
        if region.get("panelType") and type(end) is int and stamp < end <= graph.duration:
            region["screenStateEndMs"] = end
            state_region = clean_region({**candidate, "connections": (),
                                         "elements": candidate.get("screenStateElements")}, box_coordinates,
                                        stamp=stamp, interval=interval)
            region["screenStateElements"] = state_region[1] if state_region else []
        reading = region.get("readingTimeMs")
        if reading is not None and not 0 <= reading < stamp:
            region.pop("readingTimeMs")
            reading = None
        previous = region.get("previousStateMs")
        if previous is not None and not 0 <= previous < stamp:
            region.pop("previousStateMs")
        last = region.get("lastObservedMs")
        if last is not None and not stamp <= last < graph.duration:
            region.pop("lastObservedMs")
        ending = stamp + 1
        suffix = f"{stamp}:{region_index}"
        region_id = f"screen_region:{suffix}"
        frame_id = graph.frame(stamp, "interface-layout")
        title = "Screen region" if graph.language == "en" else "Область интерфейса"
        graph.node(region_id, "screen_region", title, (stamp, ending), region)
        graph.connect(region_id, "at_frame", frame_id)
        if reading is not None:
            graph.connect(region_id, "reading_from_frame", graph.frame(reading, "interface-layout"))
        graph.link_shots(region_id, stamp, ending)
        for index, element in enumerate(elements):
            element_id = f"ui_element:{suffix}:{index}"
            role = element["role"].replace("_", " ").replace("-", " ")
            label = element.get("text", role)
            data = {**element, "screenId": region_id, "description": role, "cropAvailable": True}
            add_screen_child(graph, element_id, "ui_element", label, stamp, data, region_id, frame_id)
        for index, connection in enumerate(connections):
            connection_id = f"ui_connection:{suffix}:{index}"
            title = "Diagram connection" if graph.language == "en" else "Соединение схемы"
            data = {**connection, "screenId": region_id, "box": segment_box(connection["segment"])}
            add_screen_child(graph, connection_id, "ui_connection", title, stamp, data, region_id, frame_id)
            graph.connect(connection_id, "endpoint_a", f"ui_element:{suffix}:{connection['from']}")
            graph.connect(connection_id, "endpoint_b", f"ui_element:{suffix}:{connection['to']}")


def add_screen_child(graph, node_id, kind, label, stamp, data, region_id, frame_id):
    """Preserve the enclosing screen and source time for one indexed observation."""
    graph.node(node_id, kind, label, (stamp, stamp + 1), data)
    graph.connect(node_id, "in_screen", region_id)
    graph.connect(node_id, "at_frame", frame_id)
    graph.link_shots(node_id, stamp, stamp + 1)


def restore_interface_observation(frame, regions, node_id, kind, data):
    """Rebuild the source report contract from a published graph's ordered scene rows."""
    if kind == "screen_region":
        region = {field: data[field] for field in (
            "box", "basis", "deviceLabel", "panelType", "changeAssisted", "readingTimeMs",
            "regionKind", "detailBasis", "previousStateMs", "lastObservedMs", "screenStateEndMs", "screenStateElements")
                  if field in data}
        region.update(elements=[], connections=[])
        frame.setdefault("interfaces", []).append(region)
        regions[node_id] = region
    elif region := regions.get(data.get("screenId")):
        if kind == "ui_element":
            fields = ("role", "box", "basis", "text", "mosaic", "row", "column",
                      "pixelBox", "confidence", "stateEvidence")
            region["elements"].append({field: data[field] for field in fields if field in data})
        else:
            region["connections"].append({field: data[field] for field in ("from", "to", "segment")})


def restore_verified_panel(frame, box, panel_type="mobile-ui"):
    """Preserve the layout when reusing snapshots that stored only its object observation."""
    regions = frame.setdefault("interfaces", [])
    for region in regions:
        if region["basis"] == "frame-layout" and overlap(box, region["box"]) >= .95:
            region["panelType"] = panel_type
            return
    regions.insert(0, {"box": box, "basis": "frame-layout", "panelType": panel_type,
                       "elements": [], "connections": []})
