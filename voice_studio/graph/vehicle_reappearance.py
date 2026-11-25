"""Conservative cross-shot links for vehicles with measured body color."""

from collections import defaultdict

from voice_studio.vision.vehicle_evidence import VEHICLE_LABELS, hue_distance

MAX_HUE_DISTANCE = .13
MAX_WINDOW_MS = 30_000
MAX_CHAINS = 144
PAINT_BASIS = "measured-vehicle-body"


def body_witness(rows):
    """Find a body-color measurement supported by two separate source frames."""
    witnessed = []
    for node in rows:
        body = node["data"].get("vehicleBodyColor")
        if body and node["label"].casefold() in VEHICLE_LABELS:
            score = .45 * body["coverage"] + .55 * body["saturation"] - .4 * body["background"]
            witnessed.append((score, node, body["hue"]))
    witnessed.sort(key=lambda row: (-row[0], row[1]["startMs"], row[1]["id"]))
    for score, node, hue in witnessed:
        for _, other, other_hue in witnessed:
            if node["startMs"] != other["startMs"] and hue_distance(hue, other_hue) <= .045:
                return {"score": round(score, 3), "node": node, "hue": hue, "second": other}
    return None


def visible_separately(first, second):
    """Require independent same-frame vehicle boxes, excluding nested detections."""
    x, y, width, height = first["data"]["box"]
    u, v, w, h = second["data"]["box"]
    common = max(0, min(x + width, u + w) - max(x, u)) * max(0, min(y + height, v + h) - max(y, v))
    return common < .60 * min(width * height, w * h)


def vehicle_chains(groups, samples, intervals, links):
    """Collect current profile chains and their usable source-crop evidence."""
    chains, co_visible = {}, defaultdict(list)
    followers = set(links.values())
    for head in range(len(groups)):
        if head in followers:
            continue
        parts = [head]
        while parts[-1] in links:
            parts.append(links[parts[-1]])
        rows, shots = [], set()
        for group in parts:
            shot, tracks, _ = groups[group]
            shots.add(shot)
            for track in tracks:
                rows.extend(node for node in samples[track["id"]]
                            if node["label"].casefold() in VEHICLE_LABELS
                            and not node["data"].get("vehicleIdentityAmbiguous"))
        if not rows:
            continue
        witness = body_witness(rows)
        if witness:
            chains[head] = {"tail": parts[-1], "shots": shots, "partCount": len(parts),
                            "start": intervals[head][0], "end": intervals[parts[-1]][1],
                            "witness": witness}
        for node in rows:
            co_visible[node["startMs"]].append((head, node))
    return chains, co_visible


def unambiguous_anchor(key, profile, co_visible):
    """Require two co-visible sightings with no separately visible same-color rival."""
    measurement = profile["witness"]
    for node in (measurement["node"], measurement["second"]):
        peers = [peer for other, peer in co_visible[node["startMs"]]
                 if other != key and visible_separately(node, peer)]
        if not peers:
            return False
        for peer in peers:
            color = peer["data"].get("vehicleBodyColor")
            if color and hue_distance(measurement["hue"], color["hue"]) <= MAX_HUE_DISTANCE:
                return False
    return True


def compatible(first, second):
    """Require different shots, disjoint visibility and a consistent hue."""
    if first["shots"] & second["shots"]:
        return False
    earlier, later = sorted((first, second), key=lambda profile: profile["start"])
    return (0 <= later["start"] - earlier["end"] <= MAX_WINDOW_MS
            and hue_distance(first["witness"]["hue"], second["witness"]["hue"]) <= MAX_HUE_DISTANCE)


def add_vehicle_links(groups, samples, intervals, links, diagnostics, *, max_cluster):
    """Attach only links with one visual anchor and no plausible same-shot rival."""
    chains, co_visible = vehicle_chains(groups, samples, intervals, links)
    if len(chains) > MAX_CHAINS:
        return
    anchors = {key for key, profile in chains.items() if unambiguous_anchor(key, profile, co_visible)}
    candidates = defaultdict(list)
    for key, profile in chains.items():
        if key in anchors:
            continue
        choices = [anchor for anchor in sorted(anchors) if compatible(profile, chains[anchor])]
        if len(choices) == 1:
            candidates[choices[0]].append(key)
    for anchor, choices in candidates.items():
        distinct = [key for key in choices
                    if all(key == other or not chains[key]["shots"] & chains[other]["shots"]
                           for other in choices)]
        cluster = sorted([anchor, *distinct], key=lambda key: chains[key]["start"])
        if (sum(chains[key]["partCount"] for key in cluster) > max_cluster
                or any(not compatible(chains[first], chains[second])
                       for index, first in enumerate(cluster) for second in cluster[index + 1:])):
            continue
        for first, second in zip(cluster, cluster[1:]):
            left, right = chains[first], chains[second]
            source, target = left["witness"], right["witness"]
            links[left["tail"]] = second
            diagnostics["acceptedLinks"].append({
                "fromGroup": left["tail"], "toGroup": second, "basis": PAINT_BASIS,
                "fromObservationId": source["node"]["id"], "toObservationId": target["node"]["id"],
                "score": min(source["score"], target["score"]),
                "hueDistance": round(hue_distance(source["hue"], target["hue"]), 4),
                "anchorObservationId": chains[anchor]["witness"]["node"]["id"],
            })
