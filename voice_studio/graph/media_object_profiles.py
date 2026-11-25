"""Describe shot-local objects using detector labels and text on their surface."""

from collections import defaultdict
from itertools import combinations, product

from voice_studio.graph.media_graph_objects import box_coverage, center, overlap, representative_sample
from voice_studio.graph.object_reappearance import reappearance_chains
from voice_studio.graph.object_reconciliation import color_conflict, reconcile_object_groups, source_gallery
from voice_studio.vision.scene_sampling import SMALL_OBJECT_DETAIL

MAX_LABEL_GAP_MS = 600
TRANSPORT_LABELS = {"car", "truck", "bus", "airplane", "boat", "train"}


def label_match(left, right, *, simultaneous):
    """Require nearly identical boxes over time, with a same-frame containment rule."""
    if left["label"].casefold() == right["label"].casefold():
        return False
    a, b = left["data"], right["data"]
    if "interface-layout" in {a.get("source"), b.get("source")}:
        return False
    first, second = a["box"], b["box"]
    iou = overlap(first, second)
    if iou >= .35 and color_conflict([left], [right]):
        return False
    if not simultaneous:
        return iou >= .72
    scores = (a.get("score"), b.get("score"))
    if any(type(score) not in (int, float) or not 0 <= score <= 1 for score in scores):
        return False
    if abs(scores[0] - scores[1]) < .10:
        return False
    if iou >= .72:
        return True
    # A detector can outline the tabletop and the whole table in the same frame.
    # Require containment, comparable size on one axis and alignment on both.
    if iou < .35:
        return False
    wide, tall = max(first[2], second[2]), max(first[3], second[3])
    if max(min(first[2], second[2]) / wide, min(first[3], second[3]) / tall) < .85:
        return False
    area_a, area_b = first[2] * first[3], second[2] * second[3]
    intersection = iou * (area_a + area_b) / (1 + iou)
    x, y = center(first)
    u, v = center(second)
    return intersection >= .90 * min(area_a, area_b) and abs(x - u) <= .15 * wide and abs(y - v) <= .25 * tall


def appearance_match(left, right):
    """Allow a shifted cross-label box only with strong interior and edge evidence."""
    if left["label"].casefold() == right["label"].casefold():
        return False
    a, b = left["data"], right["data"]
    if overlap(a["box"], b["box"]) < .45:
        return False
    witness = observation_key(left)
    for match in b.get("objectMatches") or []:
        if witness != match_key(match) or not .65 <= match.get("scaleRatio", 0) <= 1.54:
            continue
        inside = match.get("objectSimilarity") or 0
        edges = match.get("edgeSimilarity") or 0
        context = match.get("backgroundSimilarity") or 0
        if inside >= .94 and edges >= .75:
            return True
        if inside >= .92 and edges >= .80 and context >= .80:
            return True
    return False


def nested_transport(left, right):
    """Detect a competing transport label whose smaller box repeatedly lies inside the larger one."""
    labels = {left["label"].casefold(), right["label"].casefold()}
    if len(labels) != 2 or not labels <= TRANSPORT_LABELS:
        return False
    boxes = (left["data"]["box"], right["data"]["box"])
    inner, outer = sorted(boxes, key=lambda box: box[2] * box[3])
    ratio = inner[2] * inner[3] / (outer[2] * outer[3])
    u, v = center(inner)
    x, y = center(outer)
    return (.18 <= ratio <= .65 and box_coverage(inner, outer) >= .9
            and abs(x - u) <= .27 * outer[2] and abs(y - v) <= .38 * outer[3]
            and not color_conflict([left], [right]))


def match_candidates(tracks):
    """Find unique track partners without comparing unrelated frames or shots."""
    shots = defaultdict(lambda: defaultdict(list))
    for key, shot, samples in tracks:
        for sample in samples:
            shots[shot][sample["startMs"]].append((key, sample))
    partners = defaultdict(set)
    visual_partners = defaultdict(set)
    nested_support = defaultdict(set)
    for frames in shots.values():
        previous_stamp, previous = -MAX_LABEL_GAP_MS - 1, []
        for stamp, current in sorted(frames.items()):
            comparisons = [(combinations(current, 2), True)]
            if stamp - previous_stamp <= MAX_LABEL_GAP_MS:
                comparisons.append((product(previous, current), False))
            for pairs, simultaneous in comparisons:
                for (key, left), (other, right) in pairs:
                    if key == other:
                        continue
                    if label_match(left, right, simultaneous=simultaneous):
                        partners[key].add(other)
                        partners[other].add(key)
                    elif simultaneous and nested_transport(left, right):
                        nested_support[tuple(sorted((key, other)))].add(stamp)
                    if not simultaneous and appearance_match(left, right):
                        visual_partners[key].add(other)
                        visual_partners[other].add(key)
            previous_stamp, previous = stamp, current
    for (key, other), stamps in nested_support.items():
        if len(stamps) >= 3:
            partners[key].add(other)
            partners[other].add(key)
    # Supplemental matches must never displace a partner already backed by geometry.
    geometric_tracks = set(partners)
    for key, others in visual_partners.items():
        if key not in geometric_tracks:
            partners[key].update(others - geometric_tracks)
    return partners, visual_partners, geometric_tracks


def object_groups(nodes, tracks, partners, visual_partners, geometric_tracks):
    """Accept compatible multi-label groups while preserving ambiguous one-to-many tracks."""
    unseen = {key for key, _, _ in tracks}
    for key, shot, _ in tracks:
        if key not in unseen:
            continue
        pending, component = [key], set()
        while pending:
            member = pending.pop()
            if member in unseen:
                unseen.remove(member)
                component.add(member)
                pending.extend(partners[member] & unseen)
        members = sorted((nodes[member] for member in component),
                         key=lambda node: (node["startMs"], node["id"]))
        compatible = all(component - {member} <= partners[member] for member in component)
        # A missing edge may also be safe when one exclusive track hands off to the next.
        exclusive_chain = all(left["endMs"] <= right["startMs"] and right["id"] in partners[left["id"]]
                              for left, right in zip(members, members[1:]))
        if compatible or exclusive_chain:
            basis = "single-track"
            if len(members) > 1:
                basis = ("same-shot-geometry" if component <= geometric_tracks
                         else "same-shot-appearance")
            yield shot, members, basis
        else:
            available = set(component)
            for member in members:
                key = member["id"]
                if key not in available:
                    continue
                available.remove(key)
                # A unique reciprocal visual witness can resolve part of an ambiguous geometry group.
                possibilities = visual_partners[key]
                if len(possibilities) == 1:
                    other, = possibilities
                    if other in available and visual_partners[other] == {key}:
                        available.remove(other)
                        yield shot, [member, nodes[other]], "same-shot-appearance"
                        continue
                yield shot, [member], "single-track"


def identity_tracks(nodes, tracks):
    """Use only baseline geometry, labels and times when associating existing object identities."""
    identity_nodes, identity_rows = {}, []
    for key, shot, samples in tracks:
        baseline = [sample for sample in samples if sample["data"].get("detailBasis") != SMALL_OBJECT_DETAIL]
        source = nodes[key]
        if baseline and len(baseline) < len(samples):
            source = {**source, "startMs": baseline[0]["startMs"], "endMs": baseline[-1]["endMs"]}
        identity_nodes[key] = source
        identity_rows.append((key, shot, baseline))
    return identity_nodes, identity_rows


def add_object_profiles(graph, tracks, *, object_reconciliation=False):
    """Publish object identities with measured appearances and unchanged original tracks.

    The optional pilot preserves legacy reappearance chains unless measured
    time/color conflicts veto them, then permits conservative source-part
    extensions and publishes a real-crop gallery. Default calls keep existing behavior.
    """
    identity_nodes, identity_rows = identity_tracks(graph.nodes, tracks)
    partners, visual_partners, geometric_tracks = match_candidates(identity_rows)
    identity_samples = {key: samples for key, _, samples in identity_rows}
    samples_by_track = {key: samples for key, _, samples in tracks}
    groups = list(object_groups(identity_nodes, identity_rows, partners, visual_partners, geometric_tracks))
    if object_reconciliation:
        links, reconciliation = reconcile_object_groups(
            groups, identity_samples, baseline_links=reappearance_chains(groups, identity_samples))
    else:
        links = reappearance_chains(groups, identity_samples)
    following = set(links.values())
    profiles = []
    for head, (shot, members, basis) in enumerate(groups):
        if head in following:
            continue
        parts = [groups[head]]
        tail = head
        while tail in links:
            tail = links[tail]
            parts.append(groups[tail])
        if len(parts) > 1:
            members = [node for _, part, _ in parts for node in part]
            basis = ("reappearance-measured-local-parts" if object_reconciliation
                     else "reappearance-foreground-and-geometry")
        # The ID stays tied to the earliest track; bounded temporal evidence can stabilize its label.
        first = min(members, key=lambda node: (node["startMs"], node["id"]))
        profile_id = "object_profile:" + first["id"].split(":", 1)[1]
        # Additional measurements can extend visibility without redefining the identity.
        parts = [(area, [graph.nodes[node["id"]] for node in part], reason) for area, part, reason in parts]
        members = [node for _, part, _ in parts for node in part]
        labels = []
        for node in members:
            facts = node["data"]
            baseline = identity_samples[node["id"]]
            sample = (representative_sample(baseline) if baseline and len(baseline) < facts["sampleCount"]
                      else facts["representative"])
            labels.append({"label": node["label"], "trackId": node["id"], "sampleCount": facts["sampleCount"],
                           "observationId": sample["observationId"], "timeMs": sample["timeMs"],
                           "score": sample["score"]})
        labels.sort(key=label_order)
        shots = list(dict.fromkeys(part[0] for part in parts))
        data = {"labels": labels, "shotId": shot, "shotIds": shots, "basis": basis,
                "appearances": appearance_intervals(parts, samples_by_track)}
        samples = [sample for node in members for sample in samples_by_track[node["id"]]]
        identity_evidence = [sample for node in members for sample in identity_samples[node["id"]]]
        if object_reconciliation:
            member_ids = {sample["id"] for sample in identity_evidence}
            data["objectReconciliation"] = {
                **{key: value for key, value in reconciliation.items() if key != "acceptedLinks"},
                "acceptedLinks": [link for link in reconciliation["acceptedLinks"]
                                  if link["fromObservationId"] in member_ids
                                  and link["toObservationId"] in member_ids],
            }
            if not data["objectReconciliation"]["acceptedLinks"] and len(parts) > 1:
                data["basis"] = "reappearance-preserved-baseline"
            if any(link["basis"] == "measured-vehicle-body"
                   for link in data["objectReconciliation"]["acceptedLinks"]):
                data["basis"] = "reappearance-measured-vehicle-body"
            witnesses = []
            for link in data["objectReconciliation"]["acceptedLinks"]:
                witnesses.extend((link["fromObservationId"], link["toObservationId"]))
            data["objectGallery"] = source_gallery(identity_evidence or samples, witnesses)
        screens = [sample for sample in (identity_evidence or samples) if sample["data"].get("interfaceEvidence")]
        types = {sample["data"]["interfaceEvidence"]["type"] for sample in screens}
        if len(types) == 1:
            support = max(screens, key=lambda sample: (len(sample["data"]["interfaceEvidence"].get("signals", ())),
                                                       sample["data"].get("score") or 0))
            data["interfaceEvidence"] = {**support["data"]["interfaceEvidence"], "timeMs": support["startMs"],
                                         "observationId": support["id"]}
        if appearance := profile_appearance(samples):
            data["appearance"] = appearance
        start = min(node["startMs"] for node in members)
        end = max(node["endMs"] for node in members)
        data["labelEvidence"] = stable_label_evidence(labels, identity_evidence or samples, data.get("interfaceEvidence"))
        name = data.get("interfaceEvidence", {}).get("displayLabel", data["labelEvidence"]["selectedLabel"])
        data["identityId"] = profile_id
        data["displayLabel"] = name
        data["hasLabelConflict"] = len({label["label"].casefold() for label in labels}) > 1
        profile = graph.node(profile_id, "object_profile", name, (start, end), data)
        for area in shots:
            graph.connect(profile_id, "in_shot", area)
        for member in members:
            member["data"]["profileId"] = profile_id
            graph.connect(member["id"], "part_of_profile", profile_id)
            for sample in samples_by_track[member["id"]]:
                sample["data"]["identityId"] = profile_id
        profiles.append(profile)
    attach_object_text(graph, tracks)
    return profiles



def measured_label_support(samples):
    """Count unique source timestamps per label and retain each label's best raw witness."""
    timestamps = defaultdict(set)
    best = {}
    for sample in samples:
        label = sample["label"]
        timestamps[label].add(sample["startMs"])
        candidate = {"label": label, "observationId": sample["id"], "timeMs": sample["startMs"],
                     "score": sample["data"].get("score"), "trackId": ""}
        previous = best.get(label)
        if previous is None or label_order(candidate) < label_order(previous):
            best[label] = candidate
    counts = {label: len(stamps) for label, stamps in timestamps.items()}
    return counts, best


def temporal_label_candidate(peak, counts, best):
    """Return a unique near-tie leader with at least three timestamps and twice peak support."""
    if peak["score"] is None:
        return peak
    contenders = []
    for label, count in counts.items():
        candidate = best[label]
        score = candidate["score"]
        if (label != peak["label"] and count >= 3 and count >= 2 * counts.get(peak["label"], 0)
                and score is not None and 0 <= peak["score"] - score <= .10 + 1e-9):
            contenders.append(candidate)
    if not contenders:
        return peak
    largest = max(counts[item["label"]] for item in contenders)
    winners = [item for item in contenders if counts[item["label"]] == largest]
    return winners[0] if len(winners) == 1 else peak


def interface_supports_peak(interface, peak, selected):
    """Protect a measured UI's compatible peak device class from non-device temporal votes."""
    if not interface or interface.get("basis") not in {"layout-and-text", "verified-panel", "desktop-controls"}:
        return False
    devices = {"mobile-ui": {"cell phone", "mobile phone", "phone", "smartphone"},
               "ui": {"tv", "tv screen", "monitor", "screen"}}
    compatible = devices.get(interface.get("type"), set())
    return peak["label"].casefold() in compatible and selected["label"].casefold() not in compatible


def stable_label_evidence(labels, samples, interface=None):
    """Publish a bounded temporal label choice, its own score and the unchanged raw peak.

    Each label gets one vote per source timestamp, regardless of crop/tile count.
    Votes describe temporal support, never an estimated class probability. Only
    near-tie, unique temporal leaders may replace the peak; measured UI evidence
    protects a compatible peak device class from non-device votes.
    """
    peak = labels[0]
    counts, best = measured_label_support(samples)
    selected = temporal_label_candidate(peak, counts, best)
    basis = "strongest-observation" if selected is peak else "near-tie-temporal-support"
    if selected is not peak and interface_supports_peak(interface, peak, selected):
        selected = peak
        basis = "interface-supported-strongest-observation"
    support = [{"label": label, "uniqueTimestampCount": count, "maxScore": best[label]["score"]}
               for label, count in sorted(counts.items())]
    fields = ("label", "observationId", "timeMs", "score")
    return {"basis": basis, "selectedLabel": selected["label"], "selectedLabelScore": selected["score"],
            "selectedObservation": {key: selected[key] for key in fields},
            "peakRawObservation": {key: peak[key] for key in fields}, "support": support,
            "supportMeaning": "unique-source-timestamps-not-probability"}



def appearance_intervals(parts, samples_by_track):
    """Publish one measured interval and representative source frame per appearance."""
    intervals = []
    for shot, nodes, _ in parts:
        ordered = sorted(nodes, key=lambda node: (node["startMs"], node["endMs"], node["id"]))
        for node in ordered:
            start, end = node["startMs"], node["endMs"]
            observations = samples_by_track[node["id"]]
            if intervals and intervals[-1][0] == shot and start <= intervals[-1][2]:
                _, first, last, samples = intervals[-1]
                intervals[-1] = (shot, first, max(last, end), samples + observations)
            else:
                intervals.append((shot, start, end, observations))
    appearances = []
    for shot, start, end, samples in intervals:
        ordered = sorted(samples, key=lambda node: (node["startMs"], node["id"]))
        representative = representative_sample(ordered)
        representative["label"] = next(node["label"] for node in ordered if node["id"] == representative["observationId"])
        appearances.append({"shotId": shot, "startMs": start, "endMs": end,
                            "representativeMs": representative["timeMs"], "representative": representative,
                            "sampleCount": len({node["startMs"] for node in ordered})})
    return appearances



def observation_key(node):
    """Identify one detector observation within its source frame."""
    return node["startMs"], node["label"], tuple(node["data"]["box"])


def match_key(match):
    """Resolve a scene worker reference without depending on detector order."""
    return match["timeMs"], match["label"], tuple(match["box"])


def profile_appearance(samples):
    """Keep a few real views of one profile and link the scores to their observations."""
    measured = [node for node in samples if node["data"].get("objectPalette")]
    if not measured:
        return {}
    ordered = sorted(measured, key=lambda node: (-(node["data"].get("score") or 0),
                                                node["startMs"], node["id"]))
    primary = ordered[0]
    by_key = {observation_key(node): node for node in samples}
    matches = []
    for node in samples:
        for evidence in node["data"].get("objectMatches") or []:
            previous = by_key.get(match_key(evidence))
            if previous is None or previous["id"] == node["id"]:
                continue
            matches.append({
                "fromObservationId": previous["id"], "toObservationId": node["id"],
                "fromTimeMs": previous["startMs"], "toTimeMs": node["startMs"],
                "fromLabel": previous["label"], "toLabel": node["label"],
                **{key: evidence.get(key) for key in ("objectSimilarity", "edgeSimilarity",
                                                     "backgroundSimilarity", "scaleRatio")},
            })
    matches.sort(key=lambda row: (row["fromLabel"] == row["toLabel"],
                                  row["fromTimeMs"] == row["toTimeMs"],
                                  -(row["objectSimilarity"] if row["objectSimilarity"] is not None else -1),
                                  -(row["backgroundSimilarity"] if row["backgroundSimilarity"] is not None else -1),
                                  row["fromObservationId"], row["toObservationId"]))
    views = source_views(ordered, matches)
    return {"basis": "normalized-object-crops", "palette": primary["data"]["objectPalette"],
            "paletteObservationId": primary["id"], "views": [profile_view(view) for view in views],
            "matches": matches[:6]}


def source_views(ordered, matches):
    """Include the primary sample, the comparison witnesses and distinct measured frames."""
    primary = ordered[0]
    views = [primary]
    if matches:
        pair = matches[0]
        by_id = {node["id"]: node for node in ordered}
        for key in (pair["fromObservationId"], pair["toObservationId"]):
            if key != primary["id"] and key in by_id:
                views.append(by_id[key])
    for node in ordered[1:]:
        if len(views) == 3:
            break
        if all(abs(view["startMs"] - node["startMs"]) >= 200 for view in views):
            views.append(node)
    return views


def profile_view(node):
    """Expose the original crop and its source time, box, label, and measured palette."""
    data = node["data"]
    return {"observationId": node["id"], "timeMs": node["startMs"], "label": node["label"],
            "box": data["box"], "score": data.get("score"), "palette": data["objectPalette"],
            **({"interfaceEvidence": data["interfaceEvidence"]} if "interfaceEvidence" in data else {})}


def label_order(entry):
    """Prefer a measured higher score, breaking ties by source time and then stable ID."""
    score = entry["score"]
    return (score is None, -(score or 0), entry["timeMs"], entry["trackId"])


def covered_fraction(inner, outer):
    """Measure how much of the inner source-frame box lies inside the outer box."""
    x, y, width, height = inner
    left, top, wide, tall = outer
    if width <= 0 or height <= 0:
        return 0
    overlap_x = max(0, min(x + width, left + wide) - max(x, left))
    overlap_y = max(0, min(y + height, top + tall) - max(y, top))
    return overlap_x * overlap_y / (width * height)


def text_owner(text_box, objects):
    """Choose a unique containing profile, preferring a clearly nested smaller object."""
    candidates = {}
    for profile_id, box in objects:
        if covered_fraction(text_box, box) < .90:
            continue
        area = box[2] * box[3]
        if profile_id not in candidates or area < candidates[profile_id][0]:
            candidates[profile_id] = (area, box)
    if not candidates:
        return None
    profile_id = min(candidates, key=lambda key: candidates[key][0])
    area, box = candidates[profile_id]
    for other_id, (other_area, other_box) in candidates.items():
        if other_id != profile_id and (area > .5 * other_area or covered_fraction(box, other_box) < .90):
            return None
    return profile_id


def attach_object_text(graph, tracks):
    """Store distinct OCR readings and link them to exact evidence from the same frame."""
    if not tracks:
        return
    objects_by_frame = defaultdict(list)
    for key, _, observations in tracks:
        profile_id = graph.nodes[key]["data"]["profileId"]
        for observation in observations:
            objects_by_frame[observation["startMs"]].append((profile_id, observation["data"]["box"]))
    readings = defaultdict(dict)
    for node in graph.nodes.values():
        if node["kind"] != "text_observation" or node["startMs"] not in objects_by_frame:
            continue
        stamp = node["startMs"]
        profile_id = text_owner(node["data"]["box"], objects_by_frame[stamp])
        if profile_id is None:
            continue
        text_key = " ".join(node["label"].casefold().split())
        previous = readings[profile_id].get(text_key)
        if previous is None or (stamp, node["id"]) < (previous["timeMs"], previous["observationId"]):
            readings[profile_id][text_key] = {"text": node["label"], "timeMs": stamp, "observationId": node["id"]}
    for profile_id, entries in readings.items():
        visible_text = sorted(entries.values(), key=lambda entry: (entry["timeMs"], entry["observationId"]))
        graph.nodes[profile_id]["data"]["visibleText"] = visible_text
        for entry in visible_text:
            graph.connect(profile_id, "has_visible_text", entry["observationId"])
