"""Build a frame-local containment tree while preserving the identity and timing graphs."""

from collections import defaultdict

from voice_studio.graph.media_graph_objects import DISPLAY_LABELS, box_coverage

LAYOUT_KINDS = frozenset({"object_observation", "object_alias", "screen_region", "ui_element", "ui_connection"})
DISPLAY_HOSTS = DISPLAY_LABELS | {"cell phone", "laptop", "mobile-ui"}
MIN_CONTAINMENT = .9
MIN_AREA_RATIO = 1.15
PARENT_AMBIGUITY = 1.1


def box_area(node):
    """Read the normalized area of one measured node."""
    width, height = node["data"]["box"][2:]
    return width * height


def choose_owner(child, candidates):
    """Pick the tightest measured container when the next alternative is clearly larger."""
    size = box_area(child)
    box = child["data"]["box"]
    covering = [node for node in candidates if box_area(node) >= MIN_AREA_RATIO * size
                and box_coverage(box, node["data"]["box"]) >= MIN_CONTAINMENT]
    covering.sort(key=box_area)
    if covering and (len(covering) == 1 or box_area(covering[1]) >= PARENT_AMBIGUITY * box_area(covering[0])):
        return covering[0]["id"]
    return None


def display_host(region, objects):
    """Associate a verified layout with its matching display observation, even for equal boxes."""
    data = region["data"]
    if data.get("panelType") in ("mobile-ui", "ui"):
        peers = [item for item in objects if item["label"] == data["panelType"]
                 and item["data"].get("source") == "interface-layout"
                 and item["data"]["box"] == data["box"]]
        if len(peers) == 1:
            return peers[0]["id"]
    return choose_owner(region, (item for item in objects if item["label"].casefold() in DISPLAY_HOSTS))


def add_scene_hierarchy(graph):
    """Attach at most one contained_in parent per visible node on the same measured frame."""
    frames = defaultdict(list)
    for node in graph.nodes.values():
        if node["kind"] in LAYOUT_KINDS:
            frames[node["startMs"]].append(node)
    aliases = {source: target for source, relation, target in graph.edges
               if relation in {"alias_of", "part_of_observation"}}
    for items in frames.values():
        objects = [node for node in items if node["kind"] == "object_observation" and node["id"] not in aliases]
        regions = [node for node in items if node["kind"] == "screen_region"]
        ui = [node for node in items if node["kind"] == "ui_element"]
        region_hosts = {region["id"]: display_host(region, objects) for region in regions}
        equivalent = {host for region, host in region_hosts.items() if host
                      and graph.nodes[region]["data"]["box"] == graph.nodes[host]["data"]["box"]}
        object_hosts = [node for node in objects if node["id"] not in equivalent] + regions
        for child in items:
            key, kind, data = child["id"], child["kind"], child["data"]
            parent = aliases.get(key)
            if parent is None and kind == "screen_region":
                parent = region_hosts[key]
            elif parent is None and kind == "ui_element":
                parent = choose_owner(child, (item for item in ui if item["data"]["screenId"] == data["screenId"]))
                parent = parent or data["screenId"]
            elif parent is None and kind == "ui_connection":
                parent = data["screenId"]
            elif parent is None and kind == "object_observation":
                # A mobile panel already hosts its matching region, so it stays above it in the tree.
                hosts = objects if child["label"] == "mobile-ui" else object_hosts
                parent = choose_owner(child, hosts)
            if parent and graph.nodes[parent]["startMs"] == child["startMs"]:
                graph.connect(key, "contained_in", parent)
