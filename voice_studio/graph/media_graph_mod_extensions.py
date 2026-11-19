"""Validate and project bounded user-defined graph facts without changing native nodes."""

from collections import Counter
from copy import deepcopy
import json
from math import isfinite
import re

from voice_studio.mods import ModError
from voice_studio.mods.schema import identifier

MAX_CONTRIBUTIONS = 64
MAX_STAGES = 2048
MAX_NODES = 2048
MAX_EDGES = 8192
MAX_NODES_PER_STAGE = 256
MAX_EDGES_PER_STAGE = 1024
MAX_EXTENSION_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 2 * MAX_EXTENSION_BYTES
MAX_DATA_BYTES = 16384
LOCAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
TYPE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")


def _name(value, pattern, label):
    """Accept a bounded literal identifier; identifiers never contain graph separators."""
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ModError(f"Invalid graph extension {label}")
    return value


def _json_value(value, depth=0):
    """Reject nonfinite, non-JSON and deeply nested custom attributes before persistence."""
    if depth > 8:
        raise ModError("Graph extension data nesting exceeds 8 levels")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and isfinite(value):
        return
    if isinstance(value, str) and len(value) <= 4096:
        return
    if isinstance(value, dict) and len(value) <= 128:
        for key, item in value.items():
            if not isinstance(key, str) or not 1 <= len(key) <= 128:
                raise ModError("Graph extension data requires bounded string keys")
            _json_value(item, depth + 1)
        return
    if isinstance(value, list) and len(value) <= 1024:
        for item in value:
            _json_value(item, depth + 1)
        return
    raise ModError("Graph extension data must be bounded finite JSON")


def _encoded(value, limit, label):
    """Measure finite JSON bytes and translate serialization failures to a public mod error."""
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise ModError(f"Invalid graph extension {label}: {error}") from error
    if len(payload) > limit:
        raise ModError(f"Graph extension {label} exceeds {limit} bytes")
    return len(payload)


def _interval(row, duration):
    """Require paired half-open source-clock milliseconds; timeless concepts omit both bounds."""
    if "startMs" not in row and "endMs" not in row:
        return None, None
    start, end = row.get("startMs"), row.get("endMs")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= duration:
        raise ModError("Graph extension interval must satisfy 0 <= startMs < endMs <= durationMs")
    return start, end


def _provenance(report, mod_id, capability):
    """Retain attributable captured package identity without copying settings into every node."""
    result = {"modId": mod_id, "capability": capability, "timeBasis": "source-ms",
              "contract": {"id": "media.graph-extension", "apiVersion": 1}}
    records = report.get("modProvenance", [])
    if not isinstance(records, list):
        return result
    for row in records:
        if isinstance(row, dict) and row.get("modId") == mod_id:
            for key in ("version", "fingerprint", "settingsHash", "origin"):
                if isinstance(row.get(key), str) and len(row[key]) <= 160:
                    result[key] = row[key]
            break
    return result


def _reference(value, local_ids, native_ids):
    """Resolve a same-contribution local ID or an explicit existing native node reference."""
    if isinstance(value, str) and value in local_ids:
        return local_ids[value]
    if isinstance(value, dict) and set(value) == {"nativeId"}:
        native_id = value["nativeId"]
        if isinstance(native_id, str) and native_id in native_ids:
            return native_id
    raise ModError("Graph extension edge references an unknown local or explicit native node")


def _contribution(stage, extension, report, duration, native_ids):
    """Validate one complete stage contribution, returning detached nodes and deduplicated edges."""
    mod_id = identifier(stage.get("modId"), "graph extension mod id")
    capability = identifier(stage.get("capability"), "graph extension capability")
    if stage.get("state") not in ("done", "reused"):
        raise ModError("Only completed or reused mod stages can contribute graph data")
    if (not isinstance(extension, dict) or set(extension) != {"apiVersion", "nodes", "edges"}
            or type(extension["apiVersion"]) is not int or extension["apiVersion"] != 1):
        raise ModError("Graph extension requires apiVersion 1, nodes and edges")
    rows, relations = extension["nodes"], extension["edges"]
    if not isinstance(rows, list) or len(rows) > MAX_NODES_PER_STAGE:
        raise ModError("Graph extension stage exceeds its node budget")
    if not isinstance(relations, list) or len(relations) > MAX_EDGES_PER_STAGE:
        raise ModError("Graph extension stage exceeds its edge budget")
    size = _encoded(extension, MAX_EXTENSION_BYTES, "contribution")
    prefix = f"extension:{mod_id}:{capability}:"
    provenance = _provenance(report, mod_id, capability)
    nodes, local_ids = [], {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) - {"id", "type", "label", "startMs", "endMs", "data"}
                or not {"id", "type", "label"} <= set(row)):
            raise ModError("Invalid graph extension node fields")
        local_id = _name(row["id"], LOCAL_ID, "node id")
        type_name = _name(row["type"], TYPE_NAME, "type")
        if local_id in local_ids:
            raise ModError("Duplicate graph extension local node id")
        label = row["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > 1024 or "\0" in label:
            raise ModError("Graph extension label must contain 1–1024 characters")
        properties = row.get("data", {})
        if not isinstance(properties, dict):
            raise ModError("Graph extension node data must be an object")
        _json_value(properties)
        _encoded(properties, MAX_DATA_BYTES, "node data")
        start, end = _interval(row, duration)
        key = prefix + local_id
        local_ids[local_id] = key
        nodes.append({"id": key, "kind": "extension", "label": label, "startMs": start, "endMs": end,
                      "data": {"extensionType": type_name, "description": type_name, "localId": local_id,
                               "properties": deepcopy(properties), "modProvenance": dict(provenance)}})
    edges = set()
    for row in relations:
        if not isinstance(row, dict) or set(row) != {"source", "relation", "target"}:
            raise ModError("Invalid graph extension edge fields")
        relation = _name(row["relation"], LOCAL_ID, "relation")
        source = _reference(row["source"], local_ids, native_ids)
        target = _reference(row["target"], local_ids, native_ids)
        if source in native_ids and target in native_ids:
            raise ModError("A graph extension cannot add native-to-native edges")
        edge = source, prefix + relation, target
        if edge in edges:
            raise ModError("Duplicate graph extension edge")
        edges.add(edge)
    return (mod_id, capability), nodes, edges, size


def _identity(stage):
    """Return a diagnostic-safe namespace even when imported stage metadata is malformed."""
    result = []
    for key in ("modId", "capability"):
        try:
            result.append(identifier(stage.get(key)))
        except ModError:
            result.append("unknown")
    return tuple(result)


def _rejected(stage, error):
    """Expose one explicit failed namespace without copying its raw payload into diagnostics."""
    mod_id, capability = _identity(stage)
    return {"modId": mod_id, "capability": capability, "state": "rejected",
            "nodeCount": 0, "edgeCount": 0, "error": str(error)[:500]}


def add_mod_extensions(graph, report):
    """Append valid namespaces and report rejected contributions without losing native work.

    Each namespace validates completely before any of its writes. Invalid extension
    facts produce a persisted state/error entry; native nodes, other namespaces and
    the caller's raw modStages remain intact. Repeated valid projection is idempotent.
    """
    summary = {"apiVersion": 1, "nodeCount": 0, "edgeCount": 0, "rejectedCount": 0, "contributors": []}
    stages = report.get("modStages", [])
    if not isinstance(stages, list) or len(stages) > MAX_STAGES:
        summary.update(rejectedCount=1, contributors=[_rejected({}, "modStages must be an array of at most 2048 stages")])
        return summary
    candidates = []
    for stage in stages:
        if isinstance(stage, dict) and isinstance(stage.get("result"), dict) and "graphExtension" in stage["result"]:
            candidates.append(stage)
    counts = Counter(_identity(stage) for stage in candidates)
    native_ids = {key for key, node in graph.nodes.items() if node["kind"] != "extension"}
    nodes, edges, total_bytes = {}, set(), 0
    for index, stage in enumerate(candidates):
        if index >= MAX_CONTRIBUTIONS:
            summary["contributors"].append(_rejected(stage, "Graph extension contributor budget exceeded; remaining contributions omitted"))
            summary["rejectedCount"] += len(candidates) - index
            summary["truncated"] = True
            break
        try:
            if counts[_identity(stage)] != 1:
                raise ModError("Duplicate graph extension mod/capability contribution")
            namespace, declared_nodes, declared_edges, size = _contribution(
                stage, stage["result"]["graphExtension"], report, graph.duration, native_ids)
            if (len(nodes) + len(declared_nodes) > MAX_NODES or len(edges) + len(declared_edges) > MAX_EDGES
                    or total_bytes + size > MAX_TOTAL_BYTES):
                raise ModError("Graph extension total budget exceeded")
            for node in declared_nodes:
                existing = graph.nodes.get(node["id"])
                if existing is not None and existing != node:
                    raise ModError("Graph extension node conflicts with an existing graph node")
        except ModError as error:
            summary["contributors"].append(_rejected(stage, error))
            summary["rejectedCount"] += 1
            continue
        for node in declared_nodes:
            nodes[node["id"]] = node
        edges.update(declared_edges)
        total_bytes += size
        summary["contributors"].append({"modId": namespace[0], "capability": namespace[1], "state": "ready",
                                         "nodeCount": len(declared_nodes), "edgeCount": len(declared_edges)})
    graph.nodes.update(nodes)
    graph.edges.update(edges)
    summary.update(nodeCount=len(nodes), edgeCount=len(edges))
    return summary


def _unencodable_diagnostic(value):
    """Return bounded diagnostic text only when a raw value cannot be stored as finite JSON."""
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return None
    except (ValueError, TypeError, RecursionError) as error:
        reason = type(error).__name__
    try:
        raw = json.dumps(value, ensure_ascii=True, allow_nan=True,
                         default=lambda item: "<non-JSON " + type(item).__name__ + ">")
    except (ValueError, TypeError, RecursionError):
        raw = "<unserializable graph extension; inspect original analysis report>"
    return {"encoding": "raw-json-text", "serializationError": reason,
            "raw": raw[:MAX_DATA_BYTES], "truncated": len(raw) > MAX_DATA_BYTES}


def snapshot_mod_provenance(provenance):
    """Keep rejected non-JSON graph payloads diagnosable without breaking native persistence.

    Finite raw extension payloads stay unchanged. A payload containing NaN, invalid
    Unicode or non-JSON values becomes a bounded diagnostic text in the snapshot's
    copied provenance. Oversized stage lists retain a bounded prefix and an explicit
    omitted-remainder diagnostic. The caller's raw report is never modified.
    """
    stages = provenance.get("modStages") if isinstance(provenance, dict) else None
    if not isinstance(stages, list):
        diagnostic = _unencodable_diagnostic(stages)
        if diagnostic is None:
            return provenance
        result = dict(provenance)
        result["modStages"] = {"rejectedDiagnostic": diagnostic}
        return result
    result = provenance
    if len(stages) > MAX_STAGES:
        # Do not scan or retain a rejected oversized remainder: it may itself contain
        # unencodable data. Keep its count and the location of the untouched raw report.
        result = dict(provenance)
        result["modStages"] = stages[:MAX_STAGES]
        result["modStagesDiagnostic"] = {"encoding": "omitted-remainder", "limit": MAX_STAGES,
                                         "omittedCount": len(stages) - MAX_STAGES,
                                         "rawAvailableIn": "original-analysis-report"}
    for index, stage in enumerate(stages[:MAX_STAGES]):
        if not isinstance(stage, dict) or not isinstance(stage.get("result"), dict) or "graphExtension" not in stage["result"]:
            continue
        value = stage["result"]["graphExtension"]
        diagnostic = _unencodable_diagnostic(value)
        if diagnostic is None:
            continue
        if result is provenance:
            result = dict(provenance)
            result["modStages"] = list(stages)
        copied_stage = dict(stage)
        copied_stage["result"] = dict(stage["result"])
        copied_stage["result"]["graphExtension"] = {"rejectedDiagnostic": diagnostic}
        result["modStages"][index] = copied_stage
    return result
