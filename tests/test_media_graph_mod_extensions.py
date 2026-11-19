"""Verify arbitrary mod graph types, strict boundaries and persisted generic navigation."""

from copy import deepcopy
import json
from math import isnan
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest import IsolatedAsyncioTestCase, TestCase

from voice_studio.graph.media_graph_mod_extensions import add_mod_extensions, MAX_NODES_PER_STAGE
from voice_studio.graph.media_graph_model import MediaGraph, build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, identify
from voice_studio.mods import ModRegistry

ROOT = Path(__file__).resolve().parents[1]


def extension():
    """Return an unknown domain type with a local relation and an explicit native source link."""
    return {"apiVersion": 1, "nodes": [
        {"id": "event-1", "type": "acoustic.whistle", "label": "Quantum signal", "startMs": 100, "endMs": 500,
         "data": {"score": .9, "units": "probability", "nested": {"custom": True}}},
        {"id": "concept", "type": "new-domain-concept", "label": "Signal category", "data": {"arbitrary": [1, False]}}],
        "edges": [{"source": "event-1", "relation": "classified-as", "target": "concept"},
                  {"source": "event-1", "relation": "derived-from", "target": {"nativeId": "utterance:100:0"}}]}


def report(value=None):
    """Wrap one completed stage alongside a native source-clock utterance and captured mod identity."""
    return {"language": "en", "text": "Native words", "durationMs": 1000,
            "segments": [{"startMs": 100, "endMs": 700, "text": "Native words"}],
            "modStages": [{"modId": "test.graph", "capability": "analysis.graph-fixture", "state": "done",
                           "result": {"graphExtension": extension() if value is None else value}}],
            "modProvenance": [{"modId": "test.graph", "version": "1.0.0", "fingerprint": "a" * 64,
                               "settingsHash": "b" * 64, "origin": "user"}]}


def real_stage_snapshot(folder, value=None, publication=None):
    """Execute an enabled user stage in its actual copied-code subprocess and publish its graph."""
    root = Path(folder)
    registry = ModRegistry(root / "mods")
    package = registry.directory / "test.graph"
    package.mkdir(parents=True)
    declared = {"apiVersion": 1, "id": "test.graph", "name": "New graph type", "version": "1.0.0",
                "capabilities": [{"id": "analysis.graph-fixture", "kind": "stage", "entrypoint": "worker.py:run"}],
                "license": "Test fixture", "provenance": "Local graph integration fixture"}
    (package / "mod.json").write_text(json.dumps(declared))
    (package / "worker.py").write_text("def run(payload, settings):\n    return " + repr({"graphExtension": extension() if value is None else value}) + "\n")
    row = registry.catalog()["mods"][0]
    registry.enable(row["id"], row["fingerprint"])
    snapshot = registry.snapshot()
    analyzed = report()
    analyzed["modStages"][0]["result"] = snapshot.invoke_mod("test.graph", "analysis.graph-fixture", {"report": analyzed})
    analyzed["modProvenance"] = snapshot.provenance()
    graph = build_graph(analyzed, kind="audio", duration_ms=1000)
    source = root / "source.wav"
    source.write_bytes(b"local graph source fixture")
    store = MediaGraphStore(root / "graphs")
    def publish(progress=None):
        """Publish only after all native and custom graph data is ready on the source clock."""
        return store.publish(identify(source), {"kind": "audio", "durationMs": 1000}, graph, "en",
                             {"modStages": analyzed["modStages"], "modProvenance": analyzed["modProvenance"], "progress": progress})
    published = publication.publish("complete", publish) if publication else publish()
    return store.root, published


class MediaGraphModExtensionTests(TestCase):
    """Use graph fixtures only; no inference runtimes, downloads or existing app state are needed."""

    def assert_rejected(self, value, message=None):
        """Require an explicit rejected contribution while native work and raw input remain intact."""
        analyzed = report(value)
        native = build_graph({key: value for key, value in analyzed.items() if key != "modStages"}, kind="audio", duration_ms=1000)
        built = build_graph(analyzed, kind="audio", duration_ms=1000)
        self.assertEqual(built["nodes"], native["nodes"])
        self.assertEqual(built["edges"], native["edges"])
        self.assertEqual(built["modExtensions"]["rejectedCount"], 1)
        self.assertEqual(built["modExtensions"]["contributors"][0]["state"], "rejected")
        if message:
            self.assertIn(message, built["modExtensions"]["contributors"][0]["error"])
        return built

    def test_unknown_types_preserve_native_nodes_and_are_idempotent(self):
        """Keep arbitrary domain metadata, stable identities and native observations unchanged."""
        analyzed = report()
        before = deepcopy(analyzed)
        built = build_graph(analyzed, kind="audio", duration_ms=1000)
        repeated = build_graph(analyzed, kind="audio", duration_ms=1000)
        self.assertEqual(built, repeated)
        self.assertEqual(analyzed, before)
        nodes = {node["id"]: node for node in built["nodes"]}
        key = "extension:test.graph:analysis.graph-fixture:event-1"
        self.assertEqual(nodes[key]["kind"], "extension")
        self.assertEqual(nodes[key]["data"]["extensionType"], "acoustic.whistle")
        self.assertEqual(nodes[key]["data"]["properties"], extension()["nodes"][0]["data"])
        self.assertEqual(nodes[key]["data"]["modProvenance"]["fingerprint"], "a" * 64)
        self.assertEqual(nodes[key]["data"]["modProvenance"]["contract"], {"id": "media.graph-extension", "apiVersion": 1})
        self.assertEqual(nodes["utterance:100:0"]["label"], "Native words")
        self.assertIn((key, "extension:test.graph:analysis.graph-fixture:derived-from", "utterance:100:0"), built["edges"])
        graph = MediaGraph(1000, "en")
        graph.node("utterance:100:0", "utterance", "Native words", (100, 700))
        add_mod_extensions(graph, analyzed)
        first = deepcopy(graph.nodes), set(graph.edges)
        add_mod_extensions(graph, analyzed)
        self.assertEqual((graph.nodes, graph.edges), first)

    def test_real_stage_persists_search_timeline_links_and_readiness_after_reopen(self):
        """Publish actual subprocess output and recover custom facts through generic indexed reads."""
        with TemporaryDirectory() as folder:
            directory, published = real_stage_snapshot(folder)
            store = MediaGraphStore(directory)
            identity = published["snapshotId"]
            metadata = store.overview(identity)
            self.assertEqual(metadata["nodesByCategory"]["extension"], 2)
            self.assertEqual(metadata["modExtensions"]["nodeCount"], 2)
            self.assertEqual(metadata["modExtensions"]["contributors"][0]["modId"], "test.graph")
            found = store.search(identity, "Quantum signal", "extension")["nodes"]
            self.assertEqual(len(found), 1)
            self.assertEqual(store.search(identity, "acoustic whistle", "extension")["nodes"][0]["id"], found[0]["id"])
            self.assertEqual(store.timeline(identity, 0, 800, "extension")["nodes"], found)
            detail = store.node(identity, found[0]["id"])
            self.assertEqual(detail["data"]["properties"]["units"], "probability")
            self.assertTrue(any(row["id"] == "utterance:100:0" for row in detail["neighbors"]))

    def test_rejected_real_stage_persists_diagnostics_without_native_loss(self):
        """Retain the original report and native snapshot when actual user graph output is invalid."""
        value = extension()
        value["edges"][0]["target"] = {"nativeId": "missing"}
        with TemporaryDirectory() as folder:
            directory, published = real_stage_snapshot(folder, value)
            store = MediaGraphStore(directory)
            metadata = store.overview(published["snapshotId"])
            self.assertEqual(metadata["modExtensions"]["rejectedCount"], 1)
            self.assertEqual(metadata["nodesByCategory"]["extension"], 0)
            self.assertEqual(metadata["modExtensions"]["contributors"][0]["state"], "rejected")
            self.assertEqual(metadata["provenance"]["modStages"][0]["result"]["graphExtension"], value)
            self.assertEqual(store.search(published["snapshotId"], "Native words", "utterance")["nodes"][0]["id"], "utterance:100:0")

    def test_nonfinite_rejected_raw_graph_has_diagnostic_text_and_native_snapshot(self):
        """Persist invalid imported extension diagnostics without relaxing the store's finite JSON policy."""
        analyzed = report()
        analyzed["modStages"][0]["result"]["graphExtension"]["nodes"][0]["data"]["score"] = float("nan")
        graph = build_graph(analyzed, kind="audio", duration_ms=1000)
        self.assertEqual(graph["modExtensions"]["rejectedCount"], 1)
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(b"native fixture")
            store = MediaGraphStore(Path(folder) / "graphs")
            published = store.publish(identify(source), {"kind": "audio", "durationMs": 1000}, graph, "en",
                                      {"modStages": analyzed["modStages"]})
            diagnostic = published["provenance"]["modStages"][0]["result"]["graphExtension"]["rejectedDiagnostic"]
            self.assertIn("NaN", diagnostic["raw"])
            self.assertEqual(diagnostic["encoding"], "raw-json-text")
            self.assertEqual(published["nodesByKind"]["utterance"], 1)
            self.assertNotIn("rejectedDiagnostic", analyzed["modStages"][0]["result"]["graphExtension"])

    def test_deep_and_invalid_unicode_raw_payloads_cannot_break_native_persistence(self):
        """Replace only unencodable rejected graph payloads, without deep-copying their invalid trees."""
        nested = {}
        for _ in range(1100):
            nested = {"next": nested}
        cyclic = {}
        cyclic["cycle"] = cyclic
        for value in (nested, {"invalidUnicode": "\ud800"}, {"unsupported": {1, 2}}, cyclic):
            analyzed = report(value)
            graph = build_graph(analyzed, kind="audio", duration_ms=1000)
            self.assertEqual(graph["modExtensions"]["rejectedCount"], 1)
            with TemporaryDirectory() as folder:
                source = Path(folder) / "source.wav"
                source.write_bytes(b"native fixture")
                store = MediaGraphStore(Path(folder) / "graphs")
                published = store.publish(identify(source), {"kind": "audio", "durationMs": 1000}, graph, "en", {"modStages": analyzed["modStages"]})
                self.assertEqual(published["nodesByKind"]["utterance"], 1)
                persisted = published["provenance"]["modStages"][0]["result"]["graphExtension"]
                if value is nested:
                    self.assertIn("next", persisted)  # Finite raw JSON stays unchanged when encodable.
                else:
                    self.assertIn("rejectedDiagnostic", persisted)
                self.assertIs(analyzed["modStages"][0]["result"]["graphExtension"], value)

    def test_oversized_nonfinite_stage_remainder_preserves_native_snapshot_and_raw_report(self):
        """Cover reviewer G1: a rejected list's unscanned NaN tail cannot abort publish/reopen."""
        for position in (0, 2048):
            analyzed = report()
            bad = analyzed["modStages"][0]
            bad["result"]["graphExtension"]["nodes"][0]["data"]["score"] = float("nan")
            original = [{"state": "done", "result": {}} for _ in range(2049)]
            original[position] = bad
            analyzed["modStages"] = original
            graph = build_graph(analyzed, kind="audio", duration_ms=1000)
            self.assertEqual(graph["modExtensions"]["rejectedCount"], 1)
            with TemporaryDirectory() as folder:
                source = Path(folder) / "source.wav"
                source.write_bytes(b"native fixture")
                store = MediaGraphStore(Path(folder) / "graphs")
                published = store.publish(identify(source), {"kind": "audio", "durationMs": 1000}, graph, "en", {"modStages": original})
                reopened = MediaGraphStore(store.root).overview(published["snapshotId"])
                diagnostic = reopened["provenance"]["modStagesDiagnostic"]
                self.assertEqual(diagnostic["encoding"], "omitted-remainder")
                self.assertEqual(diagnostic["omittedCount"], 1)
                self.assertEqual(len(reopened["provenance"]["modStages"]), 2048)
                self.assertEqual(reopened["nodesByKind"]["utterance"], 1)
                self.assertEqual(store.search(published["snapshotId"], "Native words", "utterance")["nodes"][0]["id"], "utterance:100:0")
            self.assertIs(analyzed["modStages"], original)
            self.assertEqual(len(original), 2049)
            self.assertIs(original[position], bad)
            self.assertTrue(isnan(bad["result"]["graphExtension"]["nodes"][0]["data"]["score"]))

    def test_unencodable_non_array_stages_preserve_native_persistence(self):
        """Handle the same rejected-container boundary for malformed imported stage metadata."""
        for value in (float("nan"), {"invalid": float("inf")}):
            analyzed = report()
            analyzed["modStages"] = value
            graph = build_graph(analyzed, kind="audio", duration_ms=1000)
            with TemporaryDirectory() as folder:
                source = Path(folder) / "source.wav"
                source.write_bytes(b"native fixture")
                store = MediaGraphStore(Path(folder) / "graphs")
                published = store.publish(identify(source), {"kind": "audio", "durationMs": 1000}, graph, "en", {"modStages": value})
                self.assertEqual(published["nodesByKind"]["utterance"], 1)
                self.assertIn("rejectedDiagnostic", published["provenance"]["modStages"])
                self.assertIs(analyzed["modStages"], value)

    def test_existing_completion_publication_announces_persisted_extension_readiness(self):
        """Reuse the existing atomic completion envelope without adding an event bus or frontend lane."""
        from voice_studio.graph.media_graph_progress import GraphPublication
        publication = GraphPublication("extension-fixture")
        self.assertEqual(publication.state(), {})
        with TemporaryDirectory() as folder:
            directory, published = real_stage_snapshot(folder, publication=publication)
            state = publication.state()
            self.assertEqual(state["phase"], "complete")
            self.assertEqual(state["snapshot"]["snapshotId"], published["snapshotId"])
            self.assertEqual(state["snapshot"]["modExtensions"]["nodeCount"], 2)
            self.assertEqual(MediaGraphStore(directory).overview(published["snapshotId"])["modExtensions"], state["snapshot"]["modExtensions"])

    def test_missing_optional_extension_and_pending_stages_add_no_nodes(self):
        """Leave ordinary stage results and absent optional graph contributions untouched."""
        analyzed = report()
        analyzed["modStages"] = [{"modId": "test.graph", "state": "pending"}, {"state": "done", "result": {"summary": "ordinary result"}}]
        result = build_graph(analyzed, kind="audio", duration_ms=1000)
        self.assertEqual(result["modExtensions"]["nodeCount"], 0)
        self.assertFalse(any(node["kind"] == "extension" for node in result["nodes"]))

    def test_invalid_node_ids_types_intervals_and_finite_data_are_rejected(self):
        """Reject malformed custom node fields while retaining the complete native graph."""
        invalid = [({"id": "../native"}), ({"id": "audio_track:0"}), ({"type": "bad/type"}),
                   ({"label": ""}), ({"label": "x" * 1025}), ({"startMs": True}),
                   ({"startMs": -1}), ({"endMs": 1001}), ({"endMs": 100}),
                   ({"data": {"score": float("nan")}}), ({"data": {1: "bad key"}}),
                   ({"data": {"large": "x" * 4097}}), ({"data": []}), ({"unknown": "field"})]
        for changes in invalid:
            with self.subTest(changes=changes):
                value = extension()
                value["nodes"][0].update(changes)
                self.assert_rejected(value)
        value = extension()
        value["nodes"][0].pop("endMs")
        self.assert_rejected(value)

    def test_invalid_refs_duplicates_versions_and_native_mutations_fail_before_writes(self):
        """Prevent unresolved links, duplicates and native-to-native writes atomically per namespace."""
        cases = []
        for reference in ("utterance:100:0", {"nativeId": "missing"}, {"nativeId": "extension:foreign:id"}, {"nativeId": []}):
            value = extension()
            value["edges"][0]["target"] = reference
            cases.append(value)
        duplicate = extension()
        duplicate["nodes"].append(deepcopy(duplicate["nodes"][0]))
        cases.append(duplicate)
        duplicate_edge = extension()
        duplicate_edge["edges"].append(deepcopy(duplicate_edge["edges"][0]))
        cases.append(duplicate_edge)
        native_edge = extension()
        native_edge["edges"] = [{"source": {"nativeId": "utterance:100:0"}, "relation": "owns", "target": {"nativeId": "utterance:100:0"}}]
        cases.append(native_edge)
        cases.append({**extension(), "apiVersion": True})
        for value in cases:
            graph = MediaGraph(1000, "en")
            graph.node("utterance:100:0", "utterance", "untouched")
            old_nodes = deepcopy(graph.nodes)
            with self.subTest(value=value):
                summary = add_mod_extensions(graph, report(value))
                self.assertEqual(summary["rejectedCount"], 1)
            self.assertEqual(graph.nodes, old_nodes)
            self.assertFalse(graph.edges)

    def test_namespace_collision_rejection_and_cross_mod_isolation(self):
        """Reject conflicting/duplicate namespaces and keep different mods' local IDs independent."""
        analyzed = report()
        graph = MediaGraph(1000, "en")
        graph.node("utterance:100:0", "utterance", "Native")
        key = "extension:test.graph:analysis.graph-fixture:event-1"
        graph.node(key, "utterance", "Cannot overwrite")
        self.assertEqual(add_mod_extensions(graph, analyzed)["rejectedCount"], 1)
        duplicate = deepcopy(analyzed["modStages"][0])
        analyzed["modStages"].append(duplicate)
        self.assertEqual(build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]["rejectedCount"], 2)
        duplicate["modId"] = "another.mod"
        built = build_graph(analyzed, kind="audio", duration_ms=1000)
        self.assertEqual(built["modExtensions"]["nodeCount"], 4)

    def test_budgets_bound_nodes_bytes_nesting_and_total_projection(self):
        """Exercise per-node, per-stage and contributor limits with explicit rejection diagnostics."""
        value = extension()
        value["nodes"] = [{"id": str(index), "type": "custom", "label": "Node"} for index in range(MAX_NODES_PER_STAGE + 1)]
        self.assert_rejected(value, "node budget")
        value = extension()
        nested = {}
        for _ in range(10):
            nested = {"next": nested}
        value["nodes"][0]["data"] = nested
        self.assert_rejected(value, "nesting")
        value["nodes"][0]["data"] = {str(index): "x" * 4000 for index in range(5)}
        self.assert_rejected(value, "node data")
        too_many_edges = extension()
        too_many_edges["edges"] = [{"source": "event-1", "relation": f"relation-{index}", "target": "concept"} for index in range(1025)]
        self.assert_rejected(too_many_edges, "edge budget")
        oversized = {"apiVersion": 1, "nodes": [], "edges": []}
        for index in range(256):
            oversized["nodes"].append({"id": str(index), "type": "custom", "label": "Node", "data": {"text": "x" * 4096}})
        self.assert_rejected(oversized, "contribution")
        analyzed = report()
        analyzed["modStages"] = []
        for index in range(65):
            analyzed["modStages"].append({"modId": f"test.mod-{index}", "capability": "analysis.custom", "state": "done",
                                          "result": {"graphExtension": {"apiVersion": 1, "nodes": [], "edges": []}}})
        summary = build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]
        self.assertEqual(summary["rejectedCount"], 1)
        self.assertIn("contributor budget", summary["contributors"][-1]["error"])

    def test_total_byte_and_edge_budgets_apply_across_individually_valid_stages(self):
        """Bound aggregate work while retaining earlier valid independent namespaces."""
        analyzed = report()
        analyzed["modStages"] = []
        for number in range(9):
            value = extension()
            value["edges"] = [{"source": "event-1", "relation": f"r-{index}", "target": "concept"} for index in range(1000)]
            analyzed["modStages"].append({"modId": f"test.mod-{number}", "capability": "analysis.custom", "state": "done", "result": {"graphExtension": value}})
        summary = build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]
        self.assertEqual(summary["edgeCount"], 8000)
        self.assertEqual(summary["rejectedCount"], 1)
        self.assertIn("total budget", summary["contributors"][-1]["error"])
        analyzed["modStages"] = []
        for number in range(3):
            nodes = []
            for index in range(256):
                nodes.append({"id": str(index), "type": "custom", "label": "Node", "data": {"text": "x" * 3000}})
            value = {"apiVersion": 1, "nodes": nodes, "edges": []}
            analyzed["modStages"].append({"modId": f"test.mod-{number}", "capability": "analysis.custom", "state": "done", "result": {"graphExtension": value}})
        summary = build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]
        self.assertEqual(summary["nodeCount"], 512)
        self.assertEqual(summary["rejectedCount"], 1)
        self.assertIn("total budget", summary["contributors"][-1]["error"])

    def test_rejected_contribution_does_not_lose_other_namespaces_or_import_diagnostics(self):
        """Isolate failed contributions and malformed imported metadata from valid/native work."""
        analyzed = report()
        bad = deepcopy(analyzed["modStages"][0])
        bad["modId"] = "bad.mod"
        bad["result"]["graphExtension"]["edges"][0]["target"] = {"nativeId": "missing"}
        analyzed["modStages"].append(bad)
        before = deepcopy(analyzed)
        built = build_graph(analyzed, kind="audio", duration_ms=1000)
        self.assertEqual(analyzed, before)
        self.assertEqual(built["modExtensions"]["nodeCount"], 2)
        self.assertEqual(built["modExtensions"]["rejectedCount"], 1)
        self.assertTrue(any(node["id"] == "utterance:100:0" for node in built["nodes"]))
        self.assertFalse(any(node["id"].startswith("extension:bad.mod:") for node in built["nodes"]))
        for malformed in (None, {}, [None] * 2049):
            analyzed["modStages"] = malformed
            self.assertEqual(build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]["rejectedCount"], 1)
        analyzed = report()
        analyzed["modStages"][0]["state"] = []
        self.assertEqual(build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]["rejectedCount"], 1)

    def test_maximum_node_projection_has_deterministic_ids_and_bounded_runtime(self):
        """Project the maximum permitted custom node count under a generous latency guard."""
        analyzed = report()
        analyzed["modStages"] = []
        for number in range(8):
            value = {"apiVersion": 1, "nodes": [{"id": str(index), "type": "custom.metric", "label": f"Metric {index}", "startMs": 0, "endMs": 1000, "data": {"value": index}} for index in range(MAX_NODES_PER_STAGE)], "edges": []}
            analyzed["modStages"].append({"modId": f"test.mod-{number}", "capability": "analysis.metrics", "state": "done", "result": {"graphExtension": value}})
        started = perf_counter()
        built = build_graph(analyzed, kind="audio", duration_ms=1000)
        elapsed = perf_counter() - started
        self.assertEqual(built["modExtensions"]["nodeCount"], 2048)
        self.assertLess(elapsed, 2.0, f"2048 node projection took {elapsed:.3f}s")
        self.assertEqual(len({node["id"] for node in built["nodes"]}), len(built["nodes"]))
        extra = deepcopy(analyzed["modStages"][-1])
        extra["modId"] = "test.overflow"
        analyzed["modStages"].append(extra)
        summary = build_graph(analyzed, kind="audio", duration_ms=1000)["modExtensions"]
        self.assertEqual(summary["nodeCount"], 2048)
        self.assertEqual(summary["rejectedCount"], 1)


class MediaGraphModMcpTests(IsolatedAsyncioTestCase):
    """Verify the shipped MCP node/search/timeline contracts need no custom domain code."""

    async def test_fresh_mcp_queries_real_stage_graph_after_reopen(self):
        """Read actual user-stage graph output through fresh shipped stdio MCP tools."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with TemporaryDirectory() as folder:
            directory, published = real_stage_snapshot(folder)
            work = Path(folder)
            params = StdioServerParameters(command=sys.executable,
                args=[str(ROOT / "voice_mcp.py"), "--graph-root", str(directory), "--studio-data", str(work / "studio")],
                cwd=ROOT, env={**os.environ, "VOICE_MEDIA_MODS_DIR": str(work / "mods")})
            async with stdio_client(params) as streams, ClientSession(*streams) as session:
                await session.initialize()
                async def call(name, arguments):
                    """Use real typed MCP tools and reject protocol errors before decoding JSON."""
                    response = await session.call_tool(name, arguments)
                    self.assertFalse(response.isError, response.content)
                    return json.loads(next(item.text for item in response.content if hasattr(item, "text")))
                identity = published["snapshotId"]
                found = await call("media_graph_search", {"snapshot_id": identity, "query": "Quantum signal", "kind": "extension"})
                self.assertEqual(len(found["nodes"]), 1)
                timed = await call("media_graph_timeline", {"snapshot_id": identity, "start_ms": 0, "end_ms": 800, "kind": "extension"})
                self.assertEqual(timed["nodes"][0]["id"], found["nodes"][0]["id"])
                detail = await call("media_graph_node", {"snapshot_id": identity, "node_id": found["nodes"][0]["id"]})
                self.assertEqual(detail["data"]["extensionType"], "acoustic.whistle")
                self.assertEqual(detail["data"]["modProvenance"]["modId"], "test.graph")
