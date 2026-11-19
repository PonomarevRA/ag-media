"""Check benchmark parity projections and failure accounting against real SQLite."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch
from types import SimpleNamespace

from voice_studio.graph.media_graph_store import write_snapshot

PATH = Path(__file__).resolve().parents[1] / 'scripts/benchmark_media_events.py'
SPEC = importlib.util.spec_from_file_location('media_event_benchmark', PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class EventBenchmarkTests(TestCase):
    """Keep the measurement's equivalence relation narrower than graph semantics."""

    def fingerprint(self, manifest, graph):
        """Create a real snapshot and return the benchmark's content projection."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'snapshot.sqlite'
            write_snapshot(path, manifest, graph)
            return BENCHMARK.snapshot_fingerprint(path)

    def fixture(self):
        """Provide one unknown fact and a timed edge so accidental erasure is visible."""
        manifest = {'snapshotId': 'random', 'createdAt': 'now', 'durationMs': 1000,
                    'source': {'sha256': 'source'}, 'coverage': {'scene': [{'status': 'unknown'}]},
                    'provenance': {'progress': {'jobId': 'random', 'revision': 1}, 'model': 'fixed'}}
        graph = {'nodes': [{'id': 'a', 'kind': 'object', 'startMs': 10, 'endMs': 20,
            'label': 'object', 'data': {'confidence': .8, 'unknownFact': None, 'elapsedMs': 2}},
            {'id': 'b', 'kind': 'shot', 'startMs': 0, 'endMs': 1000, 'label': 'shot', 'data': {}}],
            'edges': [('a', 'in_shot', 'b')], 'faceFrames': []}
        return manifest, graph

    def test_only_declared_telemetry_is_ignored(self):
        """Random snapshot/job metadata and elapsed telemetry cannot break parity."""
        manifest, graph = self.fixture()
        before = self.fingerprint(manifest, graph)
        manifest.update(snapshotId='other', createdAt='later')
        manifest['provenance']['progress'] = {'jobId': 'other', 'revision': 9}
        manifest['provenance']['eventBus'] = {'mode': 'priority', 'handlerSeconds': 5}
        graph['nodes'][0]['data']['elapsedMs'] = 100
        self.assertEqual(before, self.fingerprint(manifest, graph))

    def test_ids_clocks_unknowns_confidence_edges_and_coverage_are_preserved(self):
        """Each meaningful graph change must fail the exact stored parity check."""
        manifest, graph = self.fixture()
        before = self.fingerprint(manifest, graph)
        changes = [('clock', lambda value: value['nodes'][0].update(startMs=11)),
                   ('confidence', lambda value: value['nodes'][0]['data'].update(confidence=.7)),
                   ('unknown', lambda value: value['nodes'][0]['data'].pop('unknownFact')),
                   ('edge', lambda value: value.update(edges=[]))]
        for name, mutate in changes:
            with self.subTest(name=name):
                changed = deepcopy(graph)
                mutate(changed)
                self.assertNotEqual(before, self.fingerprint(manifest, changed))
        manifest['coverage']['scene'][0]['status'] = 'sampled'
        self.assertNotEqual(before, self.fingerprint(manifest, graph))

    def test_failed_samples_cannot_be_reported_as_parity(self):
        """One successful sample cannot hide a failed repetition or missing facts."""
        rows = [{'case': 'ui', 'mode': 'sync', 'status': 'passed', 'final': {'nodes': 2},
                 'snapshots': [], 'totalSeconds': 2},
                {'case': 'ui', 'mode': 'priority', 'status': 'passed', 'final': {'nodes': 2},
                 'snapshots': [], 'totalSeconds': 1},
                {'case': 'ui', 'mode': 'priority', 'status': 'failed'}]
        result = BENCHMARK.summarize(rows)['ui']['priority']
        self.assertFalse(result['parity'])
        self.assertFalse(result['partialParity'])
        self.assertEqual((result['n'], result['passed']), (2, 1))
        self.assertEqual(result['metrics']['totalSeconds'], {'n': 1, 'median': 1, 'min': 1, 'max': 1})

    def test_partial_content_regression_is_reported_separately(self):
        """A matching final snapshot does not prove intermediate publication parity."""
        rows = [{'case': 'ui', 'mode': 'sync', 'status': 'passed', 'final': {'nodes': 2}, 'snapshots': ['early']},
                {'case': 'ui', 'mode': 'fifo', 'status': 'passed', 'final': {'nodes': 2}, 'snapshots': ['different']}]
        result = BENCHMARK.summarize(rows)['ui']['fifo']
        self.assertTrue(result['parity'])
        self.assertFalse(result['partialParity'])

    def test_publication_order_rejects_duplicates_reversal_and_early_final(self):
        """Matching stored facts do not excuse invalid public revision ordering."""
        good = [{'phase': 'base', 'revision': 1}, {'phase': 'complete', 'revision': 2}]
        self.assertTrue(BENCHMARK.publication_order(good)['passed'])
        bad = [[], [{'phase': 'base', 'revision': 1}],
               [{'phase': 'base', 'revision': 2}, {'phase': 'complete', 'revision': 1}],
               [{'phase': 'base', 'revision': 1}, {'phase': 'base', 'revision': 2},
                {'phase': 'complete', 'revision': 3}],
               [{'phase': 'complete', 'revision': 1}, {'phase': 'base', 'revision': 2}]]
        for events in bad:
            with self.subTest(events=events):
                self.assertFalse(BENCHMARK.publication_order(events)['passed'])

    def test_stale_success_cannot_hide_a_crashed_worker(self):
        """A rerun removes its previous result before invoking an isolated host."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'rerun.json').write_text('{"status":"passed"}')
            options = SimpleNamespace(output=root, manifest=root / 'manifest.json', scratch=root,
                gap_ms=50, baseline=root / 'baseline.py', timeout=1)
            process = Mock(returncode=1)
            process.poll.return_value = 1
            with patch.object(BENCHMARK.subprocess, 'Popen', return_value=process):
                result = BENCHMARK.run_isolated(options, 'rerun', 'sync', 0)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['exitCode'], 1)

    def test_native_identity_proof_preserves_unknown_graph_facts(self):
        """The bounded UUID proof must reject any additional graph fact drift."""
        path = PATH.parents[1] / 'data/exports/event-bus-20261118/qa/check_native_identity.py'
        spec = importlib.util.spec_from_file_location('native_identity_proof', path)
        proof = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proof)
        reference, candidate = proof.read_store('baseline'), proof.read_store('sync')
        mapping = proof.identity_mapping(reference, candidate)
        self.assertTrue(proof.check(reference, candidate, mapping)['passed'])
        changed = deepcopy(candidate)
        changed['nodes'][0][5]['unknown_fact_must_survive'] = None
        self.assertFalse(proof.check(reference, changed, mapping)['passed'])
