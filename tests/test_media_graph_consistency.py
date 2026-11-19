"""Protect measured scene facts across window alignment and immutable graph reuse."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.graph.media_graph_ingest import apply_scene_evidence, offset_object_matches
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.graph.media_graph_store import MediaGraphStore, identify


class MediaGraphConsistencyTests(TestCase):
    def test_snapshot_scene_reuse_preserves_measured_object_identity(self):
        """A frozen scene must rebuild the same measured cross-cut object profile."""
        first = {'label': 'car', 'box': [.1, .1, .6, .5], 'score': .8}
        second = {'label': 'car', 'box': [.2, .2, .5, .4], 'score': .7,
                  'objectPartMatches': [{'timeMs': 100, 'label': 'car', 'box': first['box'],
                      'basis': 'orb-local-parts', 'inliers': 24, 'inlierFraction': .9,
                      'sourceCoverage': .3, 'targetCoverage': .3, 'sourceCells': 5, 'targetCells': 5}]}
        report = {'language': 'en', 'objectReconciliationEnabled': True,
                  'videoEvidence': {'editorTimeline': {'shots': [{'startMs': 0}, {'startMs': 2000}]}},
                  'sceneEvidence': {'status': 'ok', 'frames': [
                      {'timeMs': 100, 'objects': [first]}, {'timeMs': 3100, 'objects': [second]}]}}
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'synthetic.mp4'
            source.write_bytes(b'synthetic identity fixture; no decoder runs')
            original = identify(source)
            store = MediaGraphStore(Path(directory) / 'graphs')
            graph = build_graph(report, kind='video', duration_ms=5000)
            snapshot = store.publish(original, {'kind': 'video', 'durationMs': 5000}, graph, 'en', {})
            restored, _ = store.scene_evidence(snapshot['snapshotId'], original['sha256'], 'en')
            rebuilt = build_graph({**report, 'sceneEvidence': restored}, kind='video', duration_ms=5000)
        self.assertEqual(restored['frames'][1]['objects'][0], second)
        self.assertEqual([row for row in rebuilt['nodes'] if row['kind'] == 'object_profile'],
                         [row for row in graph['nodes'] if row['kind'] == 'object_profile'])
        self.assertEqual(sum(row['kind'] == 'object_profile' for row in rebuilt['nodes']), 1)
        self.assertEqual(rebuilt['edges'], graph['edges'])

    def test_window_alignment_shifts_all_pair_witnesses_without_mutating_checkpoint(self):
        """Both reappearance and local-part references use the final source clock."""
        original = {'label': 'car', 'score': .9,
                    'objectMatches': [{'timeMs': 10, 'objectSimilarity': .8}],
                    'objectPartMatches': [{'timeMs': 20, 'inliers': 24, 'inlierFraction': .9}]}
        before = deepcopy(original)
        scene = {'status': 'ok', 'frames': [{'timeMs': 50, 'objects': [original]}]}
        report = {'sceneEvidence': {'frames': []}, 'coverage': {'scene': []}}
        apply_scene_evidence(report, scene, 120_000, 120_100)
        frame, = report['sceneEvidence']['frames']
        measured, = frame['objects']
        self.assertEqual(frame['timeMs'], 120_050)
        expected = deepcopy(original)
        expected['objectMatches'][0]['timeMs'] += 120_000
        expected['objectPartMatches'][0]['timeMs'] += 120_000
        self.assertEqual(measured, expected)
        self.assertEqual(original, before)
        self.assertEqual(offset_object_matches({'label': 'car'}, 120_000), {'label': 'car'})
        self.assertEqual(offset_object_matches({'objectPartMatches': [{'timeMs': 1}]}, 5),
                         {'objectPartMatches': [{'timeMs': 6}]})

    def test_complete_scroll_may_end_at_window_boundary_but_sample_may_not(self):
        """Retain complete half-open intervals while excluding samples at the next window."""
        pages = [{'startMs': 0, 'endMs': end, 'viewports': [{'timeMs': 5}]} for end in (99, 100, 101)]
        scene = {'status': 'ok', 'frames': [{'timeMs': 99}, {'timeMs': 100}],
                 'interfaceScrolls': pages}
        original = deepcopy(scene)
        report = {'sceneEvidence': {'frames': []}, 'coverage': {'scene': []}}
        apply_scene_evidence(report, scene, 1000, 1100)
        self.assertEqual([row['timeMs'] for row in report['sceneEvidence']['frames']], [1099])
        self.assertEqual(report['sceneEvidence']['interfaceScrolls'], [
            {'startMs': 1000, 'endMs': 1099, 'viewports': [{'timeMs': 1005}]},
            {'startMs': 1000, 'endMs': 1100, 'viewports': [{'timeMs': 1005}]},
        ])
        self.assertEqual(scene, original)
