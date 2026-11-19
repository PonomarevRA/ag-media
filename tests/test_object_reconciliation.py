"""Conservative optional object identity and bounded real-crop local-part checks."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

from unittest import skipIf

from voice_studio.graph.object_reconciliation import (MAX_CANDIDATES, reconcile_object_groups,
                                                       source_gallery)
from voice_studio.vision.object_reconciliation import prepare_object_reconciliation
if cv2 is not None:
    from voice_studio.vision.object_reconciliation_parts import crop_features, compare_parts


def fixture(stamp, *, color='#cc2222', box=None, end=None):
    """Create one independently identified raw observation and its original track."""
    node = {'id': f'obs:{stamp}', 'label': 'car', 'startMs': stamp, 'endMs': stamp + 1,
            'data': {'box': box or [.1, .1, .6, .5], 'objectPalette': [{'hex': color, 'share': .8}]}}
    track = {'id': f'track:{stamp}', 'label': 'car', 'startMs': stamp, 'endMs': end or stamp + 1}
    return ('shot', [track], 'single-track'), node


def witness(source, target, *, inliers=24):
    """Attach exact measured-part evidence for identity-policy tests."""
    target['data'].setdefault('objectPartMatches', []).append({
        'timeMs': source['startMs'], 'label': source['label'], 'box': source['data']['box'],
        'basis': 'orb-local-parts', 'inliers': inliers, 'inlierFraction': .9,
        'sourceCoverage': .3, 'targetCoverage': .3, 'sourceCells': 5, 'targetCells': 5})


def reconcile(*pairs):
    """Apply the pure graph policy to source fixtures."""
    return reconcile_object_groups([pair[0] for pair in pairs],
                                   {pair[0][1][0]['id']: [pair[1]] for pair in pairs})


class ReconciliationTests(TestCase):
    def test_measured_occlusion_views_merge_without_raw_mutation(self):
        first, second = fixture(0), fixture(3000)
        witness(first[1], second[1])
        before = deepcopy((first, second))
        links, stats = reconcile(first, second)
        self.assertEqual(links, {0: 1})
        self.assertEqual(stats['acceptedLinks'][0]['fromObservationId'], 'obs:0')
        self.assertEqual((first, second), before)
        self.assertEqual(reconcile(first, second), (links, stats))

    def test_name_color_and_background_cannot_create_identity(self):
        first, second = fixture(0), fixture(3000)
        second[1]['data']['objectMatches'] = [{'objectSimilarity': 1, 'backgroundSimilarity': 1}]
        self.assertEqual(reconcile(first, second)[0], {})

    def test_red_white_veto_even_with_part_score(self):
        first, second = fixture(0), fixture(3000, color='#eeeeee')
        witness(first[1], second[1])
        self.assertEqual(reconcile(first, second)[0], {})

    def test_simultaneous_distinct_objects_never_merge(self):
        first, second = fixture(0, end=4000), fixture(3000)
        witness(first[1], second[1])
        self.assertEqual(reconcile(first, second)[0], {})

    def test_ambiguous_reciprocal_choices_abstain(self):
        first, second, third = fixture(0), fixture(3000), fixture(3100)
        witness(first[1], second[1])
        witness(first[1], third[1])
        self.assertEqual(reconcile(first, second, third)[0], {})

    def test_no_transitive_bridge_without_direct_witness(self):
        first, second, third = fixture(0), fixture(3000), fixture(6000)
        witness(first[1], second[1])
        witness(second[1], third[1])
        self.assertEqual(reconcile(first, second, third)[0], {0: 1})

    def test_invalid_and_tiny_part_support_abstains(self):
        for invalid in (float('nan'), True, 7):
            first, second = fixture(0), fixture(3000)
            witness(first[1], second[1], inliers=invalid)
            self.assertEqual(reconcile(first, second)[0], {})

    def test_candidate_overflow_abstains_and_comparisons_bounded(self):
        pairs = [fixture(index * 100) for index in range(200)]
        for pair in pairs[1:]:
            witness(pairs[0][1], pair[1])
        links, stats = reconcile(*pairs)
        self.assertNotIn(0, links)
        self.assertGreater(stats['overflowGroups'], 0)
        self.assertLessEqual(stats['comparisons'], len(pairs) * MAX_CANDIDATES)

    def test_gallery_has_only_bounded_measured_crop_locators(self):
        rows = [fixture(index * 100)[1] for index in range(200)]
        result = source_gallery(rows)
        self.assertLessEqual(len(result['views']), 8)
        self.assertEqual(result['basis'], 'measured-source-crops-not-3d')
        self.assertTrue({v['observationId'] for v in result['views']} <= {r['id'] for r in rows})


@skipIf(cv2 is None, "Requires the existing mediapipe vision environment")
class PartMeasurementTests(TestCase):
    def texture(self):
        """Create reproducible distributed features; no external model or media needed."""
        image = np.zeros((320, 320, 3), np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(180):
            point = tuple(int(v) for v in rng.integers(20, 300, 2))
            color = tuple(int(v) for v in rng.integers(60, 255, 3))
            cv2.circle(image, point, int(rng.integers(2, 8)), color, -1)
        return image

    def test_real_descriptor_rotation_and_partial_occlusion(self):
        image = self.texture()
        rotation = cv2.getRotationMatrix2D((160, 160), 24, .93)
        changed = cv2.warpAffine(image, rotation, (320, 320))
        changed[:80, :150] = 0
        detector = cv2.ORB_create(nfeatures=512, edgeThreshold=12, fastThreshold=12)
        first = crop_features(image, [0, 0, 1, 1], detector)
        second = crop_features(changed, [0, 0, 1, 1], detector)
        result = compare_parts(first, second)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['inliers'], 8)
        self.assertGreaterEqual(result['sourceCoverage'], .12)
        self.assertEqual(compare_parts(first, second), result)

    def test_unrelated_and_uniform_views_have_no_evidence(self):
        detector = cv2.ORB_create(nfeatures=512)
        image = self.texture()
        other = np.random.default_rng(12).integers(0, 255, image.shape, dtype=np.uint8)
        self.assertIsNone(compare_parts(crop_features(image, [0, 0, 1, 1], detector),
                                       crop_features(other, [0, 0, 1, 1], detector)))
        self.assertIsNone(crop_features(np.zeros_like(image), [0, 0, 1, 1], detector))

    def test_missing_source_is_explicit_and_preserves_input(self):
        report = {'sceneEvidence': {'frames': []}}
        result = prepare_object_reconciliation('/private/tmp/absent-object-pilot-video.mp4', report)
        self.assertEqual(report, {'sceneEvidence': {'frames': []}})
        self.assertEqual(result['objectReconciliationEvidence']['status'], 'source-unavailable')


@skipIf(cv2 is None, "Requires existing vision runtime")
class WorkerBudgetTests(TestCase):
    def test_decoder_crop_and_descriptor_budgets_and_stale_evidence(self):
        """Large reports cannot bypass frame/crop limits or retain stale pilot pairs."""
        from voice_studio.vision import object_reconciliation_parts as worker
        image = np.zeros((120, 120, 3), np.uint8)
        class Capture:
            def isOpened(self):
                return True
            def set(self, *args):
                return True
            def read(self):
                return True, image
            def release(self):
                pass
        frames = [{'timeMs': stamp * 100, 'objects': [
            {'label': 'car', 'box': [0, 0, 1, 1], 'objectPartMatches': [{'stale': True}]}
            for _ in range(20)]} for stamp in range(100)]
        report = {'sceneEvidence': {'frames': frames}}
        with patch.object(worker.cv2, 'VideoCapture', return_value=Capture()):
            result = worker.prepare_object_reconciliation('fixture.mp4', report)
        stats = result['objectReconciliationEvidence']
        self.assertLessEqual(stats['decodedFrames'], 48)
        self.assertLessEqual(stats['describedCrops'], 96)
        self.assertLessEqual(stats['comparisons'], 96 * 16)
        self.assertEqual(stats['status'], 'no-evidence')
        self.assertFalse(any('objectPartMatches' in item for frame in result['sceneEvidence']['frames']
                             for item in frame['objects']))
        self.assertIn('objectPartMatches', report['sceneEvidence']['frames'][0]['objects'][0])


class GraphProjectionTests(TestCase):
    def test_parts_across_cut_keep_tracks_and_publish_real_gallery(self):
        """Optional measured matches cross cuts without changing default raw tracking."""
        from voice_studio.graph.media_graph_model import build_graph, object_layers
        first = {'label': 'car', 'box': [.1, .1, .6, .5], 'score': .8}
        second = {'label': 'car', 'box': [.2, .2, .5, .4], 'score': .8,
                  'objectPartMatches': [{'timeMs': 100, 'label': 'car', 'box': first['box'],
                     'basis': 'orb-local-parts', 'inliers': 24, 'inlierFraction': .9,
                     'sourceCoverage': .3, 'targetCoverage': .3, 'sourceCells': 5, 'targetCells': 5}]}
        report = {'durationMs': 5000, 'videoEvidence': {'editorTimeline': {'shots': [
            {'startMs': 0}, {'startMs': 2000}]}}, 'sceneEvidence': {'frames': [
            {'timeMs': 100, 'objects': [first]}, {'timeMs': 3100, 'objects': [second]}]}}
        baseline = object_layers(report)
        self.assertEqual(len(baseline['objectProfiles']), 2)
        report['objectReconciliationEnabled'] = True
        pilot = object_layers(report)
        self.assertEqual(len(pilot['objectProfiles']), 1)
        profile, = pilot['objectProfiles']
        self.assertEqual(profile['shotIds'], ['shot:0', 'shot:2000'])
        self.assertEqual([row['timeMs'] for row in profile['objectGallery']['views']], [100, 3100])
        self.assertEqual([row['id'] for row in baseline['objectTimeline']],
                         [row['id'] for row in pilot['objectTimeline']])
        self.assertEqual(pilot, object_layers(report))
        graph = build_graph(report, kind='video', duration_ms=5000, object_reconciliation=True)
        self.assertEqual(sum(node['kind'] == 'object_profile' for node in graph['nodes']), 1)
        self.assertEqual(sum(node['kind'] == 'object_observation' for node in graph['nodes']), 2)

    def test_existing_baseline_chain_preserved_but_explicit_color_conflict_vetoes(self):
        """The pilot extends legacy behavior without silently undoing its safe identities."""
        from voice_studio.graph.object_reconciliation import reconcile_object_groups
        first, second = fixture(0), fixture(3000)
        groups = [first[0], second[0]]
        samples = {'track:0': [first[1]], 'track:3000': [second[1]]}
        links, diagnostics = reconcile_object_groups(groups, samples, baseline_links={0: 1})
        self.assertEqual(links, {0: 1})
        self.assertEqual(diagnostics['preservedBaselineLinks'], 1)
        second[1]['data']['objectPalette'] = [{'hex': '#eeeeee', 'share': .8}]
        links, diagnostics = reconcile_object_groups(groups, samples, baseline_links={0: 1})
        self.assertEqual(links, {})
        self.assertEqual(diagnostics['rejectedBaselineLinks'][0]['reason'], 'time-or-color-conflict')

    def test_baseline_chain_cannot_bridge_red_unknown_white(self):
        """A mixed middle palette cannot hide a clear conflict elsewhere in a legacy chain."""
        from voice_studio.graph.object_reconciliation import reconcile_object_groups
        first, middle, last = fixture(0), fixture(3000), fixture(6000, color='#eeeeee')
        middle[1]['data'].pop('objectPalette')
        groups = [first[0], middle[0], last[0]]
        samples = {pair[0][1][0]['id']: [pair[1]] for pair in (first, middle, last)}
        links, diagnostics = reconcile_object_groups(groups, samples, baseline_links={0: 1, 1: 2})
        self.assertEqual(links, {0: 1})
        self.assertEqual(diagnostics['rejectedBaselineLinks'][0]['fromGroup'], 1)

    def test_malformed_imported_witness_reference_abstains(self):
        """Untrusted optional match locators cannot crash graph publication."""
        first, second = fixture(0), fixture(3000)
        for raw in ({'timeMs': [], 'label': {}, 'box': []}, {'timeMs': 0, 'label': 'car', 'box': 'bad'}, None):
            second[1]['data']['objectPartMatches'] = [raw]
            self.assertEqual(reconcile(first, second)[0], {})
