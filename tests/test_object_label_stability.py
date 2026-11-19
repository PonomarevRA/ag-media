"""Constrain temporal label repair to near ties and retain inspectable raw evidence."""
from copy import deepcopy
from unittest import TestCase

from voice_studio.graph.media_object_profiles import stable_label_evidence
from voice_studio.graph.media_graph_model import build_graph, object_layers


def sample(label, stamp, score, suffix=''):
    """Create one original detector record, including independently visible tile aliases."""
    return {'id': f'{label}:{stamp}:{suffix}', 'label': label, 'startMs': stamp, 'endMs': stamp + 1,
            'data': {'score': score, 'box': [.2, .3, .4, .3]}}


def label_rows(samples):
    """Prepare the existing peak-first per-track labels contract for this focused fixture."""
    rows = []
    for item in samples:
        rows.append({'label': item['label'], 'observationId': item['id'], 'timeMs': item['startMs'],
                     'score': item['data']['score'], 'trackId': item['id']})
    return sorted(rows, key=lambda row: (row['score'] is None, -(row['score'] or 0), row['timeMs']))


class ObjectLabelStabilityTests(TestCase):
    def test_near_tie_uses_repeated_car_but_preserves_airplane_peak_and_input(self):
        """The measured car majority changes only the display choice, with its own score."""
        samples = [sample('airplane', t, .527) for t in range(3)]
        samples += [sample('car', t, .475) for t in range(3, 15)]
        original = deepcopy(samples)
        evidence = stable_label_evidence(label_rows(samples), samples)
        self.assertEqual((evidence['selectedLabel'], evidence['selectedLabelScore']), ('car', .475))
        self.assertEqual(evidence['basis'], 'near-tie-temporal-support')
        self.assertEqual(evidence['peakRawObservation']['score'], .527)
        self.assertEqual(evidence['peakRawObservation']['label'], 'airplane')
        self.assertEqual(evidence['support'][1]['uniqueTimestampCount'], 12)
        self.assertEqual(samples, original)

    def test_high_confidence_or_insufficient_support_keeps_peak(self):
        """Large score gaps and weak temporal support must retain the strongest observation."""
        for peak, car_score, count in ((.681, .357, 12), (.527, .475, 2), (.527, .475, 5)):
            with self.subTest(peak=peak, car_score=car_score, count=count):
                samples = [sample('airplane', t, peak) for t in range(3)]
                samples += [sample('car', t + 3, car_score) for t in range(count)]
                evidence = stable_label_evidence(label_rows(samples), samples)
                self.assertEqual(evidence['selectedLabel'], 'airplane')
                self.assertEqual(evidence['basis'], 'strongest-observation')

    def test_tiled_duplicates_cannot_manufacture_majority(self):
        """Multiple crops at one instant contribute exactly one temporal vote."""
        samples = [sample('airplane', t, .527) for t in range(3)]
        samples += [sample('car', 100, .475, suffix=str(n)) for n in range(12)]
        evidence = stable_label_evidence(label_rows(samples), samples)
        self.assertEqual(evidence['selectedLabel'], 'airplane')
        self.assertEqual(evidence['support'][1]['uniqueTimestampCount'], 1)

    def test_ambiguous_equal_temporal_leaders_keep_peak(self):
        """Two equally supported alternatives provide no unique label decision."""
        samples = [sample('airplane', t, .527) for t in range(3)]
        samples += [sample(label, t + 10, .475) for label in ('car', 'boat') for t in range(6)]
        self.assertEqual(stable_label_evidence(label_rows(samples), samples)['selectedLabel'], 'airplane')

    def test_measured_ui_protects_phone_peak_from_repeated_tie_label(self):
        """The Pashto phone case is a regression unless independent UI facts constrain the votes."""
        samples = [sample('tie', t, .516) for t in (11733, 12200, 12633, 13100, 13567, 14000, 14433)]
        samples += [sample('cell phone', t, .574) for t in (14433, 14900, 15800)]
        labels = label_rows(samples)
        for interface in ({'type': 'mobile-ui', 'basis': 'layout-and-text'},
                          {'type': 'mobile-ui', 'basis': 'verified-panel'}):
            evidence = stable_label_evidence(labels, samples, interface)
            self.assertEqual(evidence['selectedLabel'], 'cell phone')
            self.assertEqual(evidence['selectedLabelScore'], .574)
            self.assertEqual(evidence['basis'], 'interface-supported-strongest-observation')
            self.assertEqual(evidence['support'][1]['uniqueTimestampCount'], 7)
        for interface in (None, {'type': 'mobile-ui'}, {'type': 'unknown', 'basis': 'layout-and-text'}):
            self.assertEqual(stable_label_evidence(labels, samples, interface)['selectedLabel'], 'tie')

    def test_ui_guard_does_not_invent_a_device_or_change_a_different_peak(self):
        """UI context can corroborate a peak device observation, never create one without evidence."""
        samples = [sample('tie', t, .7) for t in range(3)]
        samples += [sample('cell phone', 100, .4)]
        evidence = stable_label_evidence(label_rows(samples), samples,
                                         {'type': 'mobile-ui', 'basis': 'layout-and-text'})
        self.assertEqual(evidence['selectedLabel'], 'tie')
        self.assertEqual(evidence['basis'], 'strongest-observation')

    def test_missing_scores_and_single_label_keep_original(self):
        """Optional confidence values and legacy single labels remain supported."""
        for samples in ([sample('car', 1, None)], [sample('car', 1, .5)],
                        [sample('car', 1, .5), sample('airplane', 2, None)]):
            evidence = stable_label_evidence(label_rows(samples), samples)
            self.assertEqual(evidence['selectedLabel'], 'car')
            self.assertEqual(evidence['basis'], 'strongest-observation')

    def test_graph_profile_exposes_selected_score_and_keeps_preview_and_raw_nodes(self):
        """The production builder publishes provenance without altering source observations."""
        observations = [sample('car', t, .475) for t in (100, 200, 300, 400, 500, 600)]
        observations += [sample('airplane', t, .527) for t in (700, 800, 900)]
        report = {'durationMs': 1200, 'videoEvidence': {'editorTimeline': {'shots': [{'startMs': 0}]}}, 'sceneEvidence': {'frames': [
            {'timeMs': item['startMs'], 'objects': [{'label': item['label'], **item['data']}]}
            for item in observations]}}
        original = deepcopy(report)
        profile, = object_layers(report)['objectProfiles']
        self.assertEqual(profile['label'], 'car')
        self.assertEqual(profile['labelEvidence']['selectedLabelScore'], .475)
        self.assertEqual(profile['labels'][0]['label'], 'airplane')
        self.assertEqual(profile['labels'][0]['score'], .527)
        graph = build_graph(report, kind='video', duration_ms=1200)
        raw = [node for node in graph['nodes'] if node['kind'] == 'object_observation']
        self.assertEqual(len(raw), len(observations))
        by_id = {node['id']: node for node in raw}
        evidence = profile['labelEvidence']
        self.assertEqual(by_id[evidence['selectedObservation']['observationId']]['label'], 'car')
        self.assertEqual(by_id[evidence['peakRawObservation']['observationId']]['label'], 'airplane')
        self.assertEqual(report, original)
