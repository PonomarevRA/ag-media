"""Keep moving objects linked only with exact source-backed foreground evidence."""
from copy import deepcopy
from unittest import TestCase

from voice_studio.graph.media_graph_model import object_layers
from voice_studio.graph.media_graph_objects import split_tracks


FIRST_BOX = [.1005, .6767, .1648, .2981]
SECOND_BOX = [.0942, .5543, .1679, .2885]


def observation(box, **extra):
    """Build one unchanged detector observation; neutral palette excludes color-only relinking."""
    return {'label': 'sports ball', 'box': list(box), 'score': .41,
            'objectPalette': [{'hex': '#181818', 'share': .8}], **extra}


def witness(**extra):
    """Use the actual 8ap source-pair similarity values, with fixture clock shifted to100ms."""
    return {'timeMs': 100, 'label': 'sports ball', 'box': list(FIRST_BOX),
            'objectSimilarity': .951, 'edgeSimilarity': .781,
            'backgroundSimilarity': .607, 'scaleRatio': 1.001, **extra}


def report(second, *, second_time=184, first=None, shots=(0,)):
    """Make a small source-clock report through the same public graph projection as Studio."""
    return {'durationMs': 2000, 'sceneEvidence': {'frames': [
        {'timeMs': 100, 'objects': first or [observation(FIRST_BOX)]},
        {'timeMs': second_time, 'objects': second},
    ]}, 'videoEvidence': {'editorTimeline': {'shots': [{'startMs': time} for time in shots]}}}


class ObjectMotionWitnessTests(TestCase):
    def test_exact_foreground_witness_links_moving_grayscale_object_without_editing_source(self):
        """The real moving-ball geometry should form one track while preserving both measured boxes."""
        source = report([observation(SECOND_BOX, objectMatches=[witness()])])
        frozen = deepcopy(source)
        layers = object_layers(source)
        self.assertEqual(len(layers['objectTimeline']), 1)
        track, = layers['objectTimeline']
        self.assertEqual(track['sampleCount'], 2)
        self.assertEqual(track['boxSamples'], [[100, FIRST_BOX, .41], [184, SECOND_BOX, .41]])
        self.assertEqual(len(layers['objectProfiles']), 1)
        self.assertEqual(source, frozen)

    def test_missing_weak_or_wrong_source_witness_cannot_extend_overlap_threshold(self):
        """Background, unrelated frames, altered boxes and malformed metrics are not identity evidence."""
        cases = [[], [witness(objectSimilarity=.4, backgroundSimilarity=.999)],
                 [witness(edgeSimilarity=.3)], [witness(timeMs=99)],
                 [witness(box=[.11, .6767, .1648, .2981])], [witness(label='cup')],
                 [witness(scaleRatio=2.)], [witness(objectSimilarity=None)],
                 [witness(objectSimilarity=float('nan'))]]
        for matches in cases:
            with self.subTest(matches=matches):
                layers = object_layers(report([observation(SECOND_BOX, objectMatches=matches)]))
                self.assertEqual(len(layers['objectTimeline']), 2)

    def test_witness_cannot_cross_cut_or_exceed_worker_comparison_gap(self):
        """Measured appearance does not override source shot boundaries or its own sampling budget."""
        for settings in ({'shots': (0, 150)}, {'second_time': 1600}):
            with self.subTest(settings=settings):
                layers = object_layers(report([observation(SECOND_BOX, objectMatches=[witness()])], **settings))
                self.assertEqual(len(layers['objectTimeline']), 2)

    def test_two_current_candidates_remain_separate_even_with_equal_strong_witnesses(self):
        """One-to-many evidence must abstain rather than let input order select a winner."""
        first = observation(SECOND_BOX, objectMatches=[witness()])
        other_box = [.096, .553, .1679, .2885]
        second = observation(other_box, objectMatches=[witness()])
        for ordered in ([first, second], [second, first]):
            layers = object_layers(report(ordered))
            self.assertEqual(len(layers['objectTimeline']), 3)
            self.assertEqual(sum(item['sampleCount'] for item in layers['objectTimeline']), 3)

    def test_two_previous_candidates_remain_ambiguous(self):
        """Two separately observed objects cannot collapse into one because both crops look similar."""
        other_box = [.098, .6767, .1648, .2981]
        first = [observation(FIRST_BOX), observation(other_box)]
        second = observation(SECOND_BOX, objectMatches=[witness(), witness(box=other_box)])
        layers = object_layers(report([second], first=first))
        self.assertEqual(len(layers['objectTimeline']), 3)

    def test_legacy_geometry_and_detail_boundaries_are_preserved(self):
        """The normal IoU path needs no appearance field; detail passes cannot steal a baseline ID."""
        ordinary = object_layers(report([observation([.101, .675, .1648, .2981])]))
        self.assertEqual(len(ordinary['objectTimeline']), 1)
        detail = observation(SECOND_BOX, objectMatches=[witness()], detailBasis='small-object-region')
        self.assertEqual(len(object_layers(report([detail]))['objectTimeline']), 2)

    def test_verified_mobile_panel_still_breaks_a_display_track(self):
        """A saved foreground witness cannot bridge an intervening independently detected panel."""
        first = {'id': 'a', 'label': 'tv', 'startMs': 100, 'data': {'box': FIRST_BOX}}
        second = {'id': 'b', 'label': 'tv', 'startMs': 184,
                  'data': {'box': SECOND_BOX, 'objectMatches': [witness(label='tv')]}}
        tracks = split_tracks({100: [first], 184: [second]}, displays=True,
                              panel_checks=[(150, [0., 0., 1., 1.])])
        self.assertEqual(len(tracks), 2)
