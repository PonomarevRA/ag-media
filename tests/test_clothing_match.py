"""Prevent weak outfit matches from taking over face tracking."""

from math import sqrt
from unittest import TestCase

from voice_studio.vision.clothing_match import borderline_pairs, choose_sample_tracks, corroborate
from voice_studio.vision.face_reid import group_tracks


def track(start, end):
    return [(frame, None) for frame in range(start, end + 1)]


def signatures(score):
    return {1: [(1.0, 0.0)] * 2, 2: [(score, sqrt(1 - score ** 2))] * 2}


def outfit(face, histogram):
    return [{"face": face, "timeMs": time_ms, "swatch": "#102030", "hist": histogram}
            for time_ms in (face * 100, face * 100 + 40)]


class ClothingMatchTests(TestCase):
    def test_same_outfit_can_confirm_only_a_near_face_match(self):
        tracks = [track(0, 19), track(40, 60)]
        refs = signatures(.52)
        baseline, _ = group_tracks(tracks, refs)
        pairs = borderline_pairs(tracks, refs, baseline)
        self.assertEqual(choose_sample_tracks(tracks, baseline, pairs), {1: 1, 2: 2})
        revised, matches, descriptions = corroborate(pairs, outfit(1, [1, 0]) + outfit(2, [1, 0]), baseline)
        self.assertEqual(revised, {1: 1, 2: 1})
        self.assertEqual(matches[0]["basis"], "face-and-clothing")
        self.assertEqual(len(descriptions[1]["sampleTimesMs"]), 4)

    def test_same_clothes_do_not_merge_simultaneous_or_different_faces(self):
        for tracks, score in (([track(0, 30), track(20, 55)], .52),
                              ([track(0, 19), track(40, 60)], .32)):
            refs = signatures(score)
            baseline, _ = group_tracks(tracks, refs)
            self.assertEqual(borderline_pairs(tracks, refs, baseline), [])

    def test_different_or_unreliable_outfits_leave_the_existing_face_groups(self):
        baseline = {1: 1, 2: 2}
        pairs = [(1, 2, .52)]
        first = outfit(1, [1, 0])
        second = outfit(2, [0, 1])
        for evidence in (first + second, first + second[:1]):
            aliases, matches, _ = corroborate(pairs, evidence, baseline)
            self.assertEqual(aliases, baseline)
            self.assertEqual(matches, [])

    def test_clothing_cannot_change_an_already_confirmed_sface_merge(self):
        tracks = [track(0, 19), track(40, 60)]
        refs = signatures(.74)
        baseline, _ = group_tracks(tracks, refs)
        self.assertEqual(baseline, {1: 1, 2: 1})
        self.assertEqual(borderline_pairs(tracks, refs, baseline), [])


    def test_pattern_is_displayed_only_when_both_segmented_samples_agree(self):
        first = [{**sample, "pattern": "plain"} for sample in outfit(1, [1, 0])]
        second = [{**sample, "pattern": "patterned"} for sample in outfit(2, [1, 0])]
        _, _, unmerged = corroborate([], first + second, {1: 1, 2: 2})
        self.assertEqual(unmerged[1]["pattern"], "plain")
        self.assertEqual(unmerged[2]["pattern"], "patterned")
        _, _, merged = corroborate([(1, 2, .52)], first + second, {1: 1, 2: 2})
        self.assertNotIn("pattern", merged[1])
