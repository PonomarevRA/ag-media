"""Protect matching rules for local fragment scores and retry offsets."""

from unittest import TestCase

from voice_studio.synthesis.fragment_similarity import matching_windows
from voice_studio.prosody.morpheme_graph import word_graph


def graph(text, heard, interval=600):
    """Build timed recognition for a deterministic alignment fixture."""
    segments = [{"text": word, "offsets": {"from": i * interval, "to": (i + 1) * interval}}
                for i, word in enumerate(heard.split())]
    return word_graph(text, segments)


class FragmentSimilarityTests(TestCase):
    def test_context_uses_original_word_offset(self):
        original = graph("у меня пять шариков на столе", "у меня пять шариков на столе")
        retry = graph("пять шариков на столе", "пять шариков на столе")
        self.assertEqual(matching_windows(retry, original, 1, 2),
                         ((0, 1800), (1200, 3000), "пять шариков на"))
        self.assertIsNone(matching_windows(retry, original, 1, 1))

    def test_abstains_from_scoring_short_mismatched_or_extra_words(self):
        original = graph("пять шариков дома", "пять шариков дома")
        short = graph("пять шариков дома", "пять шариков дома", 100)
        mismatch = graph("пять шариков дома", "пять шириков дома")
        insertion = graph("пять шариков дома", "пять синих шариков дома")
        for candidate in (short, mismatch, insertion):
            self.assertIsNone(matching_windows(candidate, original, 1, 0))


    def test_untimed_insertions_inside_context_are_rejected(self):
        original = graph("шариков дома", "шариков дома", 900)
        candidate = word_graph("шариков дома", [
            {"text": "шариков", "offsets": {"from": 0, "to": 800}},
            {"text": "очень тихо", "offsets": {"from": 800, "to": 1000}},
            {"text": "дома", "offsets": {"from": 1000, "to": 1800}},
        ])
        self.assertIsNone(matching_windows(candidate, original, 0, 0))
