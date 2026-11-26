"""Exercise lexical gating, ordered search and sample-accurate joins."""

from array import array
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
import wave

from voice_studio.prosody.reference_script import GUIDES, spoken_words
from voice_studio.prosody.reference_take_matching import find_takes, review_numbered_takes
from voice_studio.synthesis.prosody_alignment import RATE, write_pcm
from voice_studio.synthesis.voice_sequence import Take, assemble_sequence, choose_sequence, measure_candidates


def whisper_fixture(words, *, clock=0):
    """Build consistent word boundaries without an installed Whisper model."""
    rows = []
    for text in words:
        rows.append({"text": text, "offsets": {"from": clock, "to": clock + 160}})
        clock += 190
    return rows


class VoiceSequenceTests(TestCase):
    def test_number_is_optional_and_never_included_in_saved_bounds(self):
        first, second = [spoken_words(prompt.text) for prompt in GUIDES["ru"][:2]]
        data = whisper_fixture(["номер", "один", *first, *second])
        found = find_takes("ru", data, 10_000)
        self.assertEqual([(row["number"], row["basis"]) for row in found], [(1, "number-and-text"), (2, "text")])
        self.assertGreaterEqual(found[0]["startMs"], data[1]["offsets"]["to"])
        self.assertLessEqual(found[0]["endMs"], found[1]["startMs"])

    def test_conflicting_marker_and_repeated_match_are_not_saved(self):
        phrases = spoken_words(GUIDES["en"][0].text)
        self.assertEqual(find_takes("en", whisper_fixture(["number", "two", *phrases]), 10_000), [])
        self.assertEqual(find_takes("en", whisper_fixture([*phrases, *phrases]), 15_000), [])

    def test_digit_spellings_match_and_uncertain_words_require_a_unique_number(self):
        second, eighth = [spoken_words(GUIDES["en"][index].text) for index in (1, 7)]
        second[second.index("dog")] = "duck"
        eighth[eighth.index("two")] = "2"
        words = ["number", "2", *second, "number", "8", *eighth]
        segments = whisper_fixture(words)
        exact = find_takes("en", segments, 20_000)
        self.assertEqual([row["number"] for row in exact], [8])
        review = review_numbered_takes("en", segments, 20_000, exact)
        self.assertEqual([row["number"] for row in review], [2])
        self.assertIn("duck", review[0]["heard"])
        self.assertGreaterEqual(review[0]["startMs"], segments[1]["offsets"]["to"])
        self.assertLessEqual(review[0]["endMs"], segments[2 + len(second)]["offsets"]["from"])
        self.assertEqual(review_numbered_takes("en", whisper_fixture(second), 20_000, []), [])
        self.assertEqual(review_numbered_takes("en", whisper_fixture(words + ["number", "2", *second]),
                                               30_000, exact), [])

    def test_global_selection_picks_smoother_neighbor_on_equal_content(self):
        lead = Take(Path("lead"), 0, 0, -22, 130, [1., 0.])
        jagged = Take(Path("jagged"), 0, 0, -10, 160, [1., 0.])
        smooth = Take(Path("smooth"), 1, 0, -22, 130, [1., 0.])
        result, score = choose_sequence([[lead], [jagged, smooth]], [1., 0.])
        self.assertEqual(result, [lead, smooth])
        self.assertAlmostEqual(score, 0)

    def test_word_gate_discards_worse_lexical_candidate(self):
        with TemporaryDirectory() as raw:
            folder = Path(raw)
            def render(text, output, index, number):
                """Write a stable PCM control with enough active signal to score."""
                samples = array("h", [800, -800] * 18000)
                if sys.byteorder != "little":
                    samples.byteswap()
                write_pcm(output, samples.tobytes())
            def recognize(path):
                """Misrecognize only the first take of each sentence."""
                return "Bright stars lit night." if path.stem.endswith("-0") else "Bright stars lit the night."
            groups = measure_candidates([("Bright stars lit the night.", 280)] * 2, folder, render, recognize)
        self.assertEqual([[take.number for take in group] for group in groups], [[1], [1]])

    def test_assembly_preserves_samples_pauses_and_headroom(self):
        with TemporaryDirectory() as raw:
            folder = Path(raw)
            first, second = [folder / f"take-{index}.wav" for index in range(2)]
            write_pcm(first, (1000).to_bytes(2, "little", signed=True) * RATE)
            write_pcm(second, (30000).to_bytes(2, "little", signed=True) * RATE)
            selected = [Take(first, 0, 0, -33, None, [1., 0.]), Take(second, 1, 0, -4, None, [1., 0.])]
            output = folder / "joined.wav"
            rows = assemble_sequence(selected, [280, 0], output, [1., 0.])
            with wave.open(str(output)) as audio:
                self.assertEqual(audio.getnframes(), RATE * 2 + 6720)
                samples = array("h", audio.readframes(audio.getnframes()))
                if sys.byteorder != "little":
                    samples.byteswap()
            self.assertEqual(samples[RATE:RATE + 6720].tolist(), [0] * 6720)
            self.assertEqual((samples[0], samples[-1]), (0, 0))
            self.assertLessEqual(max(samples), 29200)
            self.assertEqual(rows[0]["gainDb"], 3.)
