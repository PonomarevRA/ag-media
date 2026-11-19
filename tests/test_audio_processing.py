"""Exercise matching, noise reduction and pause punctuation on synthetic audio."""

from pathlib import Path
import math
from array import array
from tempfile import TemporaryDirectory
import unittest
import wave

from voice_studio.audio.audio_processing import band_levels, bounded, process_wave, rms
from voice_studio.audio.pauses import annotate, quiet_spans


def fixture(path, sections):
    """Write one or more timed tones and pauses in mono PCM."""
    rate = 24000
    samples = array("h")
    for seconds, tones in sections:
        for i in range(round(seconds * rate)):
            value = sum(level * math.sin(2 * math.pi * hz * i / rate) for hz, level in tones)
            samples.append(round(value * 32767))
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", ""))
        writer.writeframes(samples.tobytes())


def relative_error(candidate, reference):
    """Measure three-band error after subtracting each file's overall level."""
    shift = rms(candidate) - rms(reference)
    return sum(abs(a - b - shift) for a, b in zip(band_levels(candidate), band_levels(reference)))


class AudioProcessingTests(unittest.TestCase):
    def test_auto_brings_level_and_spectrum_closer(self):
        with TemporaryDirectory() as directory:
            source, reference, output = (Path(directory) / name for name in ("s.wav", "r.wav", "o.wav"))
            fixture(source, [(1.3, [(220, .055), (1000, .11), (3800, .16)])])
            fixture(reference, [(1.3, [(220, .16), (1000, .17), (3800, .10)])])
            before = source.read_bytes()
            applied = process_wave(source, reference, output, {"auto": True})
            self.assertLess(relative_error(output, reference), relative_error(source, reference))
            self.assertLess(abs(rms(output) - rms(reference)), abs(rms(source) - rms(reference)))
            self.assertEqual(before, source.read_bytes())
            self.assertTrue(all(abs(gain) <= 6 for gain in applied["gains"]))

    def test_manual_limits_and_applies_individual_band(self):
        for value in (float("nan"), float("inf"), True, "1", 7):
            with self.assertRaises(ValueError):
                bounded(value, 6)
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.wav", Path(directory) / "out.wav"
            fixture(source, [(1, [(1000, .1)])])
            process_wave(source, None, output, {"gains": [0, 4, 0]})
            self.assertAlmostEqual(rms(output) - rms(source), 4, delta=.25)

    def test_quiet_spans_and_safe_suggestions(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "example.wav"
            tone = [(460, .3)]
            fixture(source, [(.4, []), (.6, tone), (.5, []), (.6, tone),
                             (1.1, []), (.6, tone), (.5, [])])
            spans = quiet_spans(source)
            self.assertEqual(spans, [{"start": 1.0, "end": 1.5}, {"start": 2.1, "end": 3.2}])
            segments = [
                {"text": "первая", "offsets": {"from": 400, "to": 1000}},
                {"text": "вторая", "offsets": {"from": 1500, "to": 2100}},
                {"text": "третья!", "offsets": {"from": 3200, "to": 3800}},
            ]
            self.assertEqual(annotate(segments, spans)["suggestion"], "первая, вторая. Третья!")
            segments[0]["text"] = "первая?"
            self.assertEqual(annotate(segments, spans)["suggestion"], "первая? вторая. Третья!")
            self.assertEqual(annotate(segments, [])["suggestion"], "первая? вторая третья!")

    def test_punctuation_at_exact_period_boundary(self):
        segments = [
            {"text": "один", "offsets": {"from": 200, "to": 1100}},
            {"text": "два", "offsets": {"from": 1900, "to": 2500}},
        ]
        self.assertEqual(annotate(segments, [{"start": 1.1, "end": 1.9}])["suggestion"], "один. Два")
