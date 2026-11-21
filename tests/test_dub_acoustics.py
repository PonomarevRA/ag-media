"""Exercise the bounded acoustic gates independently of synthesis models."""

from array import array
from math import pi, sin
from random import Random
import unittest

from voice_studio.synthesis.dub_acoustics import envelope, mouth_shift, phonation_profile, shift_in_slot
from voice_studio.synthesis.prosody_alignment import RATE


class AcousticCueTests(unittest.TestCase):
    def test_shift_does_not_discard_even_low_level_pcm(self):
        pcm = array("h", [1] + [0] * (RATE // 2)).tobytes()
        self.assertIsNone(shift_in_slot(pcm, -10))
        shifted = shift_in_slot(pcm, 10)
        self.assertEqual(len(shifted), len(pcm))
        self.assertEqual(array("h", shifted)[RATE // 100], 1)

    def test_mouth_shift_recovers_a_supported_delay_on_held_out_frames(self):
        raw = array("h", [0] * (2 * RATE))
        for first, last, gain in ((170, 310, 4500), (420, 590, 7500), (770, 970, 2600),
                                  (1100, 1330, 5200), (1510, 1740, 3500)):
            for index in range(first * 24, last * 24):
                raw[index] = round(gain * sin(2 * pi * 160 * index / RATE))
        original = raw.tobytes()
        late = shift_in_slot(original, 80)
        measured = envelope(original)
        mouth = [(ms, measured[ms // 10]) for ms in range(0, 2000, 40)]
        corrected, audit = mouth_shift(original, late, mouth)
        self.assertEqual(corrected, original)
        self.assertEqual(audit["delayMs"], -80)
        self.assertGreater(audit["after"], audit["before"])

    def test_periodicity_detects_difference_without_naming_a_voice_type(self):
        random = Random(12)
        periodic = [round(7500 * sin(2 * pi * 125 * index / RATE)) for index in range(RATE)]
        with_noise = [sample + random.randint(-7000, 7000) for sample in periodic]
        clear = phonation_profile(array("h", periodic).tobytes())
        noisy = phonation_profile(array("h", with_noise).tobytes())
        self.assertGreater(clear["hnrProxyDb"], noisy["hnrProxyDb"])
        self.assertAlmostEqual(clear["medianF0Hz"], 125)
        self.assertIsNone(phonation_profile(bytes(RATE * 2)))
