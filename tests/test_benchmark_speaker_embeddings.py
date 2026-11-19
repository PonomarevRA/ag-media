"""Protect benchmark evidence from false parity and vacuous matching claims."""

import math
from unittest import TestCase

from scripts.benchmark_speaker_embeddings import matching_decisions, validate_evidence, vector_parity
from voice_studio.speech import speaker_review


class SpeakerBenchmarkEvidenceTests(TestCase):
    """Check the independent numerical and production-rule evidence helpers."""

    def test_parity_rejects_dimensions_labels_and_nonfinite_values(self):
        """A truncated or nonfinite response cannot pass by comparing a prefix."""
        expected = {"1": [[1.0, 0.0]]}
        for actual in ({"1": [[1.0]]}, {"2": [[1.0, 0.0]]},
                       {"1": [[math.nan, 0.0]]}, {"1": []}):
            with self.subTest(actual=actual):
                report = vector_parity(expected, actual)
                self.assertFalse(report["exact"])
                self.assertFalse(report["shapesEqual"] and report["finite"])

    def test_small_finite_drift_is_visible_even_when_cosine_stays_high(self):
        """Direction similarity alone must not conceal changed embeddings."""
        report = vector_parity({"1": [[1.0, 0.0]]}, {"1": [[1.0, .0001]]})
        self.assertFalse(report["exact"])
        self.assertAlmostEqual(report["maxAbsoluteDifference"], .0001)
        self.assertGreater(report["minimumCosine"], .999)

    def test_matching_controls_exercise_positive_and_exclusion_decisions(self):
        """At least one positive rule and both safety exclusions are evaluated."""
        decisions = matching_decisions(speaker_review, {"1": [[1.0, 0.0]]})
        self.assertEqual(decisions["identical-reference-positive"], [
            {"from": 2, "to": 1, "voiceCosine": 1.0,
             "basis": "same-active-face-and-voice"}])
        self.assertEqual(decisions["identical-reference-different-face"], [])
        self.assertEqual(decisions["identical-reference-overlap"], [])

    def test_reported_failures_and_leaked_children_fail_the_command(self):
        """The benchmark must not return success after preserving negative evidence."""
        for failures, children in (([{"reason": "parity mismatch"}], []), ([], [123])):
            report = {"failures": failures,
                      "cleanup": {"resident": {"hostReturnCode": 0,
                                   "survivingObservedChildren": children, "samplerError": None}},
                      "idleRelease": {"restartParity": {"exact": True, "finite": True}}}
            with self.subTest(failures=failures, children=children):
                with self.assertRaises(RuntimeError):
                    validate_evidence(report)
