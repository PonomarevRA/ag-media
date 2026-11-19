"""Test conservative exact-pixel vetoes without asserting avatar or emotion accuracy."""

from copy import deepcopy
from unittest import TestCase, skipUnless
from importlib.util import find_spec
from unittest.mock import patch

HAS_VISION = find_spec("numpy") is not None and find_spec("cv2") is not None
if HAS_VISION:
    import numpy as np

from voice_studio.vision.static_face_evidence import (annotate_static_faces, face_crop_signature,
                                                     strong_av_anchor_eligible)
from voice_studio.vision.facial_expressions import finalize_expressions
if HAS_VISION:
    from voice_studio.vision.facial_expression_model import ExpressionClassifier


def face_frame():
    """Build deterministic textured pixels so crop jitter cannot look identical."""
    return np.random.default_rng(11).integers(0, 256, (100, 120, 3), dtype=np.uint8)


def descriptor():
    """Supply the bounded worker contract for pure evidence tests."""
    return {"version": "face-crop-rgb32-v1", "sha256": "a" * 64, "basis": "decoded-face-crop",
            "normalizedWidth": 32, "normalizedHeight": 32, "channels": 3,
            "comparison": "exact-normalized-pixels"}


def observations(signature, times=None, track=1, face=1):
    """Build a high-confidence raw model series with optional face-crop signatures."""
    times = list(range(0, 11000, 480)) if times is None else times
    result = []
    for time in times:
        row = {'timeMs': time, 'track': track, 'faceId': face, 'status': 'ok',
               'label': 'happy', 'confidence': .999, 'probabilities': {'happy': .999}}
        if signature is not None:
            row['faceCropSignature'] = signature
        result.append(row)
    return result


def visual(rows, duration=11000):
    """Create continuous geometry without deriving identity from static pixels."""
    return {'timeline': {'stepMs': 40, 'frames': [[[1, .1, .1, .3, .3, 1, 1]]
                                                for _ in range((duration + 39) // 40)]},
            'editorTimeline': {'durationMs': duration, 'shots': [{'startMs': 0}]},
            'facialExpressions': {'status': 'ok', 'samples': rows}}


class StaticFaceEvidenceTests(TestCase):
    """Static risk requires sustained repeated face pixels, never a UI ROI heuristic."""

    def test_identical_face_crop_is_only_suspected_static_or_frozen(self):
        signature = descriptor()
        rows = observations(signature)
        original = deepcopy(rows)
        annotated, guard = annotate_static_faces(rows, 11000)
        exclusion, = guard['exclusions']
        self.assertEqual((exclusion['startMs'], exclusion['endMs']), (0, 11000))
        self.assertGreaterEqual(exclusion['sampleCount'], 10)
        self.assertGreaterEqual(exclusion['spanMs'], 10000)
        self.assertEqual(exclusion['avatarClassification'], 'unknown')
        self.assertEqual(exclusion['livePersonPresence'], 'unknown')
        self.assertTrue(all(not row['reactionEligible'] for row in annotated))
        self.assertTrue(all(row['label'] == 'happy' for row in annotated))
        self.assertEqual(rows, original)
        layer = {'staticFaceEvidence': guard}
        self.assertFalse(strong_av_anchor_eligible(layer, 1, 5000))
        self.assertTrue(strong_av_anchor_eligible(layer, 2, 5000))
        self.assertTrue(strong_av_anchor_eligible(layer, 1, 11000))

    @skipUnless(HAS_VISION, "Requires the installed OpenCV vision runtime")
    def test_repeated_decoded_crop_produces_exact_static_signature(self):
        frame = face_frame()
        rows = observations(None)
        for row in rows:
            row['faceCropSignature'] = face_crop_signature(frame.copy(), [20, 20, 50, 50])
        _, guard = annotate_static_faces(rows, 11000)
        self.assertEqual(len(guard['exclusions']), 1)
        self.assertEqual(guard['samplesWithSignature'], len(rows))

    @skipUnless(HAS_VISION, "Requires the installed OpenCV vision runtime")
    def test_still_live_person_with_pixel_noise_is_not_excluded(self):
        frame = face_frame()
        rows = observations(None)
        for index, row in enumerate(rows):
            noisy = frame.copy()
            noisy[20:70, 20:70] = np.clip(noisy[20:70, 20:70].astype(np.int16) + index % 3, 0, 255)
            row['faceCropSignature'] = face_crop_signature(noisy, [20, 20, 50, 50])
        _, guard = annotate_static_faces(rows, 11000)
        self.assertEqual(guard['exclusions'], [])

    @skipUnless(HAS_VISION, "Requires the installed OpenCV vision runtime")
    def test_detection_crop_jitter_does_not_become_static_evidence(self):
        frame = face_frame()
        rows = observations(None)
        for index, row in enumerate(rows):
            row['faceCropSignature'] = face_crop_signature(frame, [20 + index % 2, 20, 50, 50])
        _, guard = annotate_static_faces(rows, 11000)
        self.assertEqual(guard['exclusions'], [])

    def test_too_short_sparse_missing_or_cut_evidence_abstains(self):
        signature = descriptor()
        variants = [observations(signature, range(0, 9000, 480)),
                    observations(signature, [0, 5000, 10000]), observations(None)]
        missing = observations(signature)
        missing[10].pop('faceCropSignature')
        variants.append(missing)
        for rows in variants:
            with self.subTest(rows=len(rows)):
                self.assertEqual(annotate_static_faces(rows, 11000)[1]['exclusions'], [])
        self.assertEqual(annotate_static_faces(observations(signature), 11000, cuts=[5000])[1]['exclusions'], [])

    def test_track_or_face_reassignment_does_not_join_short_runs(self):
        signature = descriptor()
        for field in ('track', 'faceId'):
            rows = observations(signature)
            for row in rows[12:]:
                row[field] = 2
            self.assertEqual(annotate_static_faces(rows, 11000)[1]['exclusions'], [])

    def test_raw_timeline_and_statistics_retained_but_reactions_unknown(self):
        rows = observations(descriptor())
        evidence = visual(rows)
        result = finalize_expressions(evidence, [{'startMs': 1000, 'endMs': 2000, 'text': 'Why?'}])
        self.assertEqual(result['statistics']['classifiedMs'], 11000)
        self.assertEqual(result['reactionStatistics']['classifiedMs'], 0)
        self.assertEqual(result['reactionStatistics']['unknownMs'], 11000)
        self.assertTrue(all(row['label'] == 'happy' and row['rawLabel'] == 'happy' for row in result['samples']))
        self.assertTrue(all(row['evidenceStatus'] == 'model-candidate' for row in result['intervals']))
        face, = result['reactions'][0]['faces']
        self.assertEqual(face['dominantLabel'], 'unknown')
        self.assertFalse(result['internalStateInference'])
        self.assertIn('not calibrated accuracy', result['interpretationRisk'])
        self.assertEqual(finalize_expressions(deepcopy(evidence)),
                         finalize_expressions(deepcopy(evidence)))

    def test_legacy_scores_and_static_ui_roi_do_not_establish_static_face(self):
        evidence = visual(observations(None))
        evidence['uiActivity'] = {'panels': [{'staticFraction': 1.0}]}
        layer = finalize_expressions(evidence)
        self.assertEqual(layer['staticFaceEvidence']['samplesWithSignature'], 0)
        self.assertEqual(layer['staticFaceEvidence']['exclusions'], [])
        self.assertGreater(layer['reactionStatistics']['classifiedMs'], 0)
        self.assertEqual(layer['verificationStatus'], 'unverified')

    def test_imported_signature_keeps_only_bounded_descriptor_fields(self):
        signature = {**descriptor(), 'untrustedExtraPixels': 'x' * 5000}
        layer = finalize_expressions(visual(observations(signature)))
        self.assertNotIn('untrustedExtraPixels', layer['samples'][0]['faceCropSignature'])
        self.assertLess(len(str(layer['samples'][0]['faceCropSignature'])), 400)

    def test_guard_budget_abstains_for_excess_tracks_or_samples(self):
        signature = descriptor()
        rows = observations(signature)
        with patch('voice_studio.vision.static_face_evidence.MAX_SAMPLES', 2):
            _, guard = annotate_static_faces(rows, 11000)
            self.assertEqual(guard['status'], 'partial')
            self.assertEqual(guard['exclusions'], [])
            self.assertEqual(guard['skippedSamples'], len(rows) - 2)
        multi = observations(signature, [0], 1, 1) + observations(signature, [480], 2, 2)
        with patch('voice_studio.vision.static_face_evidence.MAX_TRACKS', 1):
            _, guard = annotate_static_faces(multi, 11000)
            self.assertEqual(guard['skippedSamples'], 1)

    @skipUnless(HAS_VISION, "Requires the installed OpenCV vision runtime")
    def test_worker_reuses_decoded_face_crop_and_retains_low_confidence_winner(self):
        class Network:
            """Return deliberately weak scores while retaining the raw model winner."""
            def setInput(self, blob, name):
                self.blob = blob
            def forward(self, name):
                return np.array([0, 0, 0, .1, 0, 0, 0])
        classifier = ExpressionClassifier(None)
        classifier.network = Network()
        detection = [20, 20, 50, 50, 35, 35, 55, 35, 45, 45, 35, 60, 55, 60, .99]
        with patch('voice_studio.vision.facial_expression_model.aligned_blob', return_value=np.zeros((1, 3, 112, 112))):
            classifier.observe(0, face_frame(), detection, 1)
        row, = classifier.data({1: 9})['samples']
        self.assertEqual(row['rawLabel'], 'happy')
        self.assertEqual(row['label'], 'unknown')
        self.assertIn('faceCropSignature', row)
        self.assertEqual(row['faceId'], 9)
        self.assertIsNone(face_crop_signature(face_frame(), [-10, 20, 50, 50]))
        self.assertIsNone(face_crop_signature(face_frame(), [20, 20, 10, 10]))
