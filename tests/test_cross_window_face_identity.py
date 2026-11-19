"""Check source-local face identities across independent activity windows."""

from unittest import TestCase

from voice_studio.vision.video_activity_chunks import ActivityTimeline


def window(frames, references):
    """Build one worker response with private references from repeated observations."""
    return {
        "timeline": {"frames": frames},
        "editorTimeline": {"shots": [], "facePhotos": {}},
        "faceTracking": {"status": "ok"},
        "spans": [],
        "_faceSignatures": references,
    }


def hit(face, x=.1):
    """Keep raw track and local identity separate in a normalized detection row."""
    return [face + 100, x, .2, .2, .3, 1, face]


class CrossWindowFaceIdentityTests(TestCase):
    def test_return_after_absent_boundary_keeps_face_and_changes_track(self):
        """A person returning after an empty boundary retains the source identity."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 100 + [[]] * 150, {"1": [[1., 0.]]}), 0, 10_000, 0, {})
        timeline.add(window([[]] * 20 + [[hit(7)]] * 250, {"7": [[1., 0.]]}),
                     10_000, 20_000, 9200, {})
        first, returning = timeline.frames[0][0], timeline.frames[250][0]
        self.assertEqual(first[6], returning[6])
        self.assertNotEqual(first[0], returning[0])
        self.assertEqual(timeline.data(20_000)["faceTracking"]["faces"], 1)
        self.assertNotIn("_faceSignatures", timeline.data(20_000))

    def test_concurrent_lookalikes_cannot_collapse_to_previous_face(self):
        """Even identical signatures cannot merge separately visible people."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 100 + [[]] * 150, {"1": [[1., 0.]]}), 0, 10_000, 0, {})
        timeline.add(window([[]] * 20 + [[hit(7), hit(8, .6)]] * 250,
                            {"7": [[1., 0.]], "8": [[1., 0.]]}),
                     10_000, 20_000, 9200, {})
        self.assertEqual(len({row[6] for row in timeline.frames[250]}), 2)

    def test_ambiguous_return_and_missing_references_keep_independent_ids(self):
        """Weak or absent biometric evidence never guesses a prior identity."""
        for references in ({"7": [[1., 1.]]}, {}):
            with self.subTest(references=references):
                timeline = ActivityTimeline(20_000)
                timeline.add(window([[hit(1), hit(2, .6)]] * 100 + [[]] * 150,
                                    {"1": [[1., 0.]], "2": [[0., 1.]]}),
                             0, 10_000, 0, {})
                timeline.add(window([[]] * 20 + [[hit(7)]] * 250, references),
                             10_000, 20_000, 9200, {})
                prior = {row[6] for row in timeline.frames[0]}
                self.assertNotIn(timeline.frames[250][0][6], prior)

    def test_new_upload_starts_a_separate_identity_gallery(self):
        """Equal signatures from different sources cannot share source-local numbering."""
        first, second = ActivityTimeline(1000), ActivityTimeline(1000)
        first.add(window([[hit(1)]] * 25, {"1": [[1., 0.]]}), 0, 1000, 0, {})
        second.add(window([[hit(9)]] * 25, {"9": [[0., 1.]]}), 0, 1000, 0, {})
        self.assertEqual(first.frames[0][0][6], 1)
        self.assertEqual(second.frames[0][0][6], 1)

    def test_existing_overlap_mapping_blocks_a_concurrent_lookalike(self):
        """A new face cannot take an ID already occupied by an overlapping local face."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 250, {"1": [[1., 0.]]}), 0, 10_000, 0, {})
        timeline.add(window([[hit(7)]] * 20 + [[hit(7), hit(8, .6)]] * 250,
                            {"7": [[1., 0.]], "8": [[1., 0.]]}),
                     10_000, 20_000, 9200, {})
        self.assertEqual(timeline.frames[250][0][6], 1)
        self.assertNotEqual(timeline.frames[250][1][6], 1)

    def test_many_windows_continue_matching_after_prototype_capacity(self):
        """Long uploads retain a returning identity after more than twelve references."""
        timeline = ActivityTimeline(15_000)
        for index in range(15):
            start = index * 1000
            frames = ([[]] * 20 if index else []) + [[hit(index + 1)]] * 10 + [[]] * 15
            timeline.add(window(frames, {str(index + 1): [[1., index * .01]]}),
                         start, start + 1000, max(0, start - 800), {})
        self.assertEqual(timeline.data(15_000)["faceTracking"]["faces"], 1)
        self.assertEqual(timeline.frames[350][0][6], 1)

    def test_sequential_local_faces_cannot_bridge_a_shared_prior_prototype(self):
        """Each accepted return must agree with other returns accepted in that batch."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 100 + [[]] * 150, {"1": [[1., 0.]]}), 0, 10_000, 0, {})
        timeline.add(window([[]] * 20 + [[hit(7)]] * 100 + [[hit(8)]] * 150,
                            {"7": [[.84, .54]], "8": [[.84, -.54]]}),
                     10_000, 20_000, 9200, {})
        self.assertEqual(timeline.frames[250][0][6], 1)
        self.assertNotEqual(timeline.frames[350][0][6], 1)


class FaceGalleryGuardTests(TestCase):
    def test_blocked_best_match_does_not_fall_through_to_a_different_person(self):
        """A co-visible best match cannot force the next-best identity assignment."""
        from voice_studio.vision.face_reid import FaceIdentityGallery
        gallery = FaceIdentityGallery()
        gallery.remember({1: 1, 2: 2}, {1: [(1., 0.)], 2: [(.6, .8)]})
        matches = gallery.match({7: [(1., 0.)]}, {8: 1}, {(7, 8)})
        self.assertEqual(matches, {})

    def test_gallery_budget_preserves_new_observations_and_bounds_stored_references(self):
        """The oldest gallery entry can expire without dropping source observations."""
        from voice_studio.vision.face_reid import MAX_GALLERY_FACES, MAX_FACE_REFERENCES, FaceIdentityGallery, unit_vector
        gallery = FaceIdentityGallery()
        for face in range(1, MAX_GALLERY_FACES + 2):
            vectors = [unit_vector((1., offset * .01)) for offset in range(20)]
            gallery.remember({face: face}, {face: vectors})
        self.assertEqual(len(gallery.references), MAX_GALLERY_FACES)
        self.assertNotIn(1, gallery.references)
        self.assertIn(MAX_GALLERY_FACES + 1, gallery.references)
        self.assertTrue(all(len(vectors) <= MAX_FACE_REFERENCES for vectors in gallery.references.values()))

    def test_invalid_private_payload_is_ignored_instead_of_guessing_an_identity(self):
        """Nonfinite, empty, overflowed or malformed vectors cannot become references."""
        from voice_studio.vision.face_reid import read_face_references
        self.assertEqual(read_face_references(None), {})
        payload = {"bad": [[1., 0.]], "1": [[float('nan'), 0.]], "2": [[]],
                   "3": [[1e308, 1e308]], "4": [[0., 0.]], "5": [[1., 0.], [float('inf'), 1.]]}
        self.assertEqual(read_face_references(payload), {5: [(1., 0.)]})

    def test_single_observation_does_not_qualify_a_worker_reference(self):
        """Repeated qualified frames are required before identity transport is possible."""
        from voice_studio.vision.face_reid import face_references
        self.assertEqual(face_references({1: 1}, {1: [(1., 0.)]}), {})
        self.assertEqual(face_references({1: 1}, {1: [(1., 0.), (1., 0.)]}), {"1": [(1., 0.)]})


def measured(vector, quality):
    """Attach proxy-pixel side length times detector confidence to a private reference."""
    return [{"vector": vector, "quality": quality}]


class QualityQualifiedReferenceTests(TestCase):
    def test_weak_same_window_portrait_cannot_poison_a_strong_return(self):
        """Every detection remains visible while weak references leave the identity memory."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 100 + [[]] * 150, {"1": measured([1., 0.], 120)}),
                     0, 10_000, 0, {})
        second = window([[]] * 20 + [[hit(7)]] * 100 + [[hit(8)]] * 150,
                        {"7": measured([.55, .835], 30), "8": measured([.9, -.435], 110)})
        timeline.add(second, 10_000, 20_000, 9200, {})
        self.assertEqual(timeline.frames[250][0][6], 1)
        self.assertEqual(timeline.frames[350][0][6], 1)
        self.assertEqual(len([rows for rows in timeline.frames if rows]), 350)
        self.assertEqual(timeline.frames[250][0][1:6], hit(7)[1:6])
        self.assertNotIn("_faceSignatures", second)

    def test_stronger_return_retires_an_initial_small_reference_after_a_match(self):
        """A strong match can replace a tiny initial anchor across later windows."""
        timeline = ActivityTimeline(30_000)
        cases = [(1, [1., 0.], 30), (2, [.8, .6], 120), (3, [.28, .96], 110)]
        for index, (face, vector, quality) in enumerate(cases):
            start = index * 10_000
            frames = ([[]] * 20 if index else []) + [[hit(face)]] * 100 + [[]] * 150
            timeline.add(window(frames, {str(face): measured(vector, quality)}),
                         start, start + 10_000, max(0, start - 800), {})
        self.assertEqual(timeline.data(30_000)["faceTracking"]["faces"], 1)
        self.assertEqual(timeline.frames[500][0][6], 1)

    def test_higher_quality_does_not_override_a_face_nonmatch(self):
        """A larger new crop still needs the unchanged cosine acceptance threshold."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 100 + [[]] * 150, {"1": measured([1., 0.], 30)}),
                     0, 10_000, 0, {})
        timeline.add(window([[]] * 20 + [[hit(2)]] * 250, {"2": measured([.4, .9165], 120)}),
                     10_000, 20_000, 9200, {})
        self.assertEqual(timeline.data(20_000)["faceTracking"]["faces"], 2)

    def test_within_window_quality_filter_keeps_strong_returns_and_overlap_guard(self):
        """Quality excludes poor reference views without merging co-visible tracks."""
        from voice_studio.vision.face_reid import group_tracks, unit_vector
        tracks = [[(0, None), (8, None)], [(10, None), (18, None)], [(20, None), (28, None)]]
        vectors = [(1., 0.), (.55, .835), (.9, -.435)]
        samples = {index: [unit_vector(vector)] * 2 for index, vector in enumerate(vectors, 1)}
        aliases, _ = group_tracks(tracks, samples, qualities={1: 120, 2: 30, 3: 110})
        self.assertEqual(aliases, {1: 1, 2: 1, 3: 1})
        concurrent = [[(0, None), (18, None)], [(10, None), (18, None)], tracks[2]]
        aliases, _ = group_tracks(concurrent, samples, qualities={1: 120, 2: 30, 3: 110})
        self.assertNotEqual(aliases[1], aliases[2])

    def test_equal_quality_views_preserve_complete_retained_reference_guard(self):
        """The quality rule cannot weaken the existing equal-quality bridge rejection."""
        timeline = ActivityTimeline(20_000)
        timeline.add(window([[hit(1)]] * 100 + [[]] * 150, {"1": measured([1., 0.], 100)}),
                     0, 10_000, 0, {})
        timeline.add(window([[]] * 20 + [[hit(7)]] * 100 + [[hit(8)]] * 150,
                            {"7": measured([.84, .54], 100), "8": measured([.84, -.54], 100)}),
                     10_000, 20_000, 9200, {})
        self.assertNotEqual(timeline.frames[250][0][6], timeline.frames[350][0][6])
