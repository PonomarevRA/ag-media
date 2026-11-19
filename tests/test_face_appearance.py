"""Check agreement rules for optional visual attributes."""

from unittest import TestCase

from voice_studio.vision.face_appearance import describe_appearance


class FaceAppearanceTests(TestCase):
    def test_class_age_and_hair_are_present_only_for_repeatable_samples(self):
        first = {"face": 3, "timeMs": 100, "age": 23, "appearanceClass": "female",
                 "appearanceScore": .96, "hairRgb": [30, 20, 18]}
        next_sample = {**first, "timeMs": 400, "age": 27, "hairRgb": [32, 22, 20]}
        self.assertEqual(describe_appearance([first], {}), {})
        record = describe_appearance([first, next_sample], {})[3]
        self.assertEqual((record["age"], record["appearanceClass"], record["hairSwatch"]), (25, "female", "#1f1513"))
        conflicting = {**next_sample, "appearanceClass": "male", "age": 40, "hairRgb": [120, 110, 100]}
        self.assertEqual(describe_appearance([first, conflicting], {}), {})

    def test_extra_frames_fill_age_only_when_a_unique_majority_agrees(self):
        def samples(*ages):
            return [{"face": 3, "timeMs": index * 160, "age": age}
                    for index, age in enumerate(ages)]

        result = describe_appearance(samples(24, 42, 27, 25), {}, allow_age_consensus=True)
        self.assertEqual((result[3]["age"], result[3]["ageSamples"]), (25, 3))
        for ambiguous in (samples(22, 42), samples(20, 29, 38), samples(20, 25, 41, 43)):
            self.assertEqual(describe_appearance(ambiguous, {}, allow_age_consensus=True), {})

    def test_additional_samples_do_not_enter_clothing_matching(self):
        from voice_studio.vision.clothing_review import sample_requests

        track = [(index, (50, 30, 90, 85)) for index in range(30)]
        requests, _ = sample_requests([track], {1: 1}, 640, 360, appearance_faces=[1])
        self.assertEqual({at for at, rows in requests.items() if rows == [(1, 1)]}, {10, 20})
        self.assertEqual({at for at, rows in requests.items() if rows == [(1, 1, "appearance")]}, {5, 25})
