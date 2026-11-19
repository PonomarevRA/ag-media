"""Check region-specific color shares and low-contrast hard cut recovery."""

from io import BytesIO
from unittest import TestCase

from voice_studio.vision.shot_moodbook import HEIGHT, WIDTH, collect_colors, palette, summarize_shot
from voice_studio.vision.video_scene import scene_cuts


class ShotMoodbookTests(TestCase):
    def test_clothing_pixels_do_not_become_background(self):
        """Solid regions have exact colors and disjoint palette denominators."""
        red, blue = (185, 13, 7), (2, 25, 170)
        raw = b"".join(bytes(blue if x < WIDTH // 2 else red) for _ in range(HEIGHT) for x in range(WIDTH))
        box = (0, 0, WIDTH // 2, HEIGHT)
        whole, background, outfits = collect_colors(raw, {1: (box, box, True)})
        self.assertEqual({p["hex"]: p["share"] for p in palette(whole)}, {"#0219AA": .5, "#B90D07": .5})
        self.assertEqual(palette(background), [{"hex": "#B90D07", "share": 1}])
        self.assertEqual(palette(outfits[1]), [{"hex": "#0219AA", "share": 1}])

    def test_visibility_uses_the_face_clock_not_thumbnail_count(self):
        """A face present in two of four measured frames has 50% visibility."""
        face = [3, .4, .2, .1, .15, 0, 7]
        evidence = {"timeline": {"stepMs": 40, "frames": [[face], [face], [], []]}}
        raw = BytesIO(bytes((80, 100, 90)) * WIDTH * HEIGHT)
        shot = summarize_shot(0, 160, [{"timeMs": 40}], evidence, raw, 0)
        self.assertEqual(shot["sampleCount"], 1)
        self.assertEqual(shot["personSampleCount"], 4)
        self.assertEqual(shot["characters"][0]["visibilityShare"], .5)
        self.assertIsNone(shot["characters"][0]["frameAreaShare"])

    def test_relaxed_cut_requires_an_isolated_jump(self):
        """A gradual sequence of similar scores does not make separate shot cards."""
        self.assertEqual(scene_cuts([0, .01, .18, .02, .01, 0, .13, .18, .11, .01, 0, .38]), [2, 11])
