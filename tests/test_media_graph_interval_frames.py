"""
Verify real FFmpeg frame clocks, visible gaps and renderer resource boundaries.
"""

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import TestCase, mock, skipUnless

from voice_studio.graph import media_graph_interval_frames as grid


@skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the source-frame contract")
class IntervalFrameTests(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate bounded CFR, VFR and offset-clock fixtures in disposable scratch space.
        """
        cls.scratch = tempfile.TemporaryDirectory(prefix="vs207-tests-")
        cls.folder = Path(cls.scratch.name)
        cls.cfr = cls.folder / "cfr.mp4"
        cls.vfr = cls.folder / "vfr.mp4"
        cls.offset = cls.folder / "offset.mp4"
        cls.ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=4",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(cls.cfr)])
        cls.ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=5:duration=4",
                    "-vf", "select='eq(n,0)+eq(n,1)+eq(n,3)+eq(n,5)+eq(n,8)+eq(n,16)'",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "1",
                    "-video_track_timescale", "1000", str(cls.vfr)])
        cls.ffmpeg(["-i", str(cls.cfr), "-c", "copy", "-output_ts_offset", "5", str(cls.offset)])

    @classmethod
    def tearDownClass(cls):
        """
        Remove every generated source and output fixture.
        """
        cls.scratch.cleanup()

    @staticmethod
    def ffmpeg(arguments):
        """
        Run fixture and pixel-inspection commands without an executable shell.

        arguments is an FFmpeg argv list using local test files. Return stdout
        bytes; CalledProcessError or TimeoutExpired fails the invoking test.
        """
        return subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *arguments],
                              capture_output=True, check=True, timeout=10).stdout

    def test_cfr_preserves_order_duplicates_and_exact_pts(self):
        """
        Requests sharing a decoded frame remain separate, with a measured native clock.
        """
        png, result = grid.render_interval_grid(self.cfr, [
            {"atMs": 2050}, {"atMs": 1150}, {"atMs": 1150}, {"atMs": 1200}], end_ms=2500)
        self.assertEqual(result["returned"], 4)
        self.assertEqual([item["requestedMs"] for item in result["samples"]], [2050, 1150, 1150, 1200])
        self.assertEqual([item["actualMs"] for item in result["samples"]], [2100, 1200, 1200, 1200])
        for item in result["samples"]:
            actual = Fraction(item["sourceTimeBase"]) * item["sourcePts"] * 1000
            self.assertEqual(float(actual), item["actualMs"])
            self.assertEqual(item["actualMs"] - item["requestedMs"], item["deltaMs"])
        self.assertEqual(grid.image_dimensions(png), (960, 732))
        self.assertEqual(result["decodePasses"], 1)

    def test_sparse_seeks_retain_the_same_native_frame_contract(self):
        """
        Widely spaced requests avoid decoding their gaps while preserving order and duplicate times.
        """
        _, result = grid.render_interval_grid(self.cfr, [{"atMs": at} for at in (3550, 150, 150)], end_ms=3900)
        self.assertEqual(result["strategy"], "sparse-accurate-seeks")
        self.assertEqual(result["decodePasses"], 2)
        self.assertEqual([item["actualMs"] for item in result["samples"]], [3600, 200, 200])

    def test_dense_and_sparse_paths_select_identical_pts(self):
        """
        Adding a distant request changes strategy, without changing nearby VFR selections.
        """
        _, dense = grid.render_interval_grid(self.vfr, [{"atMs": 210}], end_ms=3600)
        _, sparse = grid.render_interval_grid(self.vfr, [{"atMs": 210}, {"atMs": 3000}], end_ms=3600)
        self.assertEqual(dense["strategy"], "dense-single-pass")
        self.assertEqual(sparse["strategy"], "sparse-accurate-seeks")
        for field in ("actualMs", "sourcePts", "sourceTimeBase"):
            self.assertEqual(dense["samples"][0][field], sparse["samples"][0][field])
        self.assertEqual(sparse["samples"][1]["actualMs"], 3200)

    def test_sparse_decodes_share_one_total_deadline(self):
        """
        An exhausted global budget stops further seeks rather than granting each a fresh timeout.
        """
        run = grid.subprocess.run
        with mock.patch.object(grid, "monotonic", side_effect=[0, 1, 21]):
            with mock.patch.object(grid.subprocess, "run", wraps=run) as process:
                with self.assertRaisesRegex(ValueError, "Could not render"):
                    grid.render_interval_grid(self.cfr, [{"atMs": 0}, {"atMs": 3000}], end_ms=3900)
        self.assertEqual(process.call_count, 1)

    def test_coincident_dense_requests_stop_near_the_last_request(self):
        """
        Close distinct requests must not decode a sixty-second tail waiting for more crossings.
        """
        samples = grid.validate_samples([{"atMs": at} for at in (0, 1, 2)], 60000, 480)
        with tempfile.TemporaryDirectory(prefix="vs207-dense-tail-") as scratch:
            folder = Path(scratch)
            source = folder / "sixty-seconds.mp4"
            self.ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=320x180:r=10:d=60",
                         "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", str(source)])
            decoded = grid.selected_frames(source, folder, samples, 60000, 480)
        self.assertEqual(len(decoded), 3)
        self.assertEqual([float(frame["time"]) for frame in decoded], [0, 100, 200])

    def test_all_caption_bands_match_standalone_ass_and_keep_picture_pixels(self):
        """
        Four packed pictures must not corrupt caption positions, Unicode or source pixels.
        """
        captions = ["ASR near request (whole cue): You can't " * 5,
                    "সাপ্লাই তে আসুন আপনি আপনার " * 6,
                    "東京の字幕と長い単語 " * 8,
                    r"{\pos(0,0)\p1} " + "Широкий контекст " * 10]
        samples = grid.validate_samples([{"atMs": 1000, "caption": caption[:240]}
                                        for caption in captions], 1400, 480)
        with tempfile.TemporaryDirectory(prefix="vs207-caption-bands-") as scratch:
            folder = Path(scratch)
            decoded = grid.selected_frames(self.cfr, folder, samples, 1400, 480)
            assignments = grid.assign_frames(samples, decoded, 1400)
            png, width, height = grid.compose_grid(folder, assignments, 480)
            target = folder / "grid.png"
            target.write_bytes(png)
            raw = self.ffmpeg(["-i", str(target), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            source = self.ffmpeg(["-i", str(decoded[0]["path"]), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            canvas = bytes((16, 20, 28, 255)) * width * height
            reference = subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pixel_format", "rgba",
                 "-video_size", f"{width}x{height}", "-i", "pipe:0", "-vf", "ass=captions.ass",
                 "-frames:v", "1", "-threads:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                input=canvas, cwd=folder, capture_output=True, check=True, timeout=10).stdout
            without_captions = [({**sample, "caption": ""}, frame) for sample, frame in assignments]
            blank_png, _, _ = grid.compose_grid(folder, without_captions, 480)
            blank_target = folder / "without-captions.png"
            blank_target.write_bytes(blank_png)
            blank_raw = self.ffmpeg(["-i", str(blank_target), "-frames:v", "1", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "pipe:1"])
            for row in range(2):
                for line in range(270, 314):
                    offset = ((row * 366 + line) * width) * 3
                    self.assertEqual(raw[offset:offset + width * 3], blank_raw[offset:offset + width * 3])
                for line in range(270, 366):
                    offset = ((row * 366 + line) * width) * 3
                    self.assertEqual(raw[offset:offset + width * 3], reference[offset:offset + width * 3])
                for column in range(2):
                    picture = bytearray()
                    for line in range(270):
                        offset = ((row * 366 + line) * width + column * 480) * 3
                        picture.extend(raw[offset:offset + 480 * 3])
                    self.assertEqual(picture, source)

    def test_vfr_does_not_infer_timestamps_from_nominal_frame_rate(self):
        """
        Uneven source PTS, rather than a fabricated regular frame cadence, drive selection.
        """
        _, result = grid.render_interval_grid(self.vfr, [{"atMs": value} for value in (210, 610, 1100)], end_ms=1800)
        self.assertEqual([item["actualMs"] for item in result["samples"]], [600, 1000, 1600])
        self.assertEqual([item["sourcePts"] for item in result["samples"]], [600, 1000, 1600])

    def test_nonzero_container_start_uses_normalized_source_clock(self):
        """
        A five-second container offset must not shift a 1.15-second graph request.
        """
        _, result = grid.render_interval_grid(self.offset, [{"atMs": 1150}], end_ms=1400)
        self.assertEqual(result["samples"][0]["actualMs"], 1200)
        self.assertEqual(result["clock"], "source-normalized-start-zero")

    def test_halfopen_end_never_substitutes_a_future_frame(self):
        """
        The next frame exactly at the exclusive end produces a visible missing tile.
        """
        png, result = grid.render_interval_grid(self.cfr, [{"atMs": 2050}], end_ms=2100)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["returned"], 0)
        self.assertIsNone(result["samples"][0]["actualMs"])
        self.assertEqual(grid.image_dimensions(png), (480, 366))

    def test_eof_and_mixed_available_frames_are_explicit(self):
        """
        A short clip may supply some requests while others remain visibly unknown.
        """
        _, result = grid.render_interval_grid(self.cfr, [{"atMs": 1000}, {"atMs": 3950}], end_ms=4500)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["returned"], 1)
        self.assertEqual(result["samples"][1]["status"], "unavailable")

    def test_caption_markup_stays_outside_source_pixels(self):
        """
        Untrusted captions cannot paint on the original source-image rectangle.
        """
        samples = grid.validate_samples([{"atMs": 1000, "caption": r"{\pos(0,0)\p1} m 0 0 l 999 0 999 999"}], 1400, 480)
        with tempfile.TemporaryDirectory(prefix="vs207-pixels-") as scratch:
            folder = Path(scratch)
            decoded = grid.selected_frames(self.cfr, folder, samples, 1400, 480)
            assignments = grid.assign_frames(samples, decoded, 1400)
            png, _, _ = grid.compose_grid(folder, assignments, 480)
            target = folder / "grid.png"
            target.write_bytes(png)
            source_pixels = self.ffmpeg(["-i", str(decoded[0]["path"]), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            grid_pixels = self.ffmpeg(["-i", str(target), "-vf", "crop=480:270:0:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            self.assertEqual(source_pixels, grid_pixels)

    def test_eight_frames_obey_maximum_pixel_bound(self):
        """
        The largest admitted grid stays within the declared dimensions.
        """
        png, result = grid.render_interval_grid(self.cfr, [{"atMs": value * 100} for value in range(8)], end_ms=1000, tile_width=640)
        self.assertEqual(grid.image_dimensions(png), (1920, 1368))
        self.assertLessEqual(result["width"] * result["height"], 1920 * 1368)

    def test_invalid_and_missing_input_fail_before_decode(self):
        """
        No arbitrary timestamps, captions, sample counts or paths reach FFmpeg.
        """
        invalid = [([], 1000, 480), ([{"atMs": True}], 1000, 480),
                   ([{"atMs": -1}], 1000, 480), ([{"atMs": 1000}], 1000, 480),
                   ([{"atMs": 0}], 60001, 480), ([{"atMs": 0}] * 9, 1000, 480),
                   ([{"atMs": 0, "caption": "x" * 241}], 1000, 480),
                   ([{"atMs": 0}], 1000, 800), ([{"atMs": 0}], 1000, 321)]
        with mock.patch.object(grid.subprocess, "run") as run:
            for samples, end, width in invalid:
                with self.subTest(samples=samples, end=end, width=width), self.assertRaises(ValueError):
                    grid.render_interval_grid(self.cfr, samples, end_ms=end, tile_width=width)
            with self.assertRaises(ValueError):
                grid.render_interval_grid(self.folder / "missing.mp4", [{"atMs": 0}], end_ms=1000)
            run.assert_not_called()

    def test_timeout_cleans_scratch_and_releases_shared_slot(self):
        """
        A failed decoder leaves neither temporary files nor a renderer lease behind.
        """
        created = []

        @contextmanager
        def temporary(**kwargs):
            """
            Observe the real temporary-directory lifetime while retaining automatic cleanup.

            kwargs accepts the renderer's prefix option without changing this
            test's private directory. Yield its path and remove it on exit.
            """
            with tempfile.TemporaryDirectory(prefix="vs207-timeout-") as scratch:
                created.append(Path(scratch))
                yield scratch

        with mock.patch.object(grid, "temporary_directory", temporary):
            with mock.patch.object(grid.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 20)):
                with self.assertRaisesRegex(ValueError, "Could not render"):
                    grid.render_interval_grid(self.cfr, [{"atMs": 0}], end_ms=1000)
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())
        self.assertTrue(grid.FRAME_SLOT.acquire(blocking=False))
        grid.FRAME_SLOT.release()

    def test_busy_renderer_is_rejected_without_starting_a_process(self):
        """
        Grid requests share the existing screenshot/audio admission bound.
        """
        self.assertTrue(grid.FRAME_SLOT.acquire(blocking=False))
        self.assertTrue(grid.FRAME_SLOT.acquire(blocking=False))
        try:
            with mock.patch.object(grid.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "busy"):
                    grid.render_interval_grid(self.cfr, [{"atMs": 0}], end_ms=1000)
                run.assert_not_called()
        finally:
            grid.FRAME_SLOT.release()
            grid.FRAME_SLOT.release()
