"""Verify raw frame reuse preserves source identity, overlays and resource limits."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_graph_frame_cache import FrameCache
from voice_studio.graph import media_graph_frame as frame

PNG = frame.PNG_HEADER + b'\0\0\0\rIHDR' + struct.pack('>II', 320, 180)


class FrameCacheTests(TestCase):
    def test_limits_lru_and_palette_mutation(self):
        """
        Use small limits; expect LRU eviction, exact bytes and isolated palettes.

        No external inputs or errors are expected.
        """
        cache = FrameCache(max_entries=2, max_bytes=100)
        palette = {'colors': [1]}
        cache.put('a', b'a', palette)
        palette['colors'][0] = 9
        cached = cache.get('a')
        self.assertEqual(cached, (b'a', {'colors': [1]}))
        cached[1]['colors'][0] = 7
        self.assertEqual(cache.get('a')[1], {'colors': [1]})
        cache.put('b', b'b', None)
        cache.get('a')
        cache.put('c', b'c', None)
        self.assertIsNone(cache.get('b'))
        cache.put('oversize', b'x' * 101, None)
        self.assertIsNone(cache.get('oversize'))
        self.assertEqual(cache.bytes_used, 17)
        cache.put('d', b'x' * 90, None)
        self.assertLessEqual(cache.bytes_used, 100)
        self.assertLessEqual(len(cache.entries), 2)

    def test_concurrent_cache_operations_remain_bounded(self):
        """
        Concurrent writes/reads must keep entry/byte bounds and intact payloads.

        No external inputs; failures propagate through futures to the test.
        """
        cache = FrameCache(max_entries=8, max_bytes=800)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(cache.put, range(100), [b'x' * 100] * 100, [None] * 100))
        self.assertEqual(cache.bytes_used, 800)
        self.assertEqual(len(cache.entries), 8)
        self.assertEqual(sum(cache._size(item) for item in cache.entries.values()), 800)

    def test_concurrent_same_key_keeps_consistent_accounting(self):
        """
        Concurrent updates to one key must leave one intact entry.

        Inputs are detached palette fixtures; result is bounded state, no expected errors.
        """
        cache = FrameCache()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(cache.put, ['same'] * 100, [PNG] * 100, [{'colors': [1]}] * 100))
        self.assertEqual(len(cache.entries), 1)
        self.assertEqual(cache.get('same'), (PNG, {'colors': [1]}))
        self.assertEqual(cache.bytes_used, len(PNG) + 15)

    def test_invalid_limits_are_rejected(self):
        """
        Reject invalid resource limits.

        Input is zero max_entries; result is ValueError, with no external side effects.
        """
        with self.assertRaises(ValueError):
            FrameCache(max_entries=0)


class RenderCacheTests(TestCase):
    def setUp(self):
        """
        Create independent per-test rendering state.

        No external inputs; result is private source/cache, with cleanup registered.
        """
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source.mp4'
        self.source.write_bytes(b'original')
        self.cache = FrameCache()
        self.patcher = patch.object(frame, 'RAW_FRAMES', self.cache)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_repeated_pixels_palette_and_fresh_overlay(self):
        """
        Same source/time reuses decode but applies every current overlay.

        Inputs are private fixture pixels; result retains PNG/palette; no errors expected.
        """
        with patch.object(frame, 'extract_frame', return_value=PNG) as extract, \
             patch.object(frame, 'frame_colors', return_value={'colors': [1]}) as colors, \
             patch.object(frame, 'annotate_frame', return_value=PNG) as annotate:
            first = frame.render_frame(self.source, 10, include_palette=True)
            first[1]['colors'][0] = 9
            for caption in ('first', 'second'):
                actual = frame.render_frame(self.source, 10, include_palette=True, context={'caption': caption})
                self.assertEqual(actual, (PNG, {'colors': [1]}))
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(colors.call_count, 1)
            self.assertEqual([call.args[1] for call in annotate.call_args_list],
                             [{'caption': 'first'}, {'caption': 'second'}])
            self.source.write_bytes(b'replaced')
            frame.render_frame(self.source, 10)
            frame.render_frame(self.source, 11)
            frame.render_frame(self.source, 11, max_side=640)
            self.assertEqual(extract.call_count, 4)

    def test_failed_decode_and_source_change_do_not_enter_cache(self):
        """
        Failure or changing-source decode leaves no reusable entry.

        Fixture decoder raises ValueError then mutates the source; errors remain visible.
        """
        with patch.object(frame, 'extract_frame', side_effect=ValueError('decode failed')):
            with self.assertRaisesRegex(ValueError, 'decode failed'):
                frame.render_frame(self.source, 10)
        self.assertFalse(self.cache.entries)
        def changing_decode(*args):
            """
            Simulate a changing source during decode.

            Inputs args mirror extraction; result is fixture PNG, without external I/O.
            """
            self.source.write_bytes(b'changed!')
            return PNG
        with patch.object(frame, 'extract_frame', side_effect=changing_decode):
            frame.render_frame(self.source, 10)
        self.assertFalse(self.cache.entries)

    def test_warm_hits_keep_validation_busy_and_unavailable_errors(self):
        """
        Warm hits still validate inputs, slot capacity and source existence.

        Inputs use a private fixture; invalid/busy/unavailable calls raise ValueError.
        """
        with patch.object(frame, 'extract_frame', return_value=PNG):
            frame.render_frame(self.source, 10)
            with self.assertRaises(ValueError):
                frame.render_frame(self.source, 10, max_side=1)
            with patch.object(frame.FRAME_SLOT, 'acquire', return_value=False):
                with self.assertRaisesRegex(ValueError, 'busy'):
                    frame.render_frame(self.source, 10)
            self.source.unlink()
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                frame.render_frame(self.source, 10)
