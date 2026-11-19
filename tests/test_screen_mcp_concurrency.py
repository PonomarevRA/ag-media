"""Real FastMCP dispatch keeps screen work independent, bounded, and cancel-safe."""

import asyncio
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from mcp.server.fastmcp import FastMCP

from voice_studio.core.mcp_lanes import AsyncWorkLane
from voice_studio.mcp import screen_mcp
from voice_studio.mcp.media_graph_mcp import GraphBuildQueue
from voice_studio.vision.video_revolver import VISUAL_SLOT


DRAFT = {'ocr': [], 'ocrStatuses': {'tesseract':'ok'}, 'interface':None,
         'structure':{'windows':[], 'lists':[], 'timelineClips':[]}, 'ocrJobs':1, 'mode':'overview'}


def structured(result):
    """Read the actual SDK's structured or text content without assuming a fake tool shape."""
    if isinstance(result, tuple):
        return result[1]
    return json.loads(next(block.text for block in result if block.type == 'text'))


class ScreenMCPConcurrencyTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Give each test one private UI lane, server and source file."""
        self.folder = TemporaryDirectory()
        self.source = Path(self.folder.name)/'screen.png'
        self.source.write_bytes(b'controlled-source-bytes')
        self.lane = AsyncWorkLane('test-screen-ui', workers=1, capacity=1)
        self.override = patch.object(screen_mcp, 'UI_LANE', self.lane)
        self.override.start()
        self.server = FastMCP('screen-concurrency-test')
        screen_mcp.register_screen_tools(self.server)

        @self.server.tool()
        async def responsiveness_probe() -> dict:
            """Prove that the MCP event loop responds while native work is blocked."""
            return {'status':'responsive'}

    async def asyncTearDown(self):
        """Close the owned admission lane after each test's controlled operations finish."""
        self.override.stop()
        self.lane.close()
        self.folder.cleanup()

    async def screen(self):
        """Call the registered screen tool through actual SDK argument validation."""
        return structured(await self.server.call_tool('screen_analyze',
            {'file_path':str(self.source), 'language':'en', 'mode':'overview'}))

    async def test_video_queue_and_native_slot_do_not_block_source_anchored_ui_result(self):
        """A real graph queue worker and video semaphore remain held while UI completes."""
        queue, started, release = GraphBuildQueue(), Event(), Event()

        def held_video():
            """Represent a controlled long analysis boundary without running another model."""
            started.set()
            if not release.wait(3):
                raise TimeoutError('Video boundary was not released.')
            return {'controlled':'video-finished'}

        job = queue.submit(held_video)
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        self.assertTrue(VISUAL_SLOT.acquire(blocking=False))
        try:
            with patch.object(screen_mcp, 'analyze_image_bytes', return_value=DRAFT):
                result = await asyncio.wait_for(self.screen(), 1)
            self.assertEqual(result['status'], 'draft')
            self.assertEqual(result['source']['sha256'], sha256(self.source.read_bytes()).hexdigest())
            self.assertEqual(queue.status(job['jobId'])['status'], 'running')
            self.assertFalse(release.is_set())
        finally:
            VISUAL_SLOT.release()
            release.set()
            queue.worker.shutdown(wait=True)

    async def test_ocr_work_does_not_block_mcp_event_loop_and_second_ui_call_is_busy(self):
        """Single admission rejects excess UI requests while allowing control tools to respond."""
        started, release = Event(), Event()

        def held_ocr(*args):
            """Hold one worker until the test verifies MCP responsiveness and admission bounds."""
            started.set()
            if not release.wait(3):
                raise TimeoutError('OCR boundary was not released.')
            return DRAFT

        with patch.object(screen_mcp, 'analyze_image_bytes', side_effect=held_ocr):
            request = asyncio.create_task(self.screen())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                ping = structured(await asyncio.wait_for(self.server.call_tool('responsiveness_probe', {}), .5))
                self.assertEqual(ping['status'], 'responsive')
                with self.assertRaisesRegex(Exception, 'занята|скриншот'):
                    await asyncio.wait_for(self.screen(), .5)
                self.assertFalse(request.done())
            finally:
                release.set()
                await asyncio.wait_for(request, 1)

    async def test_canceled_mcp_waiter_cannot_admit_another_running_ocr(self):
        """Caller cancellation does not release the still-running native operation's slot."""
        started, release, finished = Event(), Event(), Event()

        def held_ocr(*args):
            """Signal the actual worker's exit independently of the canceled waiter."""
            started.set()
            try:
                if not release.wait(3):
                    raise TimeoutError('OCR boundary was not released.')
                return DRAFT
            finally:
                finished.set()

        with patch.object(screen_mcp, 'analyze_image_bytes', side_effect=held_ocr):
            request = asyncio.create_task(self.screen())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
                with self.assertRaisesRegex(Exception, 'занята|скриншот'):
                    await self.screen()
            finally:
                release.set()
                self.assertTrue(await asyncio.to_thread(finished.wait, 1))
                # Synchronize on the actual future callback rather than a fixed sleep.
                await asyncio.to_thread(self.lane.executor.submit(lambda: None).result, 1)
            result = await asyncio.wait_for(self.screen(), 1)
            self.assertEqual(result['status'], 'draft')

    async def test_failure_releases_ui_lane_and_does_not_mutate_video_work(self):
        """A failed screen operation permits a subsequent request and leaves other lanes intact."""
        with patch.object(screen_mcp, 'analyze_image_bytes', side_effect=ValueError('controlled OCR failure')):
            with self.assertRaisesRegex(Exception, 'controlled OCR failure'):
                await self.screen()
        with patch.object(screen_mcp, 'analyze_image_bytes', return_value=DRAFT):
            result = await self.screen()
        self.assertEqual(result['status'], 'draft')

    async def test_blocked_transcription_http_mock_keeps_screen_tool_callable(self):
        """The full server offloads a held StudioClient operation independently of screen OCR."""
        from voice_studio.mcp.voice_mcp import StudioClient, build_server

        started, release = Event(), Event()
        studio = StudioClient(8766, Path(self.folder.name)/'output')

        def held_transcript(*args, **options):
            """Stand in for the blocking HTTP exchange while retaining its ordinary sync API."""
            started.set()
            if not release.wait(3):
                raise TimeoutError('Transcription HTTP boundary was not released.')
            return {'controlledVideoBoundary':True}

        with patch.object(studio, 'transcript', side_effect=held_transcript):
            server = build_server(studio, studio_root=Path(self.folder.name)/'studio',
                                  graph_root=Path(self.folder.name)/'graphs')
            transcript = asyncio.create_task(server.call_tool('media_transcribe',
                {'file_path':str(self.source), 'language':'en', 'visual':True, 'scene':True}))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                with patch.object(screen_mcp, 'analyze_image_bytes', return_value=DRAFT):
                    ui = structured(await asyncio.wait_for(server.call_tool('screen_analyze',
                        {'file_path':str(self.source), 'mode':'overview'}), 1))
                self.assertEqual(ui['status'], 'draft')
                self.assertFalse(transcript.done())
                self.assertFalse(release.is_set())
            finally:
                release.set()
                result = structured(await asyncio.wait_for(transcript, 1))
            self.assertTrue(result['controlledVideoBoundary'])

    async def test_scroll_uses_same_ui_admission_and_offloads_native_work(self):
        """Scroll parsing shares the single UI lane, rather than the long-video queue."""
        with patch.object(screen_mcp, 'analyze_scroll_screenshots', return_value={'status':'draft'}) as recognize:
            result = structured(await self.server.call_tool('screen_scroll',
                {'file_paths':[str(self.source),str(self.source)], 'timestamps_ms':[0,250]}))
        self.assertEqual(result['status'], 'draft')
        self.assertEqual(recognize.call_count, 1)
