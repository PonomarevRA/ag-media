"""Exercise actual FastMCP dispatch at blocking Studio and source-render boundaries."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Timer
from time import perf_counter
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from mcp.server.fastmcp import FastMCP

from voice_studio.mcp import media_graph_mcp, voice_mcp


class MCPIoIsolationTests(IsolatedAsyncioTestCase):
    """Control worker boundaries without loading models or touching running services."""

    async def assert_off_loop(self, server, tool, arguments, entered, release):
        """Require control dispatch to progress before the held IO operation is released."""
        @server.tool()
        async def architecture_control_probe() -> dict:
            """Respond on the same event loop while another tool owns a worker slot."""
            return {"responsive": True}

        # The timer also releases the baseline's blocking call, so a regression fails
        # deterministically instead of hanging the entire test runner.
        timer = Timer(.4, release.set)
        timer.start()
        request = asyncio.create_task(server.call_tool(tool, arguments))
        started = perf_counter()
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            await asyncio.wait_for(server.call_tool("architecture_control_probe", {}), .2)
            self.assertFalse(release.is_set(), "Blocking IO prevented independent MCP dispatch")
            self.assertLess(perf_counter() - started, .3)
        finally:
            release.set()
            timer.cancel()
            await request
            timer.join()

    async def test_studio_http_does_not_block_other_tools(self):
        """A slow local HTTP response must not stall the MCP control channel."""
        entered, release = Event(), Event()
        with TemporaryDirectory() as folder:
            client = voice_mcp.StudioClient(1, Path(folder))

            def held_request(*args, **kwargs):
                """Represent a slow local response while retaining the original return type."""
                entered.set()
                release.wait(2)
                return {"sources": []}

            server = voice_mcp.build_server(client, studio_root=Path(folder) / "studio",
                                            graph_root=Path(folder) / "graphs")
            with patch.object(client, "request", side_effect=held_request):
                await self.assert_off_loop(server, "speech_sources", {}, entered, release)

    async def test_frame_render_does_not_block_other_tools(self):
        """PNG extraction and source verification must execute outside MCP dispatch."""
        entered, release = Event(), Event()
        with TemporaryDirectory() as folder:
            server = FastMCP("architecture-frame-test")
            media_graph_mcp.register_media_graph_tools(server, Path(folder) / "graphs", Path(folder))

            def held_render(*args, **kwargs):
                """Hold a source-render boundary without invoking FFmpeg or a model."""
                entered.set()
                release.wait(2)
                return b"controlled-png", {}

            with patch.object(media_graph_mcp.MediaGraphStore, "frame_context",
                              return_value=({"source": {}}, 0, {})), \
                    patch.object(media_graph_mcp, "checked_source", return_value=Path(folder) / "source"), \
                    patch("voice_studio.graph.media_graph_frame.render_frame", side_effect=held_render):
                await self.assert_off_loop(server, "media_graph_frame", {"snapshot_id": "fixture", "at_ms": 0},
                                           entered, release)

    async def test_render_capacity_is_retained_after_cancellation_while_reads_continue(self):
        """Canceled waiters cannot grow the render queue or occupy its separate read lane."""
        entered, release = Event(), Event()
        with TemporaryDirectory() as folder:
            server = FastMCP("architecture-capacity-test")
            resources = media_graph_mcp.register_media_graph_tools(server, Path(folder) / "graphs", Path(folder))

            def held_render(*args, **kwargs):
                """Retain one actual render slot until all cancellation assertions finish."""
                entered.set()
                release.wait(2)
                return b"controlled-png", {}

            arguments = {"snapshot_id": "fixture", "at_ms": 0}
            with patch.object(media_graph_mcp.MediaGraphStore, "frame_context",
                              return_value=({"source": {}}, 0, {})), \
                    patch.object(media_graph_mcp, "checked_source", return_value=Path(folder) / "source"), \
                    patch("voice_studio.graph.media_graph_frame.render_frame", side_effect=held_render):
                running = asyncio.create_task(server.call_tool("media_graph_frame", arguments))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    queued = asyncio.create_task(server.call_tool("media_graph_frame", arguments))
                    await asyncio.sleep(.01)
                    queued.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await queued
                    with self.assertRaisesRegex(Exception, "занята"):
                        await asyncio.wait_for(server.call_tool("media_graph_frame", arguments), .2)
                    # SQLite readers have a distinct bounded executor from source rendering.
                    await asyncio.wait_for(server.call_tool("media_graph_list", {}), .2)
                    self.assertFalse(release.is_set())
                finally:
                    release.set()
                    await running
                    for resource in resources:
                        resource.close()

    async def test_stdio_lifespan_retires_all_registered_queues_and_lanes(self):
        """Session cleanup closes exactly its owned resources, including on context failure."""
        with TemporaryDirectory() as folder:
            client = voice_mcp.StudioClient(1, Path(folder))
            server = voice_mcp.build_server(client, studio_root=Path(folder) / "studio",
                                            graph_root=Path(folder) / "graphs")
            with patch("voice_studio.core.mcp_lanes.AsyncWorkLane.close") as close_lane, \
                    patch.object(media_graph_mcp.GraphBuildQueue, "close") as close_queue:
                with self.assertRaisesRegex(RuntimeError, "controlled exit"):
                    async with server.settings.lifespan(server):
                        raise RuntimeError("controlled exit")
            # Studio3 + graph2 + voice-profile1 + mod1, and graph/audio/audit3 queues.
            self.assertEqual(close_lane.call_count, 7)
            self.assertEqual(close_queue.call_count, 3)

    async def test_one_cleanup_failure_does_not_abandon_other_owned_resources(self):
        """Unexpected queue-close errors still unwind every sibling resource callback."""
        with TemporaryDirectory() as folder:
            client = voice_mcp.StudioClient(1, Path(folder))
            server = voice_mcp.build_server(client, studio_root=Path(folder) / "studio",
                                            graph_root=Path(folder) / "graphs")
            with patch("voice_studio.core.mcp_lanes.AsyncWorkLane.close") as close_lane, \
                    patch.object(media_graph_mcp.GraphBuildQueue, "close", side_effect=RuntimeError("close failed")) as close_queue:
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    async with server.settings.lifespan(server):
                        pass
            self.assertEqual(close_lane.call_count, 7)
            self.assertEqual(close_queue.call_count, 3)
