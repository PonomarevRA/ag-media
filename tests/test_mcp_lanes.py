"""Prove event-loop isolation and finite admission under running and queued cancellation."""

import asyncio
from threading import Event
from unittest import IsolatedAsyncioTestCase

from voice_studio.core.mcp_lanes import AsyncWorkLane


class WorkLaneTests(IsolatedAsyncioTestCase):
    async def test_cancelling_queued_and_running_awaits_does_not_grow_queue(self):
        entered, release = Event(), Event()
        lane = AsyncWorkLane('test-lane', capacity=2)

        def blocked():
            entered.set()
            release.wait(3)
            return 'done'

        running = asyncio.create_task(lane.run(blocked))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            queued = asyncio.create_task(lane.run(lambda: 'queued'))
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            with self.assertRaisesRegex(ValueError, 'занята'):
                await lane.run(lambda: 'must-not-enter')
            # An independent UI lane remains available while the video lane is occupied.
            ui = AsyncWorkLane('test-ui')
            try:
                self.assertEqual(await asyncio.wait_for(ui.run(lambda: 'UI draft'), 1), 'UI draft')
            finally:
                ui.close()
        finally:
            release.set()
            lane.close()

    async def test_failed_operation_releases_its_slot(self):
        lane = AsyncWorkLane('test-errors')
        def fail():
            raise ValueError('expected')
        try:
            with self.assertRaisesRegex(ValueError, 'expected'):
                await lane.run(fail)
            self.assertEqual(await lane.run(lambda: 7), 7)
        finally:
            lane.close()
