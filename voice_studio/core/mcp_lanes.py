"""Bound blocking MCP operations without blocking the server event loop or other media lanes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from threading import BoundedSemaphore


class AsyncWorkLane:
    """Own a small independent executor; cancellation never releases a still-running slot."""

    def __init__(self, name, *, workers=1, capacity=1):
        """Bound running plus queued jobs; each domain chooses its own independent lane."""
        if not 1 <= workers <= capacity <= 8:
            raise ValueError('Invalid worker-lane budget')
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.slots = BoundedSemaphore(capacity)

    async def run(self, operation, *args, **kwargs):
        """Run one blocking operation off-loop or fail immediately when this lane is full."""
        if not self.slots.acquire(blocking=False):
            raise ValueError('Очередь этого инструмента занята. Другие инструменты доступны.')
        try:
            future = self.executor.submit(partial(operation, *args, **kwargs))
        except BaseException:
            self.slots.release()
            raise
        # asyncio cancellation cannot stop native work already running in the thread.
        future.add_done_callback(lambda completed: self.slots.release())
        wrapped = asyncio.wrap_future(future)
        # Keep queued entries admitted until the executor physically drains them too.
        # Otherwise cancelling queued awaits could fill its private unbounded work queue.
        wrapped.add_done_callback(lambda completed: None if completed.cancelled() else completed.exception())
        return await asyncio.shield(wrapped)

    def offload(self, operation):
        """Preserve a synchronous tool's schema while dispatching it through this bounded lane."""
        @wraps(operation)
        async def run(*args, **kwargs):
            """Keep blocking IO off the MCP event loop without changing its result or errors."""
            return await self.run(operation, *args, **kwargs)
        return run

    def close(self):
        """Cancel queued work during shutdown; running native operations retain their own deadlines."""
        self.executor.shutdown(wait=False, cancel_futures=True)
