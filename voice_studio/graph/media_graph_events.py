"""Detach cumulative graph reports before asynchronous immutable publication."""

from copy import deepcopy
from threading import Lock, RLock

from voice_studio.core.media_event_bus import payload_weight


class AsyncGraphPublisher:
    """Retain one successful snapshot per phase using the existing publisher.

    All graph callbacks share one ordering key. Priority can reorder independent
    object work, never cumulative graph states. Oversized reports use a drained
    inline callback instead of allocating another asynchronous report copy.
    """

    def __init__(self, bus, publish, ready):
        """Bind this adapter to one source job's consumer and readiness policy.
        """
        self.bus, self.publish, self.ready = bus, publish, ready
        self.lock, self.ingress = Lock(), RLock()
        self.pending, self.completed = set(), set()

    def __call__(self, phase, report):
        """Freeze only ready, not-yet-published phases after bounded admission.
        """
        with self.ingress:
            return self._publish(phase, report)

    def _publish(self, phase, report):
        """Serialize ingress while consumers use a separate phase-state lock.
        """
        if self.bus.mode == 'sync':
            future = self.bus.submit(lambda _payload: self.publish(phase, report),
                                     topic='graph-partial', ordering_key='graph')
            return future.result()
        with self.lock:
            if phase in self.pending or phase in self.completed or not self.ready(phase, report):
                return
            self.pending.add(phase)
        weight = payload_weight(report, self.bus.payload_budget)

        def project(payload):
            """Publish detached state and release a failed phase for a later retry.
            """
            succeeded = False
            try:
                succeeded = bool(self.publish(phase, payload))
                return succeeded
            finally:
                with self.lock:
                    self.pending.discard(phase)
                    if succeeded:
                        self.completed.add(phase)

        try:
            if weight > self.bus.payload_budget:
                future = self.bus.submit(lambda _payload: project(report), topic='graph-partial',
                                         ordering_key='graph', payload_bytes=weight)
                return future.result()
            return self.bus.submit(project, topic='graph-partial', ordering_key='graph',
                                   payload_factory=lambda: deepcopy(report), payload_bytes=weight)
        except BaseException:
            with self.lock:
                self.pending.discard(phase)
            raise
