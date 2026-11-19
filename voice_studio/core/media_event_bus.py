"""Bounded job-local scheduling for graph publication and object enrichment.

This pilot overlaps existing callbacks; it neither changes model parallelism nor
provides crash recovery. Payload accounting estimates retained Python containers,
not whole-process RSS or the transient memory used by graph construction.
"""

from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
import os
import sys
from threading import Condition, RLock, Thread, current_thread
from time import perf_counter

CURRENT_BUS = ContextVar('media_event_bus', default=None)
MODES = ('sync', 'fifo', 'priority')
PAYLOAD_BUDGET = 16 * 1024 * 1024
MAX_REPLAY_RECEIPTS = 64


@dataclass(frozen=True)
class EventEnvelope:
    """Identify one operation within its source-bound job and local sequence.
    """

    job_id: str
    source_sha256: str
    sequence: int
    topic: str
    priority: int
    ordering_key: str


def payload_weight(value, limit=PAYLOAD_BUDGET):
    """Estimate ordinary retained containers, stopping once the budget is exceeded.

    Unknown objects use the inline fallback. Shared references are counted once,
    matching deepcopy's memoization; no opaque object's custom sizing is invoked.
    """
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if type(item) not in (dict, list, tuple, str, int, float, bool, type(None)):
            return limit + 1
        total += sys.getsizeof(item)
        if total > limit:
            return total
        if type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return total


class MediaEventBus:
    """Execute independent work with bounded admission and one consumer owner.

    Capacity includes running work. Priority zero is object computation; priority
    one is graph publication. At most three high-priority operations overtake an
    eligible normal operation. Per-key ordering is never changed by priority.
    """

    def __init__(self, job_id, *, mode=None, capacity=3, payload_budget=PAYLOAD_BUDGET):
        """Capture a mode and start one consumer only for fifo/priority execution.

        Capacity counts running plus queued work. Payload budget is bytes of
        estimated retained containers. Invalid modes or nonpositive limits raise
        ValueError before a worker starts. The default mode is sync.
        """
        self.mode = os.environ.get('VOICE_MEDIA_EVENT_MODE', 'sync') if mode is None else mode
        if self.mode not in MODES:
            raise ValueError('VOICE_MEDIA_EVENT_MODE must be sync, fifo or priority.')
        if capacity < 1 or payload_budget < 1:
            raise ValueError('Media event capacity and payload budget must be positive.')
        self.job_id, self.source_sha256 = job_id, ''
        self.capacity, self.payload_budget = capacity, payload_budget
        self.condition, self.producers, self.owner = Condition(), RLock(), RLock()
        self.pending = deque()
        self.reserving_key = None
        self.outstanding = self.retained_bytes = self.sequence = self.high_burst = 0
        self.closed, self.finishing, self.finished, self.first_error = False, False, False, None
        self.finisher = None
        self.receipts = {}
        self.counts = {'accepted': 0, 'completed': 0, 'failed': 0, 'inlineOversized': 0,
                       'backlogHighWater': 0, 'retainedBytesHighWater': 0,
                       'producerSeconds': 0.0, 'queueSeconds': 0.0,
                       'handlerSeconds': 0.0, 'drainSeconds': 0.0,
                       'partialUnpublished': 0, 'topics': {}}
        self.worker = None
        if self.mode != 'sync':
            self.worker = Thread(target=self._consume, name='media-events-' + str(job_id)[:8])
            self.worker.start()

    def bind_source(self, source_sha256):
        """Fence this dispatcher to one already verified original source digest.
        """
        if not isinstance(source_sha256, str) or not source_sha256:
            raise ValueError('A media event job requires its original source digest.')
        with self.condition:
            if self.source_sha256 and self.source_sha256 != source_sha256:
                raise ValueError('A media event job cannot change its source.')
            self.source_sha256 = source_sha256

    def metrics(self):
        """Return detached diagnostics without callback payloads or model objects.

        Timing fields are monotonic seconds summed across operations, not CPU
        utilization. Retention fields estimate frozen reports, not process RSS.
        """
        with self.condition:
            result = dict(self.counts)
            result['topics'] = {key: dict(value) for key, value in self.counts['topics'].items()}
            result.update(mode=self.mode, jobId=self.job_id, sourceSha256=self.source_sha256,
                          outstanding=self.outstanding, retainedBytes=self.retained_bytes,
                          capacity=self.capacity, payloadBudgetBytes=self.payload_budget,
                          replayReceipts=len(self.receipts), finishing=self.finishing, finished=self.finished,
                          payloadAccounting='estimated-python-containers')
            return result

    def submit(self, operation, *, topic, priority=1, ordering_key=None,
               payload_factory=None, payload_bytes=0, event=None, force_inline=False):
        """Admit one callback, freezing its payload only after space is reserved.

        A consumer's nested submission executes inline to prevent self-join. The
        returned Future preserves callback errors; drain also exposes the first
        uncaught callback error. Recoverable partial publication errors remain
        the publication adapter's responsibility.

        Accepted Futures cannot be cancelled. Explicit envelopes are fenced to
        this job/source and deduplicated by sequence, retaining at most 64 replay
        receipts until close. Full admission blocks; high priority cannot bypass
        that bound or preempt an already running callback.
        """
        if priority not in (0, 1) or payload_bytes < 0:
            raise ValueError('Media event priority or payload size is invalid.')
        if event is not None:
            if (not isinstance(event, EventEnvelope) or event.job_id != self.job_id
                    or event.source_sha256 != self.source_sha256 or event.sequence < 1
                    or event.priority != priority or event.topic != topic
                    or event.ordering_key != (ordering_key or topic)):
                raise ValueError('Media event does not belong to this job and source.')
        started = perf_counter()
        # A producer can be waiting for capacity while a consumer calls a nested
        # enrichment. That nested call must bypass the producer admission lock.
        from contextlib import nullcontext
        admission = nullcontext() if current_thread() is self.worker else self.producers
        with admission:
            with self.condition:
                if self.closed:
                    raise RuntimeError('Media event bus is closed.')
                nested_consumer = current_thread() is self.worker
                if nested_consumer and event is not None:
                    raise ValueError('Nested media work cannot introduce an explicit replay envelope.')
                key = ordering_key or topic
                if nested_consumer and (self.reserving_key == key or any(
                        item['event'].ordering_key == key for item in self.pending)):
                    raise RuntimeError('Nested media work cannot overtake an earlier event with the same key.')
                terminal = force_inline and current_thread() is self.finisher
                if self.finished or (self.finishing and not nested_consumer and not terminal):
                    raise RuntimeError('Media event job is finishing; admission is sealed.')
                if not self.source_sha256:
                    raise ValueError('Bind the original source before submitting media events.')
                if event is not None and event.sequence in self.receipts:
                    previous, future = self.receipts[event.sequence]
                    if previous != event:
                        raise ValueError('Media event sequence already identifies different work.')
                    return future
                if event is not None and event.sequence <= self.sequence:
                    raise ValueError('Media event sequence has already been consumed.')
                if event is not None and len(self.receipts) >= MAX_REPLAY_RECEIPTS:
                    raise ValueError('Media event replay receipt capacity is full.')
            inline = self.mode == 'sync' or current_thread() is self.worker or force_inline
            oversized = payload_bytes > self.payload_budget
            if (oversized or force_inline) and self.mode != 'sync' and current_thread() is not self.worker:
                self.drain()
                inline = True
            if inline:
                payload = payload_factory() if payload_factory else None
                item = self._item(operation, topic, priority, ordering_key, payload, 0, event)
                future = item['future']
                with self.condition:
                    self.counts['inlineOversized'] += int(oversized)
                self._execute(item)
            else:
                with self.condition:
                    while self.outstanding >= self.capacity or self.retained_bytes + payload_bytes > self.payload_budget:
                        self.condition.wait()
                    self.outstanding += 1
                    self.retained_bytes += payload_bytes
                    self.reserving_key = ordering_key or topic
                try:
                    payload = payload_factory() if payload_factory else None
                    item = self._item(operation, topic, priority, ordering_key, payload, payload_bytes, event)
                    future = item['future']
                except BaseException:
                    with self.condition:
                        self.outstanding -= 1
                        self.retained_bytes -= payload_bytes
                        self.reserving_key = None
                        self.condition.notify_all()
                    raise
                with self.condition:
                    self.pending.append(item)
                    self.reserving_key = None
                    self.counts['backlogHighWater'] = max(self.counts['backlogHighWater'], self.outstanding)
                    self.counts['retainedBytesHighWater'] = max(self.counts['retainedBytesHighWater'], self.retained_bytes)
                    self.condition.notify_all()
            with self.condition:
                self.counts['producerSeconds'] += perf_counter() - started
            return future

    def _item(self, operation, topic, priority, ordering_key, payload, weight, envelope=None):
        """Capture a callback's immutable identity and current execution context.
        """
        with self.condition:
            self.sequence = max(self.sequence + 1, envelope.sequence if envelope else 0)
            self.counts['accepted'] += 1
            explicit = envelope is not None
            envelope = envelope or EventEnvelope(self.job_id, self.source_sha256, self.sequence,
                                                 topic, priority, ordering_key or topic)
            future = Future()
            # Accepted evidence cannot be revoked by cancelling a Future. Keep
            # replay receipts only when an explicit external envelope requests it.
            future.set_running_or_notify_cancel()
            if explicit:
                self.receipts[envelope.sequence] = envelope, future
        return {'event': envelope, 'operation': operation, 'payload': payload, 'weight': weight,
                'context': copy_context(), 'future': future, 'queued': perf_counter()}

    def _next(self):
        """Choose an eligible key head, applying a bounded high-priority burst.
        """
        if self.mode == 'fifo':
            return self.pending.popleft()
        keys, eligible = set(), []
        for item in self.pending:
            key = item['event'].ordering_key
            if key not in keys:
                eligible.append(item)
                keys.add(key)
        normal = next((item for item in eligible if item['event'].priority == 1), None)
        high = next((item for item in eligible if item['event'].priority == 0), None)
        if high is not None and (normal is None or self.high_burst < 3):
            selected = high
            self.high_burst += 1
        else:
            selected = normal
            self.high_burst = 0
        self.pending.remove(selected)
        return selected

    def _consume(self):
        """Own queued callback execution until close has drained admitted work.
        """
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if not self.pending:
                    return
                item = self._next()
            try:
                self._execute(item)
            finally:
                weight = item['weight']
                # Release frozen payloads and copied contexts before advertising
                # available capacity; an idle consumer must retain no last report.
                item.clear()
                with self.condition:
                    self.outstanding -= 1
                    self.retained_bytes -= weight
                    self.condition.notify_all()

    def _execute(self, item):
        """Run one callback under the shared owner lock and retain its outcome.
        """
        started = perf_counter()
        failed = False
        try:
            with self.owner:
                result = item['context'].run(item['operation'], item['payload'])
            if item['event'].topic == 'graph-partial' and not result:
                with self.condition:
                    self.counts['partialUnpublished'] += 1
            item['future'].set_result(result)
        except BaseException as error:
            failed = True
            item['future'].set_exception(error)
            with self.condition:
                if self.first_error is None:
                    self.first_error = error
        finally:
            with self.condition:
                self.counts['failed' if failed else 'completed'] += 1
                self.counts['queueSeconds'] += started - item['queued']
                elapsed = perf_counter() - started
                self.counts['handlerSeconds'] += elapsed
                topic = self.counts['topics'].setdefault(item['event'].topic, {'count': 0, 'seconds': 0.0})
                topic['count'] += 1
                topic['seconds'] += elapsed

    def drain(self, *, propagate=True):
        """Wait for all admitted work; a consumer must never join its own queue.
        """
        if current_thread() is self.worker:
            raise RuntimeError('A media event consumer cannot drain itself.')
        started = perf_counter()
        with self.condition:
            while self.outstanding:
                self.condition.wait()
            self.counts['drainSeconds'] += perf_counter() - started
            error = self.first_error
        if propagate and error is not None:
            raise error

    def finish(self, operation):
        """Drain all accepted work and run final publication under the same owner.

        Holding admission across the barrier prevents a concurrent producer from
        inserting another partial snapshot between the barrier and final output.
        External admission stays sealed afterward. Callback failures propagate;
        repeated finish or a late submit raises RuntimeError.
        """
        if current_thread() is self.worker:
            raise RuntimeError('A media event consumer cannot finish itself.')
        with self.producers:
            with self.condition:
                if self.finishing or self.closed:
                    raise RuntimeError('Media event completion has already started.')
                self.finishing = True
                self.finisher = current_thread()
            try:
                return self.submit(lambda _payload: operation(), topic='graph-complete',
                                   ordering_key='graph', force_inline=True).result()
            finally:
                with self.condition:
                    self.finished = True

    def close(self, *, propagate=True):
        """Stop admission, drain callbacks, and join the job-owned consumer thread.

        Callback runtimes retain their existing limits. Python threads cannot
        forcibly interrupt a stuck native callback; no hard shutdown bound is claimed.
        """
        if current_thread() is self.worker:
            raise RuntimeError('A media event consumer cannot close itself.')
        with self.producers:
            try:
                self.drain(propagate=propagate)
            finally:
                with self.condition:
                    self.closed = True
                    self.condition.notify_all()
                if self.worker is not None:
                    self.worker.join()
                with self.condition:
                    self.receipts.clear()


@contextmanager
def media_event_scope(bus):
    """Share a dispatcher through nested stages without changing model budgets.
    """
    token = CURRENT_BUS.set(bus)
    try:
        yield bus
    except BaseException:
        bus.close(propagate=False)
        raise
    else:
        bus.close()
    finally:
        CURRENT_BUS.reset(token)


def run_enrichment(operation):
    """Run existing object computation on the current bus and await its result.

    Cache and report mutations remain in the caller. Without a graph-job scope,
    including Studio browser analysis, the operation keeps its direct behavior.
    """
    bus = CURRENT_BUS.get()
    if bus is None:
        return operation()
    future = bus.submit(lambda _payload: operation(), topic='object-enrichment',
                        priority=0, ordering_key='object-review')
    return future.result()
