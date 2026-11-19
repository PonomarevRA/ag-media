"""Check scheduling, ownership and completion contracts without loading models."""

from contextvars import ContextVar
from copy import deepcopy
from threading import Condition, Event, Thread, current_thread
from unittest import TestCase
from unittest.mock import patch
import weakref

from voice_studio.core.media_event_bus import (EventEnvelope, MediaEventBus,
                                               media_event_scope, payload_weight, run_enrichment)
from voice_studio.graph.media_graph_events import AsyncGraphPublisher


class MediaEventBusTests(TestCase):
    def bus(self, mode='priority', **settings):
        """Create a source-fenced dispatcher with cleanup even after an assertion.
        """
        bus = MediaEventBus('job', mode=mode, **settings)
        bus.bind_source('source')
        self.addCleanup(bus.close, propagate=False)
        return bus

    def test_priority_reorders_only_independent_accepted_work(self):
        """A high object task passes queued graph work, but never a running graph.
        """
        for mode, expected in (('fifo', ['running', 'graph', 'object']),
                               ('priority', ['running', 'object', 'graph'])):
            with self.subTest(mode=mode):
                bus = self.bus(mode, capacity=3)
                entered, release, order = Event(), Event(), []
                def first(_payload):
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('Test did not release the consumer')
                    order.append('running')
                bus.submit(first, topic='graph', ordering_key='graph')
                self.assertTrue(entered.wait(3))
                bus.submit(lambda _: order.append('graph'), topic='graph', ordering_key='graph')
                bus.submit(lambda _: order.append('object'), topic='object', priority=0)
                release.set()
                bus.drain()
                self.assertEqual(order, expected)

    def test_fairness_and_per_key_fifo(self):
        """Serve an eligible normal after three high tasks; preserve a key's head.
        """
        bus = self.bus(capacity=9)
        entered, release, order = Event(), Event(), []
        def gate(_payload):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release the consumer')
        bus.submit(gate, topic='gate')
        self.assertTrue(entered.wait(3))
        bus.submit(lambda _: order.append('normal'), topic='normal')
        for index in range(5):
            bus.submit(lambda _, value=index: order.append(value), topic='high', priority=0)
        release.set()
        bus.drain()
        self.assertEqual(order, [0, 1, 2, 'normal', 3, 4])
        bus.submit(lambda _: order.append('head'), topic='same', ordering_key='same')
        bus.submit(lambda _: order.append('tail'), topic='same', ordering_key='same', priority=0)
        bus.drain()
        self.assertEqual(order[-2:], ['head', 'tail'])

    def test_capacity_reserves_before_payload_freeze(self):
        """A full queue blocks admission before another frozen payload is created.
        """
        bus = self.bus(capacity=1)
        entered, release, factory, returned = Event(), Event(), Event(), Event()
        def gate(_payload):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release the consumer')
        bus.submit(gate, topic='gate')
        self.assertTrue(entered.wait(3))
        def producer():
            bus.submit(lambda payload: payload, topic='next', payload_factory=lambda: factory.set())
            returned.set()
        producer_thread = Thread(target=producer)
        producer_thread.start()
        self.assertFalse(factory.wait(.05))
        self.assertFalse(returned.is_set())
        release.set()
        producer_thread.join(3)
        self.assertFalse(producer_thread.is_alive())
        bus.drain()
        self.assertTrue(factory.is_set())
        self.assertEqual(bus.metrics()['backlogHighWater'], 1)

    def test_oversized_report_drains_then_runs_inline(self):
        """Oversized reports keep their facts without an additional queued copy.
        """
        bus = self.bus(payload_budget=100)
        entered, release, order = Event(), Event(), []
        def first(_payload):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release the consumer')
            order.append('queued')
        bus.submit(first, topic='graph')
        self.assertTrue(entered.wait(3))
        done = Event()
        def producer():
            bus.submit(lambda _: order.append('inline'), topic='graph', payload_bytes=101).result()
            done.set()
        producer_thread = Thread(target=producer)
        producer_thread.start()
        self.assertFalse(done.wait(.05))
        release.set()
        producer_thread.join(3)
        self.assertEqual(order, ['queued', 'inline'])
        self.assertEqual(bus.metrics()['inlineOversized'], 1)
        self.assertLessEqual(bus.metrics()['retainedBytesHighWater'], 100)

    def test_payloads_detach_and_final_barrier_preserves_order(self):
        """Later producer mutations cannot alter an accepted cumulative report.
        """
        bus = self.bus()
        entered, release, seen = Event(), Event(), []
        def gate(_payload):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release the consumer')
        bus.submit(gate, topic='gate')
        self.assertTrue(entered.wait(3))
        publish = AsyncGraphPublisher(bus, lambda phase, report: seen.append(deepcopy(report)) or True,
                                      lambda phase, report: True)
        report = {'segments': [{'text': 'measured'}]}
        publish('base', report)
        report['segments'][0]['text'] = 'later edit'
        release.set()
        result = bus.finish(lambda: seen.append('complete') or 'final')
        self.assertEqual(result, 'final')
        self.assertEqual(seen, [{'segments': [{'text': 'measured'}]}, 'complete'])

    def test_failed_phase_can_retry_and_duplicates_do_not_enqueue(self):
        """Only successful publication seals a phase; pending duplicates are bounded.
        """
        bus = self.bus()
        results = iter((False, True))
        seen = []
        def consume(phase, report):
            seen.append(phase)
            return next(results)
        publish = AsyncGraphPublisher(bus, consume, lambda phase, report: True)
        publish('base', {})
        bus.drain()
        publish('base', {})
        bus.drain()
        publish('base', {})
        bus.drain()
        self.assertEqual(seen, ['base', 'base'])

    def test_failures_propagate_after_other_accepted_events_drain(self):
        """One failed callback cannot strand an already accepted later callback.
        """
        bus, completed = self.bus(), Event()
        def fail(_payload):
            raise ValueError('expected callback failure')
        failed = bus.submit(fail, topic='failure')
        bus.submit(lambda _: completed.set(), topic='after')
        with self.assertRaisesRegex(ValueError, 'expected callback failure'):
            bus.drain()
        self.assertTrue(completed.is_set())
        with self.assertRaises(ValueError):
            failed.result()
        self.assertEqual(bus.metrics()['failed'], 1)

    def test_context_propagates_and_nested_work_does_not_deadlock(self):
        """Copy the actual run context and execute a consumer's nested call inline.
        """
        bus = self.bus()
        setting = ContextVar('test-media-provider', default='unset')
        with media_event_scope(bus):
            token = setting.set('frozen-provider')
            try:
                future = bus.submit(lambda _: run_enrichment(lambda: setting.get()), topic='outer')
            finally:
                setting.reset(token)
            self.assertEqual(future.result(3), 'frozen-provider')

    def test_nested_consumer_bypasses_blocked_producer_and_close(self):
        """No producer-held admission lock may prevent a consumer from finishing.
        """
        for closing in (False, True):
            with self.subTest(closing=closing):
                bus = self.bus(capacity=1)
                entered, release, nested = Event(), Event(), Event()
                def outer(_payload):
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('Test did not release the consumer')
                    bus.submit(lambda _: nested.set(), topic='nested', priority=0).result()
                bus.submit(outer, topic='outer')
                self.assertTrue(entered.wait(3))
                def producer():
                    if closing:
                        bus.close()
                    else:
                        bus.submit(lambda _: None, topic='blocked')
                thread = Thread(target=producer)
                admission_waiting = Event()
                original_wait, original_drain = bus.condition.wait, bus.drain
                def observed_wait(*args, **kwargs):
                    if current_thread() is thread:
                        admission_waiting.set()
                    return original_wait(*args, **kwargs)
                def observed_drain(*args, **kwargs):
                    if current_thread() is thread:
                        admission_waiting.set()
                    return original_drain(*args, **kwargs)
                from threading import current_thread
                with patch.object(bus.condition, 'wait', side_effect=observed_wait), \
                        patch.object(bus, 'drain', side_effect=observed_drain):
                    thread.start()
                    try:
                        self.assertTrue(admission_waiting.wait(3))
                    finally:
                        release.set()
                    thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertTrue(nested.wait(3))
                bus.drain()

    def test_accepted_future_cannot_cancel_evidence(self):
        """Cancellation cannot silently remove accepted graph or enrichment work.
        """
        bus, seen, entered, release = self.bus(), Event(), Event(), Event()
        def gate(_payload):
            """Keep accepted evidence queued while cancellation is attempted.
            """
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release cancellation gate')
        bus.submit(gate, topic='gate')
        self.assertTrue(entered.wait(3))
        future = bus.submit(lambda _: seen.set(), topic='graph')
        self.assertFalse(future.cancel())
        release.set()
        future.result(3)
        self.assertTrue(seen.is_set())

    def test_event_replay_is_local_and_fenced_to_job_source(self):
        """Explicit duplicate IDs reuse a receipt; foreign envelopes are rejected.
        """
        bus, seen = self.bus(), []
        event = EventEnvelope('job', 'source', 7, 'graph', 1, 'graph')
        first = bus.submit(lambda _: seen.append('once'), topic='graph', event=event)
        second = bus.submit(lambda _: seen.append('twice'), topic='graph', event=event)
        self.assertIs(first, second)
        bus.drain()
        self.assertEqual(seen, ['once'])
        foreign = EventEnvelope('other-job', 'source', 8, 'graph', 1, 'graph')
        with self.assertRaises(ValueError):
            bus.submit(lambda _: None, topic='graph', event=foreign)
        with self.assertRaises(ValueError):
            bus.bind_source('other-source')

    def test_estimator_rejects_opaque_payloads_and_stops_over_budget(self):
        """Only ordinary report containers qualify for estimated async retention.
        """
        self.assertGreater(payload_weight(object(), 100), 100)
        self.assertGreater(payload_weight({'text': 'x' * 1000}, 100), 100)
        self.assertGreater(payload_weight({'sample': [1, 2]}), 0)

    def test_worker_can_release_item_before_producer_returns_future(self):
        """A fast consumer cannot erase the producer's return value after enqueue.
        """
        bus, drained = self.bus(), Event()
        producer_thread = current_thread()
        armed = Event()
        class AdmissionCondition(Condition):
            def __exit__(condition, *args):
                """Pause the producer only after publishing its first queued item.
                """
                result = super().__exit__(*args)
                if armed.is_set():
                    if current_thread() is producer_thread and bus.counts['backlogHighWater']:
                        if not drained.wait(3):
                            raise TimeoutError('Consumer did not release its completed item')
                    elif current_thread() is bus.worker and bus.counts['completed'] and not bus.outstanding:
                        drained.set()
                return result
        # The idle worker is waiting on the old condition: wake and stop it before
        # replacing the condition, then start the same consumer on the instrumented one.
        bus.close()
        bus.closed = False
        bus.condition = AdmissionCondition()
        bus.worker = Thread(target=bus._consume, name='instrumented-media-consumer')
        bus.worker.start()
        armed.set()
        future = bus.submit(lambda _: 'completed', topic='graph')
        self.assertEqual(future.result(3), 'completed')

    def test_nested_same_key_cannot_overtake_an_accepted_predecessor(self):
        """Unsupported nested ordering raises instead of silently reversing facts.
        """
        bus, entered, release, rejected = self.bus(), Event(), Event(), Event()
        def outer(_payload):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release nested work')
            try:
                bus.submit(lambda _: None, topic='graph', ordering_key='graph')
            except RuntimeError:
                rejected.set()
        bus.submit(outer, topic='outer')
        self.assertTrue(entered.wait(3))
        bus.submit(lambda _: None, topic='graph', ordering_key='graph')
        release.set()
        bus.drain()
        self.assertTrue(rejected.is_set())

    def test_replay_receipts_are_bounded_and_released_on_close(self):
        """Explicit replay bookkeeping cannot grow beyond its documented bound.
        """
        bus = self.bus('sync')
        for sequence in range(1, 65):
            event = EventEnvelope('job', 'source', sequence, 'graph', 1, 'graph')
            bus.submit(lambda _: None, topic='graph', event=event).result()
        with self.assertRaisesRegex(ValueError, 'receipt capacity'):
            event = EventEnvelope('job', 'source', 65, 'graph', 1, 'graph')
            bus.submit(lambda _: None, topic='graph', event=event)
        self.assertEqual(bus.metrics()['replayReceipts'], 64)
        bus.close()
        self.assertEqual(bus.metrics()['replayReceipts'], 0)

    def test_aggregate_payload_budget_blocks_before_another_copy(self):
        """Byte admission applies to all retained reports, even below count capacity.
        """
        bus = self.bus(capacity=3, payload_budget=100)
        entered, release, blocked, copied = Event(), Event(), Event(), Event()
        def gate(_payload):
            """Retain the first payload until the second producer is budget-blocked.
            """
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release retained payload')
        bus.submit(gate, topic='graph', payload_factory=lambda: {}, payload_bytes=60)
        self.assertTrue(entered.wait(3))
        def producer():
            """Request a second individually valid payload without count pressure.
            """
            bus.submit(lambda _: None, topic='graph', payload_factory=lambda: copied.set(), payload_bytes=60)
        thread = Thread(target=producer)
        original_wait = bus.condition.wait
        def observed_wait(*args, **kwargs):
            """Expose an actual blocked byte admission to the regression.
            """
            if current_thread() is thread:
                blocked.set()
            return original_wait(*args, **kwargs)
        with patch.object(bus.condition, 'wait', side_effect=observed_wait):
            thread.start()
            try:
                self.assertTrue(blocked.wait(3))
                self.assertFalse(copied.is_set())
            finally:
                release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        bus.drain()
        self.assertTrue(copied.is_set())
        self.assertEqual(bus.metrics()['retainedBytesHighWater'], 60)

    def test_consumer_rejects_its_own_lifecycle_join(self):
        """Unsupported lifecycle joins fail before acquiring producer admission.
        """
        bus, rejected = self.bus(), []
        def consume(_payload):
            """Attempt each lifecycle operation from the sole consumer thread.
            """
            for operation in (bus.drain, bus.close, lambda: bus.finish(lambda: None)):
                try:
                    operation()
                except RuntimeError:
                    rejected.append(True)
        bus.submit(consume, topic='graph').result(3)
        bus.drain()
        self.assertEqual(rejected, [True, True, True])

    def test_finish_seals_admission_against_waiting_producer(self):
        """A producer waiting during final publication cannot append a late partial.
        """
        bus = self.bus()
        final_entered, release_final, attempted = Event(), Event(), Event()
        errors, seen = [], []
        def final():
            final_entered.set()
            if not release_final.wait(3):
                raise TimeoutError('Test did not release final publication')
            seen.append('complete')
        finisher = Thread(target=lambda: bus.finish(final))
        finisher.start()
        self.assertTrue(final_entered.wait(3))
        def late_producer():
            attempted.set()
            try:
                bus.submit(lambda _: seen.append('late-partial'), topic='graph')
            except RuntimeError as error:
                errors.append(str(error))
        producer = Thread(target=late_producer)
        producer.start()
        self.assertTrue(attempted.wait(3))
        release_final.set()
        finisher.join(3)
        producer.join(3)
        self.assertFalse(finisher.is_alive())
        self.assertFalse(producer.is_alive())
        self.assertEqual(seen, ['complete'])
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            bus.submit(lambda _: None, topic='graph')

    def test_idle_consumer_releases_its_last_frozen_payload(self):
        """A drained bus cannot retain the last copied report outside accounting.
        """
        class Report(dict):
            pass
        bus, references = self.bus(), []
        def freeze():
            report = Report(text='x' * 10000)
            references.append(weakref.ref(report))
            return report
        bus.submit(lambda report: None, topic='graph', payload_factory=freeze, payload_bytes=11000)
        bus.drain()
        self.assertIsNone(references[0]())
        self.assertEqual(bus.metrics()['retainedBytes'], 0)
