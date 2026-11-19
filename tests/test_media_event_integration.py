"""Exercise actual immutable graph storage through the MCP event lifecycle."""

from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_graph_progress import GraphPublication
from voice_studio.graph.media_graph_store import MediaGraphStore
from voice_studio.mcp.media_graph_mcp import build_snapshot


class MediaEventIntegrationTests(TestCase):
    def fixture(self, root):
        """Prepare a small original and a detached analyzer result without models.
        """
        source = root / 'original.wav'
        source.write_bytes(b'event-bus-original-fixture')
        streams = [{'kind': 'audio', 'index': 0}]
        media = {'kind': 'audio', 'durationMs': 2000, 'streams': streams}
        report = {'language': 'en', 'mediaStreams': streams, 'durationMs': 2000,
                  'segments': [{'startMs': 100, 'endMs': 800, 'text': 'Measured reply'}],
                  'analysisStages': {'speech': {'state': 'done'}},
                  'objectReconciliationEvidence': {'status': 'no-evidence'}}
        return source, media, report

    def build(self, root, source, media, mode, analyze, *, store=None, publication=None):
        """Use the real MCP entry point and store while replacing native analysis.
        """
        store = store or MediaGraphStore(root / 'graphs')
        publication = publication or GraphPublication('event-test-job')
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'VOICE_MEDIA_EVENT_MODE': mode}))
            stack.enter_context(patch('voice_studio.graph.media_graph_ingest.probe_source', return_value=media))
            stack.enter_context(patch('voice_studio.graph.media_graph_ingest.analyze_source', side_effect=analyze))
            result = build_snapshot(store, root / 'studio', str(source), 'en', None,
                                    False, False, None, publication=publication)
        return result, publication, store

    def rows(self, store, identity):
        """Compare complete stored graph content rather than volatile snapshot IDs.
        """
        with store.database(identity) as db:
            return {'nodes': [tuple(row) for row in db.execute('SELECT * FROM nodes ORDER BY seq')],
                    'edges': [tuple(row) for row in db.execute('SELECT * FROM edges ORDER BY source,relation,target')],
                    'faces': [tuple(row) for row in db.execute('SELECT * FROM face_chunks ORDER BY bucket')]}

    def test_modes_keep_graph_facts_final_provenance_and_revision_chain(self):
        """Scheduling changes leave canonical nodes, edges and coverage identical.
        """
        expected = None
        for mode in ('sync', 'fifo', 'priority'):
            with self.subTest(mode=mode), TemporaryDirectory() as folder:
                root = Path(folder)
                source, media, report = self.fixture(root)
                def analyze(*args, publish, **kwargs):
                    base = deepcopy(report)
                    publish('base', base)
                    base['segments'][0]['text'] = 'later producer mutation'
                    enriched = deepcopy(report)
                    enriched['audioLayers'] = [{'kind': 'music', 'label': 'music', 'startMs': 0,
                                                'endMs': 1500, 'confidence': .8}]
                    publish('audio', enriched)
                    publish('audio', enriched)
                    return enriched
                result, publication, store = self.build(root, source, media, mode, analyze)
                actual = self.rows(store, result['snapshotId'])
                if expected is None:
                    expected = actual
                self.assertEqual(actual, expected)
                state = publication.state()
                self.assertEqual((state['phase'], state['revision']), ('complete', 3))
                self.assertEqual(len(store.list()['snapshots']), 3)
                previous = result['provenance']['progress']['previousSnapshotId']
                self.assertEqual(store.overview(previous)['provenance']['progress']['phase'], 'audio')
                self.assertEqual(result['provenance']['analysisStages'], report['analysisStages'])
                self.assertEqual(result['provenance']['objectReconciliationEvidence'], {'status': 'no-evidence'})
                self.assertEqual(result['provenance']['eventBus']['mode'], mode)
                self.assertEqual(publication.event_bus.metrics()['outstanding'], 0)
                self.assertTrue(publication.event_bus.metrics()['finished'])

    def test_analysis_failure_drains_partial_before_scope_closes(self):
        """A late model error retains the previously accepted readable snapshot.
        """
        for mode in ('sync', 'fifo', 'priority'):
            with self.subTest(mode=mode), TemporaryDirectory() as folder:
                root = Path(folder)
                source, media, report = self.fixture(root)
                publication = GraphPublication('failed-analysis-job')
                store = MediaGraphStore(root / 'graphs')
                def analyze(*args, publish, **kwargs):
                    publish('base', report)
                    raise ValueError('late analysis failure')
                with self.assertRaisesRegex(ValueError, 'late analysis failure'):
                    self.build(root, source, media, mode, analyze, store=store, publication=publication)
                state = publication.state()
                self.assertEqual((state['phase'], state['revision']), ('base', 1))
                self.assertEqual(store.search(state['snapshot']['snapshotId'], 'Measured')['nodes'][0]['label'],
                                 'Measured reply')
                self.assertEqual(publication.event_bus.metrics()['outstanding'], 0)
                self.assertFalse(publication.event_bus.worker and publication.event_bus.worker.is_alive())

    def test_final_storage_failure_keeps_last_partial(self):
        """Final write errors propagate without advertising an unreadable revision.
        """
        for mode in ('sync', 'fifo', 'priority'):
            with self.subTest(mode=mode), TemporaryDirectory() as folder:
                root = Path(folder)
                source, media, report = self.fixture(root)
                publication = GraphPublication('failed-final-job')
                store = MediaGraphStore(root / 'graphs')
                original_publish = store.publish
                def store_publish(source, media, graph, language, provenance):
                    if (provenance.get('progress') or {}).get('phase') == 'complete':
                        raise OSError('final storage unavailable')
                    return original_publish(source, media, graph, language, provenance)
                def analyze(*args, publish, **kwargs):
                    publish('base', report)
                    return report
                with patch.object(store, 'publish', side_effect=store_publish), \
                        self.assertRaisesRegex(OSError, 'final storage unavailable'):
                    self.build(root, source, media, mode, analyze, store=store, publication=publication)
                state = publication.state()
                self.assertEqual((state['phase'], state['revision']), ('base', 1))
                self.assertEqual(store.overview(state['snapshot']['snapshotId'])['snapshotId'],
                                 state['snapshot']['snapshotId'])
                self.assertTrue(publication.event_bus.metrics()['finished'])

    def test_source_change_aborts_final_and_preserves_immutable_partial(self):
        """Deferred storage must still verify the exact original at write time.
        """
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, media, report = self.fixture(root)
            publication = GraphPublication('changed-source-job')
            store = MediaGraphStore(root / 'graphs')
            def analyze(*args, publish, **kwargs):
                publish('base', report)
                publication.event_bus.drain()
                source.write_bytes(b'replaced original bytes')
                return report
            with self.assertRaisesRegex(ValueError, 'Исходник изменился'):
                self.build(root, source, media, 'priority', analyze, store=store, publication=publication)
            state = publication.state()
            self.assertEqual((state['phase'], state['revision']), ('base', 1))
            self.assertEqual(len(store.list()['snapshots']), 1)

    def test_partial_projection_needs_no_producer_temporary_directory(self):
        """Detached evidence survives analysis scratch cleanup; original stays durable.
        """
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, media, report = self.fixture(root)
            publication = GraphPublication('temporary-producer-job')
            entered, release = Event(), Event()
            def analyze(*args, publish, **kwargs):
                def gate(_payload):
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('Test did not release the consumer')
                publication.event_bus.submit(gate, topic='hold')
                self.assertTrue(entered.wait(3))
                try:
                    with TemporaryDirectory() as scratch:
                        text = Path(scratch) / 'measured.txt'
                        text.write_text('Measured reply')
                        partial = deepcopy(report)
                        partial['segments'][0]['text'] = text.read_text()
                        publish('base', partial)
                    self.assertFalse(text.exists())
                finally:
                    release.set()
                return report
            result, publication, store = self.build(root, source, media, 'priority', analyze,
                                                    publication=publication)
            self.assertEqual(store.search(result['snapshotId'], 'Measured')['nodes'][0]['label'], 'Measured reply')
