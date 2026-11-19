"""Check real user-stage ordering, atomic plan capture, cache and graph limits."""
from pathlib import Path
from unittest import TestCase

import test_media_mod_integration as fixtures
from voice_studio import mod_runtime as runtime
from voice_studio.mod_pipeline import captured_pipeline, pipeline_catalog, save_pipeline, stage_node_id, _topological
from voice_studio.mods import ModError


class ModPipelineTests(TestCase):
    setUp = fixtures.MediaModIntegrationTests.setUp
    tearDown = fixtures.MediaModIntegrationTests.tearDown
    package = fixtures.MediaModIntegrationTests.package

    def stages(self):
        """Create two independently enabled stages whose alphabetical order is deliberately wrong."""
        self.package('z.producer', 'analysis.producer', "return {'number': 7}", kind='stage')
        self.package('a.consumer', 'analysis.consumer', "return {'number': payload['upstream']['mod:z.producer:analysis.producer']['number'] + 1}", kind='stage')
        producer = stage_node_id('z.producer', 'analysis.producer')
        consumer = stage_node_id('a.consumer', 'analysis.consumer')
        return producer, consumer

    def test_plan_is_captured_with_mod_settings_and_restored_to_worker_wire(self):
        producer, consumer = self.stages()
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        old = self.registry.snapshot()
        save_pipeline(self.registry, {}, 1)
        self.assertEqual(captured_pipeline(old)['stages'][0]['nodeId'], producer)
        restored = self.registry.snapshot_from_wire(old.to_wire())
        self.assertEqual(captured_pipeline(restored), captured_pipeline(old))
        self.assertEqual(captured_pipeline(self.registry.snapshot())['stages'][0]['nodeId'], consumer)

    def test_cycles_unknown_native_rewrites_and_stale_revision_fail_without_writing(self):
        producer, consumer = self.stages()
        for dependencies in ({producer: [consumer], consumer: [producer]}, {consumer: ['missing']}, {'builtin:speech': []}, {consumer: ['builtin:scene']}):
            with self.subTest(dependencies=dependencies), self.assertRaises(ModError):
                save_pipeline(self.registry, dependencies, 0)
            self.assertEqual(pipeline_catalog(self.registry)['revision'], 0)
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        with self.assertRaisesRegex(ModError, 'revision changed'):
            save_pipeline(self.registry, {}, 0)
        self.assertEqual(pipeline_catalog(self.registry)['revision'], 1)

    def test_disabled_dependency_fails_instead_of_running_consumer_without_input(self):
        producer, consumer = self.stages()
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        row = next(row for row in self.registry.catalog()['mods'] if row['id'] == 'z.producer')
        self.registry.enable('z.producer', row['fingerprint'], False)
        catalog = pipeline_catalog(self.registry)
        self.assertFalse(next(node for node in catalog['nodes'] if node['id'] == producer)['enabled'])
        with self.assertRaises(ModError):
            captured_pipeline(self.registry.snapshot())
        # An explicit clear can recover an obsolete plan even if its prior dependency is unavailable.
        self.assertEqual(save_pipeline(self.registry, {}, 1)['revision'], 2)

    def test_metadata_contains_disabled_nodes_without_executing_them(self):
        self.package('test.inert', 'analysis.inert', "raise RuntimeError('must not execute')", kind='stage')
        row = next(row for row in self.registry.catalog()['mods'] if row['id'] == 'test.inert')
        self.registry.enable('test.inert', row['fingerprint'], False)
        graph = pipeline_catalog(self.registry)
        node = next(node for node in graph['nodes'] if node['kind'] == 'mod')
        self.assertFalse(node['enabled'])
        self.assertTrue(node['editable'])
        self.assertEqual(node['dependsOn'], ['builtin:report'])

    def test_graph_node_and_edge_budgets_are_rejected_explicitly(self):
        with self.assertRaisesRegex(ModError, '128'):
            _topological([{'id': str(index), 'dependsOn': []} for index in range(129)])
        nodes = [{'id': str(index), 'dependsOn': [str(parent) for parent in range(index)]} for index in range(22)]
        with self.assertRaisesRegex(ModError, '200'):
            _topological(nodes)

    def test_actual_stage_processes_run_in_dependency_order_with_upstream_values(self):
        """Topological dispatch overrides alphabetical order and feeds exact parent results."""
        producer, consumer = self.stages()
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        source = self.root / 'source'
        source.write_bytes(b'same')
        @runtime.analysis_run
        def fixture(root, source, selected_stages=None):
            return {'text': 'native boundary'}
        first = fixture(self.root, source)
        self.assertEqual([row['nodeId'] for row in first['modStages']], [producer, consumer])
        self.assertEqual(first['modStages'][1]['result'], {'number': 8})
        self.assertEqual(first['modPipeline']['revision'], 1)
        restored = fixture(self.root, source, selected_stages=[])
        self.assertEqual([row['state'] for row in restored['modStages']], ['reused', 'reused'])
        self.assertEqual(restored['modStages'][1]['result'], {'number': 8})

    def test_midrun_plan_edit_does_not_change_captured_order_or_inputs(self):
        """An operation changing Settings retains its original captured plan for this result."""
        producer, consumer = self.stages()
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        @runtime.analysis_run
        def fixture(source):
            save_pipeline(self.registry, {}, 1)
            return {'text': 'native boundary'}
        report = fixture(self.root / 'source')
        self.assertEqual(report['modPipeline']['revision'], 1)
        self.assertEqual([row['nodeId'] for row in report['modStages']], [producer, consumer])
        self.assertEqual(pipeline_catalog(self.registry)['revision'], 2)

    def test_rewiring_invalidates_only_affected_custom_stage(self):
        """Changing one edge preserves independent stage results and recomputes its consumer."""
        self.package('test.one', 'analysis.one', "return {'value': 1}", kind='stage')
        self.package('test.two', 'analysis.two', "return {'value': 2}", kind='stage')
        self.package('test.consumer', 'analysis.consumer', "return {'parents': sorted(payload['upstream'])}", kind='stage')
        one = stage_node_id('test.one', 'analysis.one')
        two = stage_node_id('test.two', 'analysis.two')
        consumer = stage_node_id('test.consumer', 'analysis.consumer')
        source = self.root / 'source'
        source.write_bytes(b'unchanged')
        @runtime.analysis_run
        def fixture(root, source):
            return {'text': 'unchanged'}
        save_pipeline(self.registry, {consumer: [one]}, 0)
        first = fixture(self.root, source)
        self.assertTrue(all(row['state'] == 'done' for row in first['modStages']))
        save_pipeline(self.registry, {consumer: [two]}, 1)
        second = fixture(self.root, source)
        rows = {row['nodeId']: row for row in second['modStages']}
        self.assertEqual(rows[one]['state'], 'reused')
        self.assertEqual(rows[two]['state'], 'reused')
        self.assertEqual(rows[consumer]['state'], 'done')
        self.assertEqual(rows[consumer]['result'], {'parents': [two]})

    def test_deleted_stage_references_remain_repairable_in_settings(self):
        """Removing a package preserves a disabled graph card; clearing edges repairs the plan."""
        import shutil
        producer, consumer = self.stages()
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        shutil.rmtree(self.registry.directory / 'z.producer')
        graph = pipeline_catalog(self.registry)
        missing = next(node for node in graph['nodes'] if node['id'] == producer)
        self.assertEqual(missing['status'], 'missing')
        self.assertFalse(missing['enabled'])
        with self.assertRaises(ModError):
            captured_pipeline(self.registry.snapshot())
        repaired = save_pipeline(self.registry, {consumer: [], producer: []}, 1)
        self.assertEqual(repaired['dependencies'], {})
        self.assertFalse(any(node['id'] == producer for node in repaired['nodes']))
        self.assertEqual(len(captured_pipeline(self.registry.snapshot())['stages']), 1)

    def test_pipeline_edits_leave_all_native_layer_fingerprints_unchanged(self):
        """Only extra-stage graph keys change; the seven native cache keys exclude extensions."""
        producer, consumer = self.stages()
        before = {name: self.registry.snapshot().fingerprint(capabilities)
                  for name, capabilities in runtime.LAYER_CAPABILITIES.items()}
        save_pipeline(self.registry, {consumer: [producer]}, 0)
        after = {name: self.registry.snapshot().fingerprint(capabilities)
                 for name, capabilities in runtime.LAYER_CAPABILITIES.items()}
        self.assertEqual(before, after)
