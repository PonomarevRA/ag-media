"""Bounded, captured dependency plans for additional media mod stages.

The seven native analysis slots keep their mandatory dependencies and execution
semantics. Only edges between user stages are configurable. All user stages run
at the completed selected-native-report boundary, never between native workers.
"""
from copy import deepcopy
from hashlib import sha256

from voice_studio.graph.media_stage_selection import DEPENDENCIES, STAGE_NAMES
from voice_studio.mods import ModError
from voice_studio.mods.schema import json_bytes

MAX_NODES = 128
MAX_EDGES = 200
REPORT_NODE = 'builtin:report'
NOTICE = ('Native nodes are logical evidence slots: only selected or reusable native work runs. '
          'Additional stages consume the completed selected-native report. Only user-stage '
          'dependencies can be edited; changes apply to the next analysis.')


def stage_node_id(mod_id, capability):
    """Keep graph identity stable across display names, versions and settings edits."""
    return 'mod:' + mod_id + ':' + capability


def _native_nodes():
    """Expose mandatory native dependencies without implying all optional slots execute."""
    nodes = []
    for name in STAGE_NAMES:
        nodes.append({'id': 'builtin:' + name, 'label': name, 'kind': 'builtin',
                      'dependsOn': ['builtin:' + parent for parent in DEPENDENCIES.get(name, ())],
                      'editable': False, 'enabled': True, 'status': 'logical-slot'})
    nodes.append({'id': REPORT_NODE, 'label': 'Result of selected native stages', 'kind': 'builtin',
                  'dependsOn': ['builtin:' + name for name in STAGE_NAMES],
                  'editable': False, 'enabled': True, 'status': 'report-boundary'})
    return nodes


def _read_plan(value):
    """Validate bounded durable plan structure independently of current package discovery."""
    if value is None:
        return {'apiVersion': 1, 'revision': 0, 'dependencies': {}}
    if (not isinstance(value, dict) or set(value) != {'apiVersion', 'revision', 'dependencies'}
            or type(value['apiVersion']) is not int or value['apiVersion'] != 1
            or type(value['revision']) is not int or value['revision'] < 0):
        raise ModError('Invalid processing pipeline state.')
    dependencies = value['dependencies']
    if not isinstance(dependencies, dict) or len(dependencies) > MAX_NODES:
        raise ModError('Pipeline dependencies must be a bounded object.')
    for target, parents in dependencies.items():
        if (not isinstance(target, str) or len(target) > 200 or not isinstance(parents, list)
                or len(parents) > MAX_NODES or any(not isinstance(parent, str) or len(parent) > 200 for parent in parents)
                or len(set(parents)) != len(parents)):
            raise ModError('Pipeline dependency IDs must be bounded unique string arrays.')
    if len(json_bytes(value)) > 60000:
        raise ModError('Processing pipeline exceeds its 60 KiB state budget.')
    return deepcopy(value)


def _topological(nodes):
    """Return deterministic topological IDs, rejecting unknown references and cycles."""
    by_id = {node['id']: node for node in nodes}
    if len(nodes) > MAX_NODES or len(by_id) != len(nodes):
        raise ModError(f'Processing pipeline supports at most {MAX_NODES} unique nodes.')
    edge_count = sum(len(node['dependsOn']) for node in nodes)
    if edge_count > MAX_EDGES:
        raise ModError(f'Processing pipeline supports at most {MAX_EDGES} edges.')
    pending = {identifier: set(node['dependsOn']) for identifier, node in by_id.items()}
    for parents in pending.values():
        if not parents <= set(by_id):
            raise ModError('Processing pipeline references an unknown node.')
    order = []
    while pending:
        ready = sorted(identifier for identifier, parents in pending.items() if not parents)
        if not ready:
            raise ModError('Processing pipeline contains a dependency cycle.')
        for identifier in ready:
            order.append(identifier)
            del pending[identifier]
        completed = set(ready)
        for parents in pending.values():
            parents.difference_update(completed)
    return order


def _apply_dependencies(nodes, plan, *, active_only=False, allow_disabled=False):
    """Apply editable custom edges while retaining locked native/report dependencies."""
    by_id = {node['id']: node for node in nodes}
    for target, parents in plan['dependencies'].items():
        if target not in by_id:
            if active_only:
                continue  # Disabled targets do not execute in this captured analysis.
            raise ModError('Pipeline target is missing: ' + target)
        node = by_id[target]
        if node['kind'] != 'mod':
            raise ModError('Native processing dependencies are read-only.')
        for parent in parents:
            source = by_id.get(parent)
            if source is None or source['kind'] != 'mod':
                raise ModError('User stages may depend only on existing user stages: ' + parent)
            if not source['enabled'] and not allow_disabled:
                raise ModError('A pipeline dependency is disabled or unavailable: ' + parent)
        node['dependsOn'] = [REPORT_NODE, *parents]
    return _topological(nodes)


def _catalog_nodes(catalog):
    """Expose discovered user stages, including disabled nodes available for planning."""
    nodes = _native_nodes()
    for row in catalog['mods']:
        if row.get('origin') != 'user' or not isinstance(row.get('manifest'), dict):
            continue
        for declaration in row['manifest']['capabilities']:
            if declaration['kind'] != 'stage':
                continue
            nodes.append({'id': stage_node_id(row['id'], declaration['id']),
                          'label': row['manifest']['name'] + ' / ' + declaration['id'], 'kind': 'mod',
                          'modId': row['id'], 'capability': declaration['id'],
                          'dependsOn': [REPORT_NODE], 'editable': True,
                          'enabled': row['enabled'], 'status': row['status']})
    return nodes


def _missing_nodes(nodes, plan):
    """Retain deleted user-stage references as disabled cards so Settings can repair them."""
    known = {node['id'] for node in nodes}
    referenced = set(plan['dependencies'])
    for parents in plan['dependencies'].values():
        referenced.update(parents)
    for identifier in sorted(referenced - known):
        if not identifier.startswith('mod:'):
            raise ModError('Pipeline references an invalid user-stage ID: ' + identifier)
        nodes.append({'id': identifier, 'label': 'Unavailable stage: ' + identifier,
                      'kind': 'mod', 'dependsOn': [REPORT_NODE], 'editable': True,
                      'enabled': False, 'status': 'missing', 'missing': True})
    return nodes


def pipeline_catalog(service):
    """Read the Settings graph as metadata without executing code or changing mod activation."""
    catalog = service.catalog()
    plan = _read_plan(service.snapshot().extension('pipeline'))
    nodes = _missing_nodes(_catalog_nodes(catalog), plan)
    _apply_dependencies(nodes, plan, allow_disabled=True)
    return {'apiVersion': 1, 'revision': plan['revision'], 'nodes': nodes,
            'dependencies': plan['dependencies'], 'notice': NOTICE}


def save_pipeline(service, dependencies, expected_revision):
    """Compare-and-swap one validated user-stage plan in the registry's atomic state file."""
    if type(expected_revision) is not int or expected_revision < 0:
        raise ModError('Pipeline save requires a nonnegative expectedRevision.')
    with service.store.transaction() as state:
        current = _read_plan(state['extensions'].get('pipeline'))
        if current['revision'] != expected_revision:
            raise ModError('Pipeline revision changed; reload Settings before saving.')
        candidate = _read_plan({'apiVersion': 1, 'revision': expected_revision + 1,
                                'dependencies': dependencies})
        if any(not target.startswith('mod:') for target in candidate['dependencies']):
            raise ModError('Native processing dependencies are read-only.')
        # Clearing a deleted target is an explicit repair, not a new unknown-node edge.
        candidate['dependencies'] = {target: parents for target, parents in candidate['dependencies'].items() if parents}
        # The state transaction excludes concurrent activation/settings writes.
        # Catalog is metadata-only and reads that same committed activation state.
        nodes = _catalog_nodes(service.catalog())
        _apply_dependencies(nodes, candidate)
        state['extensions']['pipeline'] = candidate
    return pipeline_catalog(service)


def captured_pipeline(snapshot):
    """Validate one frozen plan before native work and order its enabled additional stages."""
    plan = _read_plan(snapshot.extension('pipeline'))
    nodes = _native_nodes()
    declarations = {}
    for declaration in snapshot.custom_stages():
        identifier = stage_node_id(declaration['modId'], declaration['capability'])
        declarations[identifier] = declaration
        nodes.append({'id': identifier, 'kind': 'mod', 'enabled': True, 'dependsOn': [REPORT_NODE]})
    order = _apply_dependencies(nodes, plan, active_only=True)
    ordered = []
    by_id = {node['id']: node for node in nodes}
    for identifier in order:
        if identifier in declarations:
            parents = [parent for parent in by_id[identifier]['dependsOn'] if parent != REPORT_NODE]
            ordered.append({**declarations[identifier], 'nodeId': identifier, 'dependsOn': parents})
    return {'apiVersion': 1, 'revision': plan['revision'], 'stages': ordered,
            'fingerprint': sha256(json_bytes(plan)).hexdigest()}
