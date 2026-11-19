"""Compare actual graph transport on frozen reports and optional native media.

Each sample has an isolated Python host and SQLite store. Replay gaps represent
model work; replay timings cannot establish native inference acceleration. The
optional native input is analyzed by the real build_snapshot entry point.
"""

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from hashlib import file_digest, sha256
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import resource
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METADATA_IGNORED = {'snapshotId', 'createdAt'}
TELEMETRY_IGNORED = {'elapsedMs', 'decodeMs', 'sceneStageMs', 'sceneReadMs'}
SOURCE_FILES = ('voice_studio/core/media_event_bus.py', 'voice_studio/graph/media_graph_events.py',
                'voice_studio/mcp/media_graph_mcp.py', 'voice_studio/graph/media_object_review.py',
                'voice_studio/graph/media_graph_model.py', 'voice_studio/graph/media_graph_store.py',
                'voice_studio/graph/media_graph_ingest.py')


def digest(value):
    """Hash canonical JSON without omitting unknown facts or numeric precision."""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    """Stream a file's content identity without retaining the original media."""
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def content_facts(value):
    """Remove only measured execution duration fields from nested graph facts."""
    if isinstance(value, dict):
        return {key: content_facts(item) for key, item in value.items() if key not in TELEMETRY_IGNORED}
    if isinstance(value, list):
        return [content_facts(item) for item in value]
    return value


def snapshot_fingerprint(path):
    """Fingerprint exact nodes, source clock, edges, geometry and stable manifest.

    Random snapshot IDs, creation time and publication/event-bus telemetry are
    excluded only from the manifest. Node IDs and unknown fact fields remain.
    """
    import sqlite3

    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        manifest = json.loads(db.execute('SELECT value FROM manifest').fetchone()[0])
        for key in METADATA_IGNORED:
            manifest.pop(key, None)
        provenance = manifest.get('provenance', {})
        for key in ('progress', 'eventBus', 'mediaEventBus', 'checkpoints'):
            provenance.pop(key, None)
        nodes, kinds = [], Counter()
        for row in db.execute('SELECT id,kind,start_ms,end_ms,label,data,description FROM nodes ORDER BY id'):
            identity, kind, start, end, label, data, description = row
            nodes.append([identity, kind, start, end, label, content_facts(json.loads(data)), description])
            kinds[kind] += 1
        edges = list(db.execute('SELECT source,relation,target FROM edge_refs ORDER BY source,relation,target'))
        faces = [[bucket, json.loads(zlib.decompress(payload))]
                 for bucket, payload in db.execute('SELECT bucket,payload FROM face_chunks ORDER BY bucket')]
    return {'nodesSha256': digest(nodes), 'edgesSha256': digest(edges), 'faceFramesSha256': digest(faces),
            'manifestSha256': digest(content_facts(manifest)), 'coverageSha256': digest(manifest.get('coverage')),
            'sourceSha256': manifest['source']['sha256'], 'durationMs': manifest['durationMs'],
            'nodeCount': len(nodes), 'edgeCount': len(edges), 'nodesByKind': dict(sorted(kinds.items()))}


class TimedStore:
    """Observe real successful publication boundaries without changing store behavior."""

    def __init__(self, root, started):
        from voice_studio.graph.media_graph_store import MediaGraphStore
        self.store, self.started = MediaGraphStore(root), started
        self.events = []

    def publish(self, *args, **kwargs):
        """Record when a real immutable SQLite snapshot becomes readable."""
        result = self.store.publish(*args, **kwargs)
        marker = result.get('provenance', {}).get('progress', {})
        self.events.append({'seconds': perf_counter() - self.started, 'phase': marker.get('phase'),
                            'revision': marker.get('revision'), 'snapshotId': result['snapshotId']})
        return result

    def __getattr__(self, name):
        return getattr(self.store, name)


def baseline_module(path):
    """Load the preserved pre-change MCP implementation under an isolated name."""
    spec = importlib.util.spec_from_file_location('media_events_preserved_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cumulative_reports(report):
    """Replay fixed cumulative stage boundaries from one saved, unmodified report.

    These are reconstructed boundaries rather than captured native timestamps.
    The final report retains every saved field, including unavailable evidence.
    """
    staged = {key: deepcopy(report[key]) for key in ('language', 'durationMs', 'mediaStreams', 'streams') if key in report}
    yield 'base', deepcopy(staged)
    groups = [('speech', ('segments', 'speakerProfileIndex', 'speakerScope', 'dialoguePauses', 'speechTurns')),
              ('audio', ('audioLayers', 'soundAnalysisStatus', 'recognitionModel')),
              ('visual', ('videoEvidence',)), ('scene', ('sceneEvidence', 'shotColors', 'objectTimeline', 'objectProfiles'))]
    for phase, fields in groups:
        for key in fields:
            if key in report:
                staged[key] = deepcopy(report[key])
        yield phase, deepcopy(staged)


def replay_sample(case, mode, options, work):
    """Exercise preserved or current publishers with real graph/SQLite and ORB work."""
    from voice_studio.graph.media_graph_model import build_graph
    from voice_studio.graph.media_graph_progress import GraphPublication
    from voice_studio.graph.media_graph_store import identify
    from voice_studio.mcp import media_graph_mcp
    from voice_studio.vision.object_reconciliation import prepare_object_reconciliation

    report_path = ROOT / case['report']
    if file_sha(report_path) != case['reportSha256']:
        raise ValueError('Frozen report bytes changed')
    report = json.loads(report_path.read_text())
    source = identify(case['source'])
    if source['sha256'] != case['sourceSha256']:
        raise ValueError('Original source bytes changed')
    media = {'kind': case['kind'], 'durationMs': case['durationMs'], 'streams': report.get('mediaStreams', report.get('streams', []))}
    started = perf_counter()
    store, publication = TimedStore(work / 'graphs', started), GraphPublication('fixed-replay-job')
    stalls, enrichment = [], {}
    bus = None
    if mode == 'baseline':
        publisher = baseline_module(options.baseline).intermediate_publisher(publication, store, source, media, case['language'])
    else:
        from voice_studio.core.media_event_bus import MediaEventBus
        bus = MediaEventBus('fixed-replay-job', mode=mode, capacity=3)
        bus.bind_source(source['sha256'])
        publisher = media_graph_mcp.graph_event_publisher(bus, publication, store, source, media, case['language'])
    try:
        for phase, staged in cumulative_reports(report):
            tick = perf_counter()
            publisher(phase, staged)
            stalls.append({'phase': phase, 'seconds': perf_counter() - tick})
            sleep(options.gap_ms / 1000)
        # This is the same final dependency boundary: enrichment must finish
        # before its facts can be projected into the complete graph.
        tick = perf_counter()
        operation = lambda: prepare_object_reconciliation(Path(case['source']), report)
        if bus is None:
            reviewed = operation()
        else:
            from voice_studio.core.media_event_bus import media_event_scope, run_enrichment
            from voice_studio.core.media_event_bus import CURRENT_BUS
            token = CURRENT_BUS.set(bus)
            try:
                reviewed = run_enrichment(operation)
            finally:
                CURRENT_BUS.reset(token)
        enrichment = {'seconds': perf_counter() - tick,
                      'evidence': reviewed.get('objectReconciliationEvidence', {}),
                      'factsSha256': digest(content_facts(reviewed))}
        model_done = perf_counter()
        if bus is not None:
            bus.drain()
        drained = perf_counter()
        graph = build_graph(reviewed, kind=case['kind'], duration_ms=case['durationMs'])
        def complete():
            """Advertise final facts only after all intermediate work has drained."""
            return publication.publish('complete', lambda marker: store.publish(source, media, graph, case['language'],
                        {'mode': 'benchmark-replay', 'progress': marker}))
        final = bus.finish(complete) if bus is not None else complete()
        ended = perf_counter()
        return {'case': case['name'], 'mode': mode, 'kind': 'replay', 'callbackStalls': stalls,
                'callbackStallSeconds': sum(item['seconds'] for item in stalls),
                'firstReadableSeconds': store.events[0]['seconds'] if store.events else None,
                'modelBoundarySeconds': model_done - started, 'drainSeconds': drained - model_done,
                'totalSeconds': ended - started, 'events': store.events, 'enrichment': enrichment,
                'bus': bus.metrics() if bus else None,
                'snapshots': [{'phase': event['phase'], 'fingerprint': snapshot_fingerprint(work / 'graphs' / (event['snapshotId'] + '.sqlite'))} for event in store.events],
                'final': snapshot_fingerprint(work / 'graphs' / (final['snapshotId'] + '.sqlite'))}
    finally:
        if bus is not None:
            bus.close(propagate=False)


def registry_digest(folder):
    """Fingerprint only the explicit source identity seed, retaining all biometric facts."""
    files = sorted(Path(folder).rglob('registry.json'))
    if not files:
        raise ValueError('Native registry fixture has no registry.json')
    return digest({str(path.relative_to(folder)): file_sha(path) for path in files})


def seed_native_registry(folder, studio_root, source):
    """Copy a frozen voice identity seed into a fresh private analysis workspace.

    Every run receives its own copy. No completed analysis layers are supplied;
    native speech, visual, scene, voice and object work must run normally.
    """
    from voice_paths import workspace_temp
    target = workspace_temp(studio_root) / 'analysis-checkpoints' / 'short-v1' / file_sha(source) / 'voice-registry'
    for path in sorted(Path(folder).rglob('registry.json')):
        destination = target / path.relative_to(folder)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        destination.chmod(0o600)


def native_sample(case, mode, options, work):
    """Run the actual native analyzer and final snapshot builder on a fixed clip."""
    from voice_studio.graph.media_graph_progress import GraphPublication
    from voice_studio.mcp.media_graph_mcp import build_snapshot
    from unittest.mock import patch
    from voice_studio.mcp import media_graph_mcp

    os.environ['VOICE_MEDIA_EVENT_MODE'] = 'sync' if mode == 'baseline' else mode
    module = baseline_module(options.baseline) if mode == 'baseline' else media_graph_mcp
    build_snapshot = module.build_snapshot
    callback_name = 'intermediate_publisher' if mode == 'baseline' else 'graph_event_publisher'
    original_publisher = getattr(module, callback_name)
    stalls = []

    def measured_publisher(*args, **kwargs):
        """Time only actual callbacks emitted by the native analysis pipeline."""
        publish = original_publisher(*args, **kwargs)
        if publish is None:
            return None
        def timed(phase, report):
            """Preserve native callback results and exceptions while measuring stalls."""
            tick = perf_counter()
            try:
                return publish(phase, report)
            finally:
                stalls.append({'phase': phase, 'seconds': perf_counter() - tick})
        return timed

    seed_hash = None
    if options.native_registry:
        seed_hash = registry_digest(options.native_registry)
        seed_native_registry(options.native_registry, work / 'studio', options.native_source)
    started = perf_counter()
    store = TimedStore(work / 'graphs', started)
    with patch.object(module, callback_name, measured_publisher):
        final = build_snapshot(store, work / 'studio', str(options.native_source), options.native_language,
            None, True, True, None, object_sampling='small-detail', audio_classifier='auto', people_count=True,
            publication=GraphPublication('fixed-native-job'), object_reconciliation=True)
    elapsed = perf_counter() - started
    if options.native_registry:
        retained = options.scratch / 'native-snapshots' / options.label
        shutil.copytree(work / 'graphs', retained, dirs_exist_ok=True)
    return {'registryFixtureSha256': seed_hash, 'case': 'native-clip', 'mode': mode, 'kind': 'native', 'totalSeconds': elapsed,
            'callbackStalls': stalls, 'callbackStallSeconds': sum(item['seconds'] for item in stalls),
            'firstReadableSeconds': store.events[0]['seconds'] if store.events else None, 'events': store.events,
            'configuration': {'visual': True, 'scene': True, 'quality': 'build_snapshot default true',
                'objectSampling': 'small-detail', 'audioClassifier': 'auto', 'peopleCount': True,
                'objectReconciliation': True, 'selectedStages': None, 'forceStages': None},
            'bus': final.get('provenance', {}).get('eventBus'),
            'snapshots': [{'phase': event['phase'], 'fingerprint': snapshot_fingerprint(work / 'graphs' / (event['snapshotId'] + '.sqlite'))} for event in store.events],
            'final': snapshot_fingerprint(work / 'graphs' / (final['snapshotId'] + '.sqlite'))}


def worker(options):
    """Write one isolated measurement, retaining failures as explicit evidence."""
    os.environ['TMPDIR'] = str(options.scratch)
    os.environ['VOICE_TEMP_DIR'] = str(options.scratch / ('temp-' + options.label))
    case = json.loads(options.manifest.read_text())['cases'][options.case_index]
    result = {'label': options.label, 'mode': options.mode, 'status': 'failed',
              'loadedSourceFiles': {name: file_sha(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()},
              'benchmarkSha256': file_sha(Path(__file__))}
    try:
        with TemporaryDirectory(prefix=options.label + '-', dir=options.scratch) as folder:
            operation = native_sample if options.native else replay_sample
            result.update(operation(case, options.mode, options, Path(folder)), status='passed')
            result['publicationOrder'] = publication_order(result['events'])
            if not result['publicationOrder']['passed']:
                raise ValueError('Publication ordering invariant failed')
            result['postRunTreeResources'] = process_tree_sample(os.getpid())
    except Exception as error:
        import traceback
        result['status'] = 'failed'
        result['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    result['cpuSeconds'] = {name: resource.getrusage(who).ru_utime + resource.getrusage(who).ru_stime
        for name, who in [('host', resource.RUSAGE_SELF), ('waitedChildren', resource.RUSAGE_CHILDREN)]}
    result['hostPeakRssBytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def publication_order(events):
    """Check final-last, monotonic revisions and one successful snapshot per phase."""
    phases = [event.get('phase') for event in events]
    revisions = [event.get('revision') for event in events]
    unique_phases = len(phases) == len(set(phases))
    final_last = bool(phases) and phases[-1] == 'complete'
    monotonic = all(type(value) is int for value in revisions) and all(
        later > earlier for earlier, later in zip(revisions, revisions[1:]))
    return {'passed': unique_phases and final_last and monotonic, 'uniquePhases': unique_phases,
            'completeLast': final_last, 'strictlyIncreasingRevisions': monotonic, 'phases': phases, 'revisions': revisions}


def process_tree_sample(pid):
    """Sample an isolated host plus its children; summed RSS is not unique RAM."""
    sampled = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss=,%cpu='], capture_output=True, text=True, check=False)
    if sampled.returncode:
        raise OSError(sampled.stderr.strip() or 'ps failed')
    rows = []
    for line in sampled.stdout.splitlines():
        process, parent, rss, cpu = line.split()
        rows.append((int(process), int(parent), int(rss), float(cpu)))
    owned = {pid}
    while True:
        found = {process for process, parent, _, _ in rows if parent in owned}
        if found <= owned:
            break
        owned.update(found)
    return {'rssBytes': sum(rss * 1024 for process, _, rss, _ in rows if process in owned),
            'cpuPercentDiagnostic': sum(cpu for process, _, _, cpu in rows if process in owned)}


def stop_process_group(process):
    """Terminate only a measurement's owned process group, including native children."""
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_isolated(options, label, mode, case_index, native=False):
    """Launch one measurement and sample its full host tree every 50ms."""
    output = options.output / (label + '.json')
    output.unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--manifest', str(options.manifest),
        '--output', str(output), '--label', label, '--mode', mode, '--case-index', str(case_index),
        '--scratch', str(options.scratch), '--gap-ms', str(options.gap_ms), '--baseline', str(options.baseline)]
    if native and options.native_registry:
        command += ['--native-registry', str(options.native_registry)]
    if native:
        command += ['--native', '--native-source', str(options.native_source), '--native-language', options.native_language]
    samples, error = [], None
    with (options.output / (label + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
        deadline = perf_counter() + options.timeout
        timed_out = False
        try:
            while process.poll() is None:
                if perf_counter() > deadline:
                    stop_process_group(process)
                    timed_out = True
                    break
                if error is None:
                    try:
                        samples.append(process_tree_sample(process.pid))
                    except OSError as failure:
                        error = str(failure)
                sleep(.05)
        except BaseException:
            if process.poll() is None:
                stop_process_group(process)
            raise
        if timed_out:
            result = {'label': label, 'status': 'failed', 'error': 'measurement timed out', 'timeoutSeconds': options.timeout}
        elif not output.is_file():
            result = {'label': label, 'status': 'failed', 'exitCode': process.returncode}
        else:
            result = json.loads(output.read_text())
    result['treeResources'] = {'peakRssBytes': max((row['rssBytes'] for row in samples), default=None),
        'lastRssBytes': samples[-1]['rssBytes'] if samples else None, 'samples': len(samples), 'error': error,
        'maxCpuPercentDiagnostic': max((row['cpuPercentDiagnostic'] for row in samples), default=None),
        'scope': '50ms sampled isolated measurement Python host + all children; summed RSS not unique physical RAM; ps CPU is diagnostic lifetime average'}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: result.get(key) for key in ('label', 'status', 'totalSeconds', 'callbackStallSeconds')}), flush=True)
    return result


def summarize(samples):
    """Report n, median and range, and exact parity against sync or preserved baseline."""
    groups = {}
    for row in samples:
        key = row.get('case', 'unknown')
        groups.setdefault(key, {}).setdefault(row.get('mode', 'unknown'), []).append(row)
    result = {}
    for case, modes in groups.items():
        reference_row = (modes.get('baseline') or modes.get('sync') or [{}])[0]
        reference = reference_row.get('final')
        partial_reference = reference_row.get('snapshots')
        result[case] = {}
        for mode, rows in modes.items():
            passed = [row for row in rows if row.get('status') == 'passed']
            metrics = {}
            for name in ('totalSeconds', 'callbackStallSeconds', 'firstReadableSeconds', 'drainSeconds'):
                values = [row[name] for row in passed if row.get(name) is not None]
                if values:
                    metrics[name] = {'n': len(values), 'median': median(values), 'min': min(values), 'max': max(values)}
            result[case][mode] = {'n': len(rows), 'passed': len(passed), 'parity': bool(reference) and all(row.get('final') == reference for row in passed) and len(passed) == len(rows),
                'publicationOrderPassed': all(row.get('publicationOrder', {}).get('passed') for row in passed) and len(passed) == len(rows),
                'partialParity': partial_reference is not None and all(row.get('snapshots') == partial_reference for row in passed) and len(passed) == len(rows), 'metrics': metrics}
    return result


def main():
    """Select frozen report fixtures or an explicit native clip; never download assets."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, default=Path('/private/tmp/ag-media-events-20261118/qa'))
    parser.add_argument('--baseline', type=Path, default=ROOT / 'data/exports/event-bus-20261118/qa/source-baseline/voice_studio/mcp/media_graph_mcp.py')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--modes', nargs='+', default=['baseline', 'sync', 'fifo', 'priority'])
    parser.add_argument('--gap-ms', type=float, default=50)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--native-source', type=Path)
    parser.add_argument('--native-language', default='en')
    parser.add_argument('--native-registry', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--label', default='sample')
    parser.add_argument('--mode', default='sync')
    parser.add_argument('--case-index', type=int, default=0)
    options = parser.parse_args()
    options.scratch.mkdir(parents=True, exist_ok=True)
    if options.worker:
        worker(options)
        return
    options.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(options.manifest.read_text())
    captured = {'manifestSha256': file_sha(options.manifest), 'sourceFiles': {name: file_sha(ROOT / name)
        for name in SOURCE_FILES if (ROOT / name).is_file()}, 'python': sys.version,
        'repeats': options.repeats, 'modes': options.modes, 'gapMs': options.gap_ms,
        'environment': {key: value for key, value in os.environ.items() if key.startswith('VOICE_')},
        'nativeRegistrySha256': registry_digest(options.native_registry) if options.native_registry else None,
        'benchmarkSha256': file_sha(Path(__file__)),
        'nativeSourceSha256': file_sha(options.native_source) if options.native_source else None}
    (options.output / 'run-manifest.json').write_text(json.dumps(captured, indent=2))
    samples = []
    for repeat in range(options.repeats):
        # Rotate modes across repetitions to reduce fixed warm-cache ordering bias.
        modes = options.modes[repeat % len(options.modes):] + options.modes[:repeat % len(options.modes)]
        for index, case in enumerate(manifest['cases']):
            for mode in modes:
                case_name = 'native' if options.native else case['name']
                label = f"{case_name}-{mode}-{repeat + 1}"
                samples.append(run_isolated(options, label, mode, index, native=options.native))
                (options.output / 'summary.json').write_text(json.dumps(summarize(samples), indent=2))
            if options.native:
                break
    if options.native_registry and registry_digest(options.native_registry) != captured['nativeRegistrySha256']:
        raise SystemExit('Native voice identity seed changed during measurement')
    if options.native_source and file_sha(options.native_source) != captured['nativeSourceSha256']:
        raise SystemExit('Native source changed during measurement')
    changed = [name for name, identity in captured['sourceFiles'].items() if file_sha(ROOT / name) != identity]
    (options.output / 'source-freeze.json').write_text(json.dumps({'changedDuringRun': changed}, indent=2))
    if changed:
        raise SystemExit('Sources changed during measurement; evidence must be rerun')


if __name__ == '__main__':
    main()
