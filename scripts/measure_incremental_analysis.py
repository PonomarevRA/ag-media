"""Measure one real standard-MCP build and fingerprint immutable layer data.

Run each label in a fresh process for comparable sampled RSS. Model/runtime caches
are OS-managed; this measures a cold *analysis checkpoint*, not a cold machine.
"""

import argparse
import asyncio
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
import zlib

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from check_media_mcp import Probe, memory_samples

ROOT = Path(__file__).resolve().parents[1]


def semantic(value):
    """Exclude measured execution durations only; keep every content/timecode/confidence field."""
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items() if key not in ('elapsedMs', 'decodeMs')}
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


def digest(value):
    """Hash canonical JSON so key ordering cannot imitate a quality regression."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def layers(snapshot):
    """Hash every node field and edge relation, grouping by stable graph kind."""
    path = ROOT / 'data/studio/media-graphs' / (snapshot + '.sqlite')
    grouped = defaultdict(list)
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as database:
        manifest = json.loads(database.execute('SELECT value FROM manifest').fetchone()[0])
        for identity, kind, start, end, label, data, description in database.execute(
                'SELECT id,kind,start_ms,end_ms,label,data,description FROM nodes ORDER BY id'):
            grouped[kind].append([identity, start, end, label, semantic(json.loads(data)), description])
        edges = list(database.execute('SELECT source,relation,target FROM edge_refs ORDER BY source,relation,target'))
        faces = [[bucket, json.loads(zlib.decompress(payload))] for bucket, payload in
                 database.execute('SELECT bucket,payload FROM face_chunks ORDER BY bucket')]
    return {'nodes': {kind: {'count': len(rows), 'sha256': digest(rows)} for kind, rows in grouped.items()},
            'edges': {'count': len(edges), 'sha256': digest(edges)}, 'faceFramesSha256': digest(faces),
            'coverageSha256': digest(manifest['coverage']), 'faceSampleMs': manifest.get('faceSampleMs'),
            'speakerScope': manifest.get('speakerScope')}


async def measure(options):
    """Keep the server alive through final publication, sampling the whole child process tree."""
    samples, events = [], []
    monitor = asyncio.create_task(memory_samples(samples))
    parameters = StdioServerParameters(command=sys.executable, args=[str(ROOT / 'voice_mcp.py'),
        '--graph-root', str(ROOT / 'data/studio/media-graphs'), '--studio-data', str(ROOT / 'data/studio')], cwd=ROOT,
        env={'VOICE_SCENE_OCR_WORKERS': str(options.ocr_workers)} if options.ocr_workers else None)
    request = {'file_path': str(options.source.resolve()), 'language': options.language, 'visual': True,
               'scene': True, 'progressive': True, 'object_sampling': options.sampling}
    result = {'label': options.label, 'request': request,
              'environmentOverrides': {'VOICE_SCENE_OCR_WORKERS': str(options.ocr_workers)} if options.ocr_workers else {}}
    started = perf_counter()
    try:
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            await session.initialize()
            probe = Probe(session)
            queued = await probe.call('media_graph_build', **request)
            if not queued:
                raise RuntimeError('Build request failed')
            seen = set()
            async with asyncio.timeout(900):
                while True:
                    job = await probe.call('media_graph_job', job_id=queued['jobId'])
                    if not job:
                        raise RuntimeError('Job read failed')
                    snapshot = job.get('snapshot') or job.get('latestSnapshot')
                    if isinstance(snapshot, dict) and snapshot['snapshotId'] not in seen:
                        seen.add(snapshot['snapshotId'])
                        events.append({'seconds': round(perf_counter() - started, 3),
                                       'snapshotId': snapshot['snapshotId'], 'status': job['status']})
                    if job['status'] in ('done', 'failed'):
                        result.update(seconds=round(perf_counter() - started, 3), job=job)
                        if job['status'] == 'done':
                            result['layers'] = layers(snapshot['snapshotId'])
                            if options.compare:
                                before = layers(options.compare)
                                result['comparison'] = {'snapshotId': options.compare, 'equal': before == result['layers'],
                                    'changedKinds': [kind for kind in sorted(set(before['nodes']) | set(result['layers']['nodes']))
                                                     if before['nodes'].get(kind) != result['layers']['nodes'].get(kind)],
                                    'edgesEqual': before['edges'] == result['layers']['edges'],
                                    'faceFramesEqual': before['faceFramesSha256'] == result['layers']['faceFramesSha256'],
                                    'coverageEqual': before['coverageSha256'] == result['layers']['coverageSha256']}
                        break
                    await asyncio.sleep(.2)
            result['calls'] = probe.calls
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        result.update(events=events, firstReadableSeconds=events[0]['seconds'] if events else None,
                      peakTreeRssBytes=max(samples, default=None), rssSamples=len(samples),
                      rssScope='50ms sampled sum client + MCP + children RSS; not unique physical memory')
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'label': options.label, 'seconds': result.get('seconds'),
                      'peakMiB': result['peakTreeRssBytes'] / 1048576 if result['peakTreeRssBytes'] else None,
                      'comparison': result.get('comparison'), 'status': result.get('job', {}).get('status')}))


def main():
    """Select an explicit local corpus source; never guess or download a substitute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--language', default='bn')
    parser.add_argument('--sampling', default='small-detail', choices=('basic', 'small-detail', 'every-third'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--compare')
    parser.add_argument('--ocr-workers', type=int, choices=range(1, 9))
    asyncio.run(measure(parser.parse_args()))


if __name__ == '__main__':
    main()
