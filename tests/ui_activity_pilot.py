"""Reproduce bounded UI candidate measurements on a caller-selected local clip.

Run as a module with source and output paths. FFmpeg supplies decoded BGR frames;
no network request, recognition model, or source-specific ROI is used.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import threading
from time import perf_counter

import cv2
import numpy as np
import psutil

from voice_studio.vision.ui_activity import UIActivity
from voice_studio.vision.ui_static_regions import _overlap


def sample_rss(stop, processes, peak):
    """Measure simultaneous resident bytes of this Python process and its known decoder."""
    while not stop.is_set():
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        peak[0] = max(peak[0], total)
        stop.wait(.025)


def compare_legacy(summary, legacy, start, end):
    """Compare geometric overlap only; historical color detections are not ground truth."""
    old_panels = {row['id']: row for row in legacy['panels']}
    new_panels = {row['id']: row for row in summary['panels']}
    old = {}
    new = {}
    for layer, panels, target in ((legacy, old_panels, old), (summary, new_panels, new)):
        for row in layer['samples']:
            if row['active'] and start <= row['timeMs'] < end:
                target.setdefault(row['timeMs'], []).append(panels[row['panelId']]['box'])
    expected = matched = extras = 0
    for stamp in sorted(set(old) | set(new)):
        left, right = old.get(stamp, []), new.get(stamp, [])
        expected += len(left)
        matched += sum(any(_overlap(a, b) >= .8 for b in right) for a in left)
        extras += sum(not any(_overlap(a, b) >= .8 for a in left) for b in right)
    return {'legacyObserved': expected, 'legacyGeometricallyRecovered': matched,
            'additionalGenericCandidates': extras,
            'interpretation': 'Additional generic colors/edges are raw candidates, not confirmed activity or speaking.'}


def run(source, output, start_ms, end_ms, *, ffmpeg='ffmpeg', legacy=None):
    """Write candidate evidence and process-tree performance for a 500ms sampling plan."""
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, '-v', 'error', '-ss', str(start_ms/1000), '-i', str(source),
               '-t', str((end_ms-start_ms)/1000), '-an', '-vf', 'fps=2,scale=640:360',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-threads', '2', 'pipe:1']
    started = perf_counter()
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE)
    stop, peak = threading.Event(), [0]
    monitor = threading.Thread(target=sample_rss, args=(stop, [psutil.Process(), psutil.Process(decoder.pid)], peak))
    monitor.start()
    tracker, digest, count, detector_seconds = UIActivity(), sha256(), 0, 0
    try:
        while True:
            raw = decoder.stdout.read(640*360*3)
            if not raw:
                break
            if len(raw) != 640*360*3:
                raise ValueError('Decoder returned a partial BGR frame.')
            digest.update(raw)
            image = np.frombuffer(raw, dtype=np.uint8).reshape(360,640,3)
            before = perf_counter()
            tracker.add(image, {'timeMs':start_ms+count*500, 'shot':0})
            detector_seconds += perf_counter()-before
            count += 1
        status = decoder.wait()
        if status:
            raise RuntimeError(f'FFmpeg failed with status {status}.')
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait()
        stop.set()
        monitor.join()
    summary = tracker.summary()
    stats = {'source':str(source), 'sourceBytes':source.stat().st_size, 'windowMs':[start_ms,end_ms],
             'sampleMs':500, 'frames':count, 'decodeStatus':status, 'decodedBgrSha256':digest.hexdigest(),
             'elapsedSeconds':round(perf_counter()-started,3), 'detectionSeconds':round(detector_seconds,3),
             'meanDetectionMs':round(detector_seconds*1000/count,3) if count else None,
             'peakProcessTreeRssMiB':round(peak[0]/1024**2,2), 'rssSampleMs':25,
             'peakScope':'Simultaneous Python + known FFmpeg child RSS; sampled peak may miss shorter spikes.',
             'panels':len(summary['panels']), 'samples':len(summary['samples']), 'truncated':summary['truncated'],
             'changeTypes':dict(Counter(row['changeType'] for row in summary['samples'])),
             'panelsSummary':summary['panels'],
             'limitations':['One supplied shot/layout for the controlled window.',
                            'Static runs measure sampled ROI stability, not continuous presence.',
                            'No speech/person identity or full-video qualification.']}
    if legacy is not None:
        # Load historical evidence after timing/RSS sampling, so the comparison
        # JSON does not inflate the measured detector/decoder memory footprint.
        legacy_data = json.loads(legacy.read_text()) if isinstance(legacy, Path) else legacy
        stats['legacyComparison'] = compare_legacy(summary, legacy_data, start_ms, start_ms+count*500)
    evidence_path = output/'ui-activity-generic-pilot.json'
    evidence_path.write_text(json.dumps(summary, separators=(',',':')))
    stats['evidenceSha256'] = sha256(evidence_path.read_bytes()).hexdigest()
    (output/'ui-activity-generic-pilot-stats.json').write_text(json.dumps(stats, indent=2))
    return stats


def main():
    """Read explicit local paths and a controlled window; never infer source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start-ms', type=int, default=50000)
    parser.add_argument('--end-ms', type=int, default=180000)
    parser.add_argument('--legacy', type=Path)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    args = parser.parse_args()
    if not 0 <= args.start_ms < args.end_ms:
        parser.error('The window must be positive and ordered.')
    result = run(args.source.resolve(), args.output.resolve(), args.start_ms, args.end_ms,
                 ffmpeg=args.ffmpeg, legacy=args.legacy)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
