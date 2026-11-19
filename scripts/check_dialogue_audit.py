#!/usr/bin/env python3
"""Run the standalone local dialogue audit; Ctrl-C cancels the ASR child cleanly."""

import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_studio.speech.dialogue_audit_runner import AuditCancelled, run_audit


def main():
    """Parse bounded audit inputs and print machine-readable artifact metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--turns', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path, help='full mono 16 kHz PCM WAV')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--visual', type=Path)
    parser.add_argument('--rerun', action='store_true', help='optional installed local Whisper')
    parser.add_argument('--max-windows', type=int, default=6)
    parser.add_argument('--max-audio-ms', type=int, default=30000)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    state = {'cancelled': False}

    def cancel(signum, frame):
        """Let the runner kill its owned child and retain previous successful output."""
        state['cancelled'] = True

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    try:
        result = run_audit(args.report, args.turns, args.source, args.output, visual_path=args.visual,
                           rerun=args.rerun, max_windows=args.max_windows, max_audio_ms=args.max_audio_ms,
                           timeout_seconds=args.timeout, model_path=args.model,
                           cancelled=lambda: state['cancelled'],
                           emit=lambda event: print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True))
    except (AuditCancelled, TimeoutError, ValueError, OSError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({key: result[key] for key in ('status', 'summary', 'artifactPath', 'runKey', 'cacheHit',
                                                 'elapsedSeconds')}, ensure_ascii=False))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
