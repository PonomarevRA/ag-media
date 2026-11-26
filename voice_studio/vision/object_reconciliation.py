"""Run optional local-part crop measurement in the existing vision environment."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from voice_studio.core.optional_runtime import runtime_environment
from voice_studio.core.worker_process import worker_command
from voice_studio.vision.video_revolver import PYTHON, VISION

# Keep the existing 30-second allowance plus bounded foreground diagnostics and a small margin.
TIMEOUT_SECONDS = 36


def prepare_object_reconciliation(source: Path, report: dict) -> dict:
    """Return a copied report with measured evidence, or an explicit unavailable status.

    Uses the installed vision runtime so the main application needs no
    OpenCV dependency. The child reads at most 48 timestamps and 96 crops; its
    wall time is capped at 36 seconds, including foreground diagnostics. No model
    installation or download occurs.
    """
    if not PYTHON.is_file():
        return unavailable(report, 'vision-runtime-unavailable')
    try:
        with TemporaryDirectory(prefix='object-reconciliation-') as folder:
            request, response = Path(folder) / 'request.json', Path(folder) / 'response.json'
            request.write_text(json.dumps(report), encoding='utf-8')
            command = worker_command(PYTHON, 'voice_studio.vision.object_reconciliation',
                                     '--worker', source, request, response)
            env = runtime_environment('vision', {**os.environ, 'PYTHONPATH': str(VISION)})
            process = subprocess.run(command, env=env, capture_output=True,
                timeout=TIMEOUT_SECONDS, check=False)
            if process.returncode or not response.is_file():
                return unavailable(report, 'measurement-failed')
            return json.loads(response.read_text(encoding='utf-8'))
    except subprocess.TimeoutExpired:
        return unavailable(report, 'measurement-timeout')
    except (OSError, ValueError):
        return unavailable(report, 'measurement-unavailable')


def unavailable(report, status):
    """Keep raw evidence intact when optional processing cannot be completed."""
    result = deepcopy(report)
    result['objectReconciliationEvidence'] = {'basis': 'bounded-source-crop-orb-pilot', 'status': status}
    return result


def main():
    """Consume task-owned JSON files inside the isolated vision worker process."""
    if len(sys.argv) != 5 or sys.argv[1] != '--worker':
        raise SystemExit('Expected --worker source request response')
    from voice_studio.vision.object_reconciliation_parts import prepare_object_reconciliation as measure
    result = measure(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text(encoding='utf-8')))
    Path(sys.argv[4]).write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
