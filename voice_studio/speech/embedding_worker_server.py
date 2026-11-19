"""Native-only JSONL service; model resources are resolved exclusively by its parent."""

import json
import os
from pathlib import Path
import sys
from threading import Thread
from time import sleep

from voice_studio.speech.embedding_protocol import (
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, file_signature, validate_spans,
)


def _parent_watch(parent_pid):
    """Exit an orphan even if its native inference has not returned to the stdin loop."""
    while True:
        sleep(.5)
        if os.getppid() != parent_pid:
            os._exit(0)


class NativeEmbeddingSession:
    """Own a single extractor and replace it when the resolved artifact/config changes."""

    def __init__(self, factory=None, infer=None):
        """Accept kernel hooks only for direct cache tests; production uses unchanged Sherpa."""
        if factory is None or infer is None:
            from voice_studio.speech.speaker_review import native_embedding_extractor, infer_embeddings
            factory = factory or native_embedding_extractor
            infer = infer or infer_embeddings
        self.factory = factory
        self.infer = infer
        self.extractor = None
        self.identity = None
        self.model_loads = 0

    def execute(self, request):
        """Use parent-selected absolute weights and reject changes during load or inference."""
        validate_spans(request['spans'])
        model_path = Path(request['model_path'])
        identity = request['identity']
        if not model_path.is_absolute() or not isinstance(identity, dict):
            raise ValueError('Embedding worker requires a resolved model identity')
        if identity.get('signature') != file_signature(model_path):
            raise ValueError('Embedding model changed before inference')
        cache_key = json.dumps([str(model_path), identity], sort_keys=True, separators=(',', ':'))
        if self.identity != cache_key:
            # Drop the old native object before allocating its replacement.
            self.extractor = None
            self.identity = None
            extractor = self.factory(model_path)
            if identity['signature'] != file_signature(model_path):
                raise ValueError('Embedding model changed while loading')
            self.extractor = extractor
            self.identity = cache_key
            self.model_loads += 1
        vectors = self.infer(Path(request['wav']), request['spans'], extractor=self.extractor)
        if identity['signature'] != file_signature(model_path):
            raise ValueError('Embedding model changed during inference')
        return {'id': request['id'], 'vectors': vectors, 'model_loads': self.model_loads}


def serve(stdin=None, stdout=None):
    """Serve bounded JSONL until parent EOF; errors produce one reply and end the child."""
    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout.buffer if stdout is None else stdout
    parent_pid = os.environ.get('VOICE_EMBEDDING_PARENT_PID')
    if parent_pid:
        Thread(target=_parent_watch, args=(int(parent_pid),), daemon=True).start()
    session = NativeEmbeddingSession()
    while True:
        line = stdin.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        request = None
        failed = False
        try:
            if len(line) > MAX_REQUEST_BYTES or not line.endswith(b'\n'):
                raise ValueError('Embedding request exceeds limit')
            request = json.loads(line)
            if not isinstance(request, dict) or type(request.get('id')) is not int:
                raise ValueError('Invalid embedding request identity')
            response = session.execute(request)
            payload = json.dumps(response, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError('Embedding response exceeds limit')
        except Exception as error:
            failed = True
            request_id = request.get('id') if isinstance(request, dict) else None
            payload = json.dumps({'id': request_id, 'error': str(error)[:2000]}).encode('utf-8') + b'\n'
        stdout.write(payload)
        stdout.flush()
        if failed:
            return
