"""Exercise real provider processes, captured worker config and stage cache isolation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio import mod_runtime as runtime
from voice_studio.mods import ModError
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_stage_cache import IncrementalStages
from voice_studio.speech import recognition


class MediaModIntegrationTests(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {'VOICE_MEDIA_MODS_DIR': str(self.root / 'mods')})
        self.environment.start()
        self.old_registry = runtime.REGISTRY
        runtime.REGISTRY = None
        runtime.RESOURCE_CACHE.clear()
        self.registry = runtime.registry()

    def tearDown(self):
        runtime.REGISTRY = self.old_registry
        self.environment.stop()
        self.temporary.cleanup()

    def package(self, identifier, capability, body, *, kind='provider', settings=None, declaration=None, assets=None):
        """Create and explicitly approve a real user package in isolated external state."""
        folder = self.registry.directory / identifier
        folder.mkdir(parents=True)
        manifest = {'apiVersion': 1, 'id': identifier, 'name': identifier, 'version': '1.0.0',
                    'description': 'Test fixture', 'capabilities': [{'id': capability, 'kind': kind, 'entrypoint': 'worker.py:run'}],
                    'settingsSchema': {'type': 'object', 'properties': {'tag': {'type': 'string'}}, 'additionalProperties': False},
                    'settings': settings or {'tag': identifier}, 'license': 'Test fixture', 'provenance': {'source': 'local integration fixture'}}
        if declaration:
            manifest['capabilities'][0].update(declaration)
        for filename, contents in (assets or {}).items():
            (folder / filename).write_text(contents)
        (folder / 'mod.json').write_text(json.dumps(manifest))
        (folder / 'worker.py').write_text('def run(payload, settings):\n' + ''.join('    ' + line + '\n' for line in body.splitlines()))
        row = next(row for row in self.registry.catalog()['mods'] if row['id'] == identifier)
        self.registry.enable(identifier, row['fingerprint'])
        return folder

    def test_all_builtin_capabilities_have_callbacks_and_resource_dispatch(self):
        for row in self.registry.catalog()['mods']:
            for capability in row['manifest']['capabilities']:
                self.assertIn((row['id'], capability['id']), self.registry._callbacks)
        actual = runtime.model_resource('speech.embedding-resource', 'builtin.sherpa-embedding', self.root / 'embedding.onnx')
        self.assertEqual(actual, self.root / 'embedding.onnx')

    def test_selected_builtin_changes_actual_whisper_model_and_supports_keywords(self):
        self.registry.select('speech.asr', 'builtin.whisper-small')
        with patch.object(recognition, 'ready', return_value=True), patch.object(recognition, 'read_whisper_output', return_value={'transcription': []}), patch.object(recognition.subprocess, 'run') as process:
            recognition.run_whisper(jobs=[(self.root / 'a.wav', self.root / 'out')], language='en', options=[], model=recognition.MODEL, threads=4, timeout=10, wait=0)
        command = process.call_args.args[0]
        self.assertEqual(command[command.index('-m') + 1], str(recognition.SMALL_MODEL))

    def test_asr_replacement_reaches_real_media_pipeline_and_additional_stage(self):
        self.package('test.asr', 'speech.asr', "return [{'result': {'language': 'en'}, 'transcription': [{'text': settings['tag'], 'offsets': {'from': 0, 'to': 700}}]}]", settings={'tag': 'replacement was invoked'})
        self.package('test.extra', 'analysis.custom', "return {'source': payload['sourcePath'], 'text': payload['transcript']}", kind='stage')
        self.registry.select('speech.asr', 'test.asr')
        source = self.root / 'source.wav'
        with wave.open(str(source), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\x40\x00' * 16000)
        from voice_studio.speech.media_transcript import transcribe_media
        result = transcribe_media(self.root, 'en', source, allow_empty=True)
        self.assertEqual(result['text'], 'replacement was invoked')
        self.assertEqual(result['modStages'][0]['result']['text'], result['text'])
        self.assertEqual(result['modStages'][0]['result']['source'], str(source))
        self.assertEqual(result['modStages'][0]['state'], 'done')
        self.assertTrue(any(row['modId'] == 'test.asr' for row in result['modProvenance']))

    def test_snapshot_survives_settings_edits_in_thread_and_real_worker(self):
        self.package('test.audio', 'audio.classification', "return {'tag': settings['tag']}", settings={'tag': 'captured'})
        self.registry.select('audio.classification', 'test.audio')
        with runtime.execution_scope() as snapshot:
            environment = runtime.worker_environment()
            self.registry.configure('test.audio', {'tag': 'new settings'})
            with ThreadPoolExecutor(max_workers=1) as pool:
                value = runtime.submit_with_context(pool, lambda: runtime.current_snapshot().invoke('audio.classification', {})).result()
            self.assertEqual(value, {'tag': 'captured'})
        code = "import json; from voice_studio.mod_runtime import current_snapshot; print(json.dumps(current_snapshot().invoke('audio.classification', {})))"
        raw = subprocess.check_output([sys.executable, '-c', code], env=environment, cwd=Path(__file__).resolve().parents[1], timeout=15)
        self.assertEqual(json.loads(raw), {'tag': 'captured'})
        self.assertEqual(self.registry.snapshot().invoke('audio.classification', {}), {'tag': 'new settings'})

    def test_only_changed_capability_invalidates_its_layer(self):
        self.package('test.audio', 'audio.classification', "return {}")
        counts = {'speech': 0, 'audio': 0}
        def measure(name):
            counts[name] += 1
            return {'value': counts[name]}
        def run():
            with runtime.execution_scope():
                cache = IncrementalStages(AnalysisRevolver(self.root / 'cache'), 'fixed-code')
                for name in counts:
                    cache.run(name, {}, None, lambda: measure(name), lambda value: 'value' in value)
        run()
        self.registry.select('audio.classification', 'test.audio')
        run()
        self.assertEqual(counts, {'speech': 1, 'audio': 2})

    def test_disabled_selected_provider_never_falls_back(self):
        self.package('test.asr', 'speech.asr', "return []")
        self.registry.select('speech.asr', 'test.asr')
        row = next(row for row in self.registry.catalog()['mods'] if row['id'] == 'test.asr')
        self.registry.enable('test.asr', row['fingerprint'], False)
        with patch.object(recognition, 'ready') as ready, self.assertRaises(ModError):
            recognition.run_whisper([], 'en', [], None, 4, 10, 0)
        ready.assert_not_called()

    def test_blank_selection_does_not_execute_additional_stage(self):
        self.package('test.extra', 'analysis.custom', "raise RuntimeError('must not execute')", kind='stage')
        @runtime.analysis_run
        def fixture(source, selected_stages=None):
            return {'text': 'kept'}
        result = fixture(self.root / 'source', selected_stages=[])
        self.assertEqual(result['modStages'], [{'modId': 'test.extra', 'capability': 'analysis.custom', 'nodeId': 'mod:test.extra:analysis.custom', 'dependsOn': [], 'state': 'pending'}])

    def test_resource_replacement_runs_once_for_repeated_native_constructors(self):
        target = self.root / 'compatible.onnx'
        target.write_bytes(b'test')
        marker = self.root / 'invocations.txt'
        body = "from pathlib import Path\np = Path(" + repr(str(marker)) + ")\np.write_text(p.read_text() + 'x' if p.exists() else 'x')\nreturn {'modelPath': " + repr(str(target)) + "}"
        self.package('test.resource', 'vision.faces', body)
        self.registry.select('vision.faces', 'test.resource')
        for _ in range(3):
            self.assertEqual(runtime.model_resource('vision.faces', 'builtin.yunet', self.root / 'missing.onnx'), target)
        self.assertEqual(marker.read_text(), 'x')

    def test_scene_batch_replacement_is_consumed_without_native_scene_runtime(self):
        self.package('test.scene', 'vision.scene-batch', "return {'status': 'ok', 'frames': [{'timeMs': 300, 'objects': [{'label': 'cup', 'score': .9, 'box': [.1,.2,.3,.4]}], 'text': []}], 'providerTag': settings['tag']}")
        self.registry.select('vision.scene-batch', 'test.scene')
        from voice_studio.vision.video_scene import review_scene
        from voice_studio.graph.media_graph_model import build_graph
        with patch('voice_studio.vision.video_scene.probe_video') as native_probe:
            scene = review_scene(self.root / 'missing.mp4', self.root, 'mov')
        native_probe.assert_not_called()
        graph = build_graph({'sceneEvidence': scene, 'segments': []}, kind='video', duration_ms=1000)
        self.assertTrue(any(row['kind'] == 'object_observation' for row in graph['nodes']))

    def test_additional_result_checkpoint_is_retained_by_empty_selection(self):
        self.package('test.extra', 'analysis.custom', "return {'text': payload['transcript']}", kind='stage')
        source = self.root / 'source'
        source.write_bytes(b'fixed')
        @runtime.analysis_run
        def fixture(root, source, selected_stages=None):
            return {'text': 'same evidence'}
        first = fixture(self.root, source)
        second = fixture(self.root, source, selected_stages=[])
        self.assertEqual(first['modStages'][0]['state'], 'done')
        self.assertEqual(second['modStages'][0]['state'], 'reused')
        self.assertEqual(first['modStages'][0]['result'], second['modStages'][0]['result'])

    def test_http_mod_routes_use_existing_origin_guards_and_execute_fixture(self):
        import http.client
        from http.server import ThreadingHTTPServer
        from threading import Thread
        from types import SimpleNamespace
        from voice_studio.app.server import Handler
        service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        service.studio = SimpleNamespace(root=self.root)
        thread = Thread(target=service.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection('127.0.0.1', service.server_port, timeout=5)
        origin = 'http://127.0.0.1:' + str(service.server_port)
        headers = {'Origin': origin, 'X-Voice-Studio': '1', 'Content-Type': 'application/json'}
        def post(action, value, allowed=True):
            connection.request('POST', '/api/mods/' + action, json.dumps(value), headers if allowed else {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        try:
            connection.request('GET', '/api/mods/contracts')
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(json.loads(response.read())['contracts'])
            connection.request('GET', '/api/mods/pipeline')
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(len(json.loads(response.read())['nodes']), 8)
            self.assertEqual(post('pipeline', {'dependencies': {}, 'expectedRevision': 0})[0], 200)
            self.assertEqual(post('pipeline', {'dependencies': {}, 'expectedRevision': 0})[0], 400)
            self.assertEqual(post('rescan', {}, allowed=False)[0], 403)
            status, catalog = post('template', {'id': 'http.fixture'})
            self.assertEqual(status, 200)
            row = next(row for row in catalog['mods'] if row['id'] == 'http.fixture')
            self.assertFalse(row['enabled'])
            self.assertEqual(post('enable', {'id': row['id'], 'fingerprint': row['fingerprint'], 'enabled': True})[0], 200)
            status, value = post('export-test', {'id': row['id'], 'capability': 'analysis.word-count', 'payload': {'transcript': 'one two'}})
            self.assertEqual(status, 200)
            self.assertEqual(value['result']['wordCount'], 2)
            self.assertEqual(post('rescan', {})[0], 200)
        finally:
            connection.close()
            service.shutdown()
            service.server_close()
            thread.join(timeout=5)

    def test_synthesis_keyword_dispatch_and_implicit_noise_fallback(self):
        from voice_studio.synthesis.engines import generate
        self.registry.select('synthesis.tts', 'builtin.qwen')
        with patch('voice_studio.synthesis.engines.ffmpeg') as render:
            generate(model='f5', job={'output': str(self.root / 'tone.wav')}, demo=True)
        self.assertIn('sine=frequency=440:duration=1.5', render.call_args.args[0])
        from voice_studio.audio import audio_processing
        weights = self.root / 'cb.rnnn'
        weights.write_bytes(b'test')
        with patch.object(audio_processing, 'DENOISE_MODEL', weights), patch.object(audio_processing, 'ffmpeg', side_effect=[ValueError('unsupported'), None]) as ffmpeg:
            self.assertEqual(audio_processing.denoise_wave(self.root / 'source.wav', self.root / 'clean.wav'), 'fft')
            self.assertEqual(ffmpeg.call_count, 2)
        (self.root / 'denoise.rnnn').unlink()
        self.registry.select('audio.denoise', 'builtin.rnnoise')
        with patch.object(audio_processing, 'DENOISE_MODEL', weights), patch.object(audio_processing, 'ffmpeg', side_effect=ValueError('unsupported')) as ffmpeg, self.assertRaises(ValueError):
            audio_processing.denoise_wave(self.root / 'source.wav', self.root / 'explicit.wav')
        self.assertEqual(ffmpeg.call_count, 1)

    def test_user_builtin_prefix_is_dispatched_by_origin(self):
        self.package('builtin.custom', 'speech.asr', "return [{'transcription': [], 'marker': 'user-code'}]")
        self.registry.select('speech.asr', 'builtin.custom')
        with patch.object(recognition, 'ready') as native:
            result = recognition.run_whisper([(self.root / 'input.wav', self.root / 'output')], 'en', [], None, 2, 10, 0)
        native.assert_not_called()
        self.assertEqual(result[0]['marker'], 'user-code')

    def test_mod_output_and_settings_survive_graph_snapshot_reopen(self):
        from hashlib import sha256
        from voice_studio.graph.media_graph_store import MediaGraphStore
        from voice_studio.mcp.media_graph_mcp import build_snapshot
        self.package('test.extra', 'analysis.custom', "return {'text': payload['transcript']}", kind='stage', settings={'tag': 'captured-settings'})
        source = self.root / 'source.wav'
        with wave.open(str(source), 'wb') as audio:
            audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            audio.writeframes(bytes(32000))
        @runtime.analysis_run
        def fixture(root, source):
            return {'text': 'persisted', 'segments': [{'startMs': 0, 'endMs': 700, 'text': 'persisted'}], 'language': 'en', 'durationMs': 1000}
        report = fixture(self.root, source)
        report['sourceDigest'] = sha256(source.read_bytes()).hexdigest()
        path = self.root / 'analysis.json'
        path.write_text(json.dumps(report))
        store = MediaGraphStore(self.root / 'graphs')
        with patch('voice_studio.mcp.media_graph_mcp.recover_profile_context'), patch('voice_studio.graph.media_graph_ingest.backfill_dialogue_pauses'):
            snapshot = build_snapshot(store, self.root, str(source), 'en', str(path), False, False, None)
        reopened = MediaGraphStore(self.root / 'graphs').overview(snapshot['snapshotId'])
        self.assertEqual(reopened['provenance']['modStages'][0]['result'], {'text': 'persisted'})
        entry = next(item for item in reopened['provenance']['modProvenance'] if item['modId'] == 'test.extra')
        self.assertEqual(entry['settings'], {'tag': 'captured-settings'})
        self.assertEqual(len(entry['fingerprint']), 64)

    def test_selected_resource_bypasses_missing_original_vocos_embedding_and_clothing(self):
        from hashlib import sha256
        from types import SimpleNamespace
        from voice_studio.synthesis import engines
        from voice_studio.speech import speaker_diarization, voice_profiles
        from voice_studio.vision import clothing_review
        resources = {}
        for capability, filename in [('synthesis.vocoder-resource', 'pytorch_model.bin'), ('speech.embedding-resource', 'embedding.onnx'),
                                     ('vision.objects', 'nanodet.onnx'), ('vision.segmentation', 'selfie.tflite')]:
            target = self.root / 'selected' / filename
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(('selected-' + filename).encode())
            identifier = 'test.' + filename.replace('_', '-').replace('.', '-')
            self.package(identifier, capability, "return {'modelPath': " + repr(str(target)) + "}")
            self.registry.select(capability, identifier)
            resources[capability] = target
        absent = self.root / 'missing'
        absent.mkdir()
        original_model_path = engines.model_path
        def model_path(*parts):
            return absent / 'pytorch_model.bin' if parts[0] == 'vocos' else original_model_path(*parts)
        job = {'output': str(self.root / 'output.wav'), 'text': 'text', 'references': [], 'seed': 1}
        with patch.object(engines, 'model_path', side_effect=model_path), patch.object(engines.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as process:
            engines.generate('f5', job, False)
        request = json.loads(process.call_args.kwargs['input'])
        self.assertEqual(request['vocoder'], str(resources['synthesis.vocoder-resource'].parent))
        (absent / 'segmentation.onnx').write_bytes(b'segmentation')
        with patch.object(speaker_diarization, 'MODEL_PATH', absent), patch.object(speaker_diarization.importlib.util, 'find_spec', return_value=object()):
            self.assertTrue(speaker_diarization.ready())
            self.assertEqual(voice_profiles.local_encoder_id(), 'sherpa-3d-speaker:' + sha256(resources['speech.embedding-resource'].read_bytes()).hexdigest())
        runtime_file = self.root / 'python'
        runtime_file.touch()
        with patch.object(clothing_review, 'MP_PYTHON', runtime_file), patch.object(clothing_review, 'BODY_MODEL', absent / 'body.onnx'), patch.object(clothing_review, 'MASK_MODEL', absent / 'mask.tflite'), patch.object(clothing_review, 'borderline_pairs', return_value=[]) as pairs, patch.object(clothing_review, 'choose_sample_tracks', return_value={}):
            result = clothing_review.review_clothing('source', [], {}, {}, 320, 240)
        pairs.assert_called_once()
        self.assertEqual(result[2]['status'], 'skipped')

    def test_readiness_uses_selected_builtin_in_both_availability_directions(self):
        from voice_studio.synthesis import engines
        self.registry.select('synthesis.tts', 'builtin.qwen')
        with patch.object(engines.commercial_voice, 'ready', side_effect=lambda model: model == 'qwen'):
            status = engines.availability(False, 'en')
        self.assertTrue(status['chatterbox']['ready'])
        self.assertEqual(status['chatterbox']['actualProvider'], 'builtin.qwen')
        with patch.object(engines.commercial_voice, 'ready', side_effect=lambda model: model == 'chatterbox'):
            status = engines.availability(False, 'en')
        self.assertFalse(status['chatterbox']['ready'])
        self.assertFalse(status['qwen']['ready'])

    def test_builtin_whisper_thread_settings_change_command_and_unset_preserves_default(self):
        """The Settings control changes -t; clearing the override restores caller thread planning."""
        self.registry.configure('builtin.whisper-base', {'threads': 1})
        def command_threads():
            with patch.object(recognition, 'ready', return_value=True), patch.object(recognition, 'read_whisper_output', return_value={'transcription': []}), patch.object(recognition.subprocess, 'run') as process:
                recognition.run_whisper([(self.root / 'a.wav', self.root / 'out')], 'en', [], None, 4, 10, 0)
            command = process.call_args.args[0]
            return command[command.index('-t') + 1]
        self.assertEqual(command_threads(), '1')
        self.registry.configure('builtin.whisper-base', {})
        self.assertEqual(command_threads(), '4')
        with self.assertRaises(ModError):
            self.registry.configure('builtin.whisper-base', {'threads': 0})
