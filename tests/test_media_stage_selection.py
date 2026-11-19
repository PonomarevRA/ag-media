"""Verify incremental work actions through cache and the real file-pipeline entry point."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_stage_cache import IncrementalStages
from voice_studio.graph.media_stage_selection import StageSelection, query_stage_names
from voice_studio.speech.media_transcript import transcribe_media


class StageSelectionTests(TestCase):
    def test_validation_dependencies_and_empty_query(self):
        """Only known stage lists are accepted, and dependencies do not force recomputation."""
        self.assertEqual(StageSelection(['people']).enabled, {'people', 'visual', 'speech'})
        self.assertEqual(StageSelection(['object_review']).enabled, {'object_review'})
        self.assertEqual(query_stage_names({'selected_stages': ['']}, 'selected_stages'), [])
        self.assertIsNone(query_stage_names({}, 'selected_stages'))
        for invalid in ('speech', ['unknown'], [1], {'speech': True}):
            with self.assertRaises(ValueError):
                StageSelection(invalid)
        with self.assertRaises(ValueError):
            StageSelection([], ['unknown'])

    def test_nonselected_reuses_previous_settings_and_invalidates_only_changed_inputs(self):
        """Pointers retain successful raw checkpoints without copying their reports."""
        with TemporaryDirectory() as folder:
            revolver = AnalysisRevolver(Path(folder))
            calls = []
            def operation():
                calls.append('scene')
                return {'result': 'old'}
            initial = IncrementalStages(revolver, 'revision')
            initial.run('scene', {'sampling': 'old'}, {'timeline': 1}, operation, bool)
            selection = StageSelection([])
            cache = IncrementalStages(revolver, 'revision', selection)
            self.assertEqual(cache.run('scene', {'sampling': 'new'}, {'timeline': 1}, operation, bool), {'result': 'old'})
            self.assertEqual(selection.states['scene']['state'], 'reused')
            self.assertEqual(cache.run('scene', {}, {'timeline': 2}, operation, bool), {})
            self.assertEqual(selection.states['scene']['state'], 'invalidated')
            self.assertEqual(calls, ['scene'])

    def test_dependency_keeps_completed_settings_until_explicitly_selected(self):
        """Selecting sound tags does not change the user's previously measured speech."""
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            IncrementalStages(AnalysisRevolver(directory), 'revision').run(
                'speech', {'quality': True}, None, lambda: {'text': 'original'}, bool)
            selection = StageSelection(['audio'])
            cache = IncrementalStages(AnalysisRevolver(directory), 'revision', selection)
            result = cache.run('speech', {'quality': False}, None, lambda: self.fail('ASR ran'), bool)
            self.assertEqual(result, {'text': 'original'})
            self.assertTrue(selection.states['speech']['dependency'])
            self.assertEqual(selection.states['speech']['state'], 'reused')

    def test_corrupt_dependency_pointer_does_not_cache_under_old_settings(self):
        """A cold dependency computes with current arguments and records that exact identity."""
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            initial = IncrementalStages(AnalysisRevolver(directory), 'revision')
            initial.run('speech', {'quality': True}, None, lambda: {'quality': True}, bool)
            old_key = initial.manifest['layers']['speech']['key']
            (directory / 'speech' / old_key / 'complete.json.gz').unlink()
            cache = IncrementalStages(AnalysisRevolver(directory), 'revision', StageSelection(['audio']))
            result = cache.run('speech', {'quality': False}, None, lambda: {'quality': False}, bool)
            self.assertEqual(result, {'quality': False})
            self.assertEqual(cache.manifest['layers']['speech']['settings'], {'quality': False})
            self.assertNotEqual(cache.manifest['layers']['speech']['key'], old_key)

    def test_force_unchanged_output_reuses_downstream_changed_output_invalidates(self):
        """Force bypasses only the requested worker; downstream depends on actual result bytes."""
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            initial = IncrementalStages(AnalysisRevolver(directory), 'revision')
            initial.run('speech', {}, None, lambda: {'text': 'one'}, bool)
            initial.run('audio', {}, {'text': 'one'}, lambda: {'audio': 1}, bool)
            selection = StageSelection([], ['speech'])
            cache = IncrementalStages(AnalysisRevolver(directory), 'revision', selection)
            calls = []
            speech = cache.run('speech', {}, None, lambda: calls.append('speech') or {'text': 'one'}, bool)
            self.assertEqual(cache.run('audio', {}, speech, lambda: self.fail('unselected audio ran'), bool), {'audio': 1})
            self.assertEqual(calls, ['speech'])
            self.assertEqual(selection.states['speech']['state'], 'done')
            self.assertEqual(selection.states['audio']['state'], 'reused')
            speech = cache.run('speech', {}, None, lambda: {'text': 'two'}, bool)
            self.assertEqual(cache.run('audio', {}, speech, lambda: self.fail('invalidated audio ran'), bool), {})
            self.assertEqual(selection.states['audio']['state'], 'invalidated')

    def test_empty_selection_does_not_compute_or_cache_placeholder(self):
        """Pending is distinguishable from a successful empty report."""
        with TemporaryDirectory() as folder:
            cache = IncrementalStages(AnalysisRevolver(Path(folder)), 'revision', StageSelection([]))
            value = cache.run('speech', {}, None, lambda: self.fail('pending worker ran'), bool,
                              fallback={'segments': []})
            self.assertEqual(value, {'segments': []})
            self.assertFalse(cache.manifest['layers'])
            self.assertEqual(cache.selection.states['speech']['state'], 'pending')

    def test_pipeline_runs_selected_adds_scene_and_forces_without_replaying_speech(self):
        """Exercise source-scoped checkpoint reuse and lazy decode through production orchestration."""
        module = 'voice_studio.speech.media_transcript.'
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            stack.enter_context(patch('voice_studio.graph.media_stage_cache.analysis_fingerprint', return_value='revision'))
            stack.enter_context(patch('voice_studio.graph.media_graph_ingest.probe_source', return_value={'durationMs': 1000}))
            stack.enter_context(patch(module + 'audio_offset', return_value=0))
            decode = stack.enter_context(patch(module + 'ffmpeg', side_effect=lambda args, **kwargs: Path(args[-1]).write_bytes(b'pcm')))
            base = {'segments': [{'startMs': 0, 'endMs': 200, 'text': 'hello'}], 'text': 'hello',
                    'srt': '', 'durationMs': 1000, 'language': 'en'}
            speech = stack.enter_context(patch(module + 'transcribe_wav', side_effect=lambda *args, **kwargs: deepcopy(base)))
            audio = stack.enter_context(patch(module + 'analyze_audio_scene', return_value={'audioLayers': [], 'soundAnalysisStatus': 'done'}))
            scene = stack.enter_context(patch(module + 'review_scene', return_value={'status': 'ok', 'frames': [], 'shotColors': []}))
            stack.enter_context(patch(module + 'build_moodbook', return_value=None))
            stack.enter_context(patch(module + 'object_layers', return_value={'objectProfiles': [{'id': 'preserved'}]}))
            stack.enter_context(patch(module + 'attach_person_motion', return_value=False))
            stack.enter_context(patch(module + 'reconcile_names', return_value=False))
            def run(selected, forced=None, **settings):
                return transcribe_media(root, 'en', source, revolver=AnalysisRevolver(root / 'cache'),
                                        selected_stages=selected, force_stages=forced, **settings)
            pending = run([])
            self.assertEqual(pending['analysisStages']['speech']['state'], 'pending')
            self.assertEqual((speech.call_count, audio.call_count, scene.call_count, decode.call_count), (0, 0, 0, 0))
            first = run(['speech'])
            self.assertEqual(first['analysisStages']['audio']['state'], 'pending')
            second = run(['scene'])
            self.assertEqual(second['segments'], first['segments'])
            self.assertEqual(second['objectProfiles'], [{'id': 'preserved'}])
            third = run(['audio'])
            self.assertEqual(third['objectProfiles'], second['objectProfiles'])
            self.assertEqual(third['analysisStages']['scene']['state'], 'reused')
            forced = run(['scene'], ['scene'])
            self.assertEqual(forced['segments'], first['segments'])
            self.assertEqual((speech.call_count, audio.call_count, scene.call_count), (1, 1, 2))
            self.assertEqual(forced['analysisStages']['audio']['state'], 'reused')
            self.assertEqual(forced['analysisStages']['scene']['state'], 'done')
            with patch('voice_studio.vision.object_reconciliation.prepare_object_reconciliation',
                       side_effect=AssertionError('Explicit empty selection must not run object review')):
                passive = run([], object_reconciliation=True)
            self.assertEqual(passive['analysisStages']['object_review']['state'], 'pending')

    def test_bounded_audio_windows_add_and_force_sound_without_asr(self):
        """Long sources keep every window's completed speech when only sound tags are added."""
        from voice_studio.graph.media_graph_ingest import analyze_windows, SPEECH_WINDOW_MS
        duration = SPEECH_WINDOW_MS + 1000
        metadata = {'kind': 'audio', 'durationMs': duration,
                    'streams': [{'kind': 'audio', 'index': 0, 'startMs': 0}]}
        module = 'voice_studio.graph.media_graph_ingest.'
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            source = root / 'long.wav'
            source.write_bytes(b'long-fixture')
            decode = stack.enter_context(patch(module + 'extract_speech',
                side_effect=lambda source, work, start, end, audio: (work / 'wav', 0, end - start)))
            speech = stack.enter_context(patch(module + 'transcribe_wav',
                side_effect=lambda *args, **kwargs: {'segments': [{'startMs': 0, 'endMs': 200, 'text': 'stable'}]}))
            audio = stack.enter_context(patch(module + 'analyze_audio_scene',
                return_value={'audioLayers': [{'startMs': 0, 'endMs': 500, 'kind': 'music'}], 'soundAnalysisStatus': 'done'}))
            def run(selected, forced=None):
                return analyze_windows(root, source, metadata, 'en', False, False, None,
                                       revolver=AnalysisRevolver(root / 'cache'), cache_revision='revision',
                                       selection=StageSelection(selected, forced), speakers=False)
            pending = run([])
            self.assertTrue(all(row['status'] == 'pending' for row in pending['coverage']['speech']))
            self.assertEqual(decode.call_count, 0)
            first = run(['speech'])
            second = run(['audio'])
            third = run(['audio'], ['audio'])
            self.assertEqual(first['segments'], second['segments'])
            self.assertEqual(second['segments'], third['segments'])
            self.assertEqual((speech.call_count, audio.call_count), (2, 4))
            self.assertEqual(third['analysisStages']['speech']['state'], 'reused')
            self.assertEqual(third['analysisStages']['audio']['state'], 'done')

    def test_final_object_review_is_retained_on_unrelated_run(self):
        """An optional final review is cache-only on the next audio-only invocation."""
        from voice_studio.graph.media_object_review import review_objects
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'video.mp4'
            scene = {'status': 'ok', 'frames': [{'timeMs': 0, 'objects': [{'label': 'car'}]}]}
            def measured(source, report):
                report['sceneEvidence']['frames'][0]['objects'][0]['objectPartMatches'] = [{'timeMs': 0}]
                report['objectReconciliationEvidence'] = {'status': 'measured'}
                return report
            with patch('voice_studio.vision.object_reconciliation.prepare_object_reconciliation', side_effect=measured) as worker, \
                    patch('voice_studio.graph.media_graph_model.object_layers', return_value={'objectProfiles': [{'id': 'merged'}]}):
                first = {'sceneEvidence': deepcopy(scene), 'durationMs': 1000}
                cache = IncrementalStages(AnalysisRevolver(Path(folder) / 'cache'), 'revision', StageSelection(['object_review']))
                review_objects(source, first, cache, emit=lambda event: None)
                second = {'sceneEvidence': deepcopy(scene), 'durationMs': 1000}
                cache = IncrementalStages(AnalysisRevolver(Path(folder) / 'cache'), 'revision', StageSelection(['audio']))
                review_objects(source, second, cache, emit=lambda event: None)
                self.assertEqual(worker.call_count, 1)
                empty_cache = IncrementalStages(AnalysisRevolver(Path(folder) / 'empty-cache'), 'revision', StageSelection([]))
                untouched = {'sceneEvidence': deepcopy(scene), 'durationMs': 1000}
                review_objects(source, untouched, empty_cache, emit=lambda event: None)
                self.assertEqual(worker.call_count, 1)
                self.assertEqual(empty_cache.selection.states['object_review']['state'], 'pending')
                self.assertEqual(second['objectProfiles'], first['objectProfiles'])
                self.assertTrue(second['objectReconciliationEnabled'])
                self.assertEqual(cache.selection.states['object_review']['state'], 'reused')

    def test_bounded_code_revision_invalidates_legacy_vision_and_voice(self):
        """Stable source directories cannot promote old worker results under a new revision."""
        from voice_studio.graph.media_graph_ingest import analyze_windows
        module = 'voice_studio.graph.media_graph_ingest.'
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            source = root / 'video.mp4'
            source.write_bytes(b'video-revision-fixture')
            metadata = {'kind': 'video', 'durationMs': 1000, 'streams': []}
            stack.enter_context(patch(module + 'visual_clip'))
            face = stack.enter_context(patch(module + 'collect_faces', return_value={'stages': {'light-asd': 'ok'}}))
            scene = stack.enter_context(patch(module + 'review_scene', return_value={'status': 'ok', 'frames': [], 'stages': {}}))
            stack.enter_context(patch(module + 'build_moodbook', return_value=[]))
            stack.enter_context(patch(module + 'scan_shots', return_value=[]))
            for revision in ('first', 'first', 'second'):
                analyze_windows(root, source, metadata, 'en', True, True, None,
                                revolver=AnalysisRevolver(root / 'cache'), cache_revision=revision, speakers=False)
            self.assertEqual((face.call_count, scene.call_count), (2, 2))

        class Registry:
            """Persist the test identity needed to validate a reused public voice checkpoint."""
            def __init__(self, source_id):
                self.source_id = source_id
            @classmethod
            def load(cls, path, source_id):
                return cls(source_id)
            def index(self):
                return [{'id': 'test-voice'}]
            def save(self, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            source = root / 'audio.wav'
            source.write_bytes(b'voice-revision-fixture')
            metadata = {'kind': 'audio', 'durationMs': 1000, 'streams': [{'kind': 'audio', 'index': 0, 'startMs': 0}]}
            stack.enter_context(patch(module + 'extract_speech', side_effect=lambda source, work, start, end, audio: (work / 'wav', 0, 1000)))
            stack.enter_context(patch(module + 'transcribe_wav', side_effect=lambda *args, **kwargs: {
                'segments': [{'startMs': 0, 'endMs': 200, 'text': 'stable', 'speaker': 1}],
                'speechTurns': [{'startMs': 0, 'endMs': 200, 'speaker': 1}]}))
            stack.enter_context(patch(module + 'analyze_audio_scene', return_value={'audioLayers': [], 'soundAnalysisStatus': 'done'}))
            stack.enter_context(patch('voice_studio.speech.voice_profiles.VoiceProfileRegistry', Registry))
            stack.enter_context(patch('voice_studio.speech.speaker_pipeline.project_voice_index'))
            stack.enter_context(patch('voice_studio.speech.speaker_pipeline.attach_dialogue_audit'))
            def attach(layer, *args, **kwargs):
                layer['segments'][0]['voiceProfileId'] = 'test-voice'
                return {'profiles': [{'id': 'test-voice'}]}
            voice = stack.enter_context(patch('voice_studio.speech.speaker_pipeline.attach_voice_profiles', side_effect=attach))
            for revision in ('first', 'first', 'second'):
                analyze_windows(root, source, metadata, 'en', False, False, None,
                                revolver=AnalysisRevolver(root / 'cache'), cache_revision=revision)
            self.assertEqual(voice.call_count, 2)
