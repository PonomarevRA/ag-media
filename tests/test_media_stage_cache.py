"""Exercise incremental reuse through the same file pipeline used by UI and MCP."""

from contextlib import ExitStack
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_stage_cache import IncrementalStages, usable_visual, usable_scene, usable_speech, replay_progress
from voice_studio.speech.media_transcript import transcribe_media, analyze_detail_layers

MODULE = 'voice_studio.speech.media_transcript.'


class StageCacheTests(TestCase):
    def test_changed_scene_and_audio_do_not_replay_other_layers(self):
        """Changing sampling/backend preserves speech/face calls and lazy PCM decoding."""
        with TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            stack.enter_context(patch('voice_studio.graph.media_stage_cache.analysis_fingerprint', return_value='revision1'))
            stack.enter_context(patch(MODULE + 'audio_offset', return_value=0))
            def decode(arguments, **kwargs):
                Path(arguments[-1]).write_bytes(b'pcm')
            ffmpeg = stack.enter_context(patch(MODULE + 'ffmpeg', side_effect=decode))
            speech = {'segments': [{'startMs': 0, 'endMs': 900, 'text': 'hello', 'speaker': 1}],
                      'text': 'hello', 'durationMs': 1000, 'language': 'en'}
            asr = stack.enter_context(patch(MODULE + 'transcribe_wav', side_effect=lambda *a, **k: deepcopy(speech)))
            sound = stack.enter_context(patch(MODULE + 'analyze_audio_scene', side_effect=lambda *a: {
                'audioLayers': [], 'soundAnalysisStatus': 'done', 'soundClassifier': a[-1]}))
            def faces(source, wav, offset, result, **kwargs):
                result['videoEvidence'] = {'stages': {'light-asd': 'ok', 'yunet': 'ok'}, 'phrases': {},
                                           'timeline': {'stepMs': 40, 'frames': []}}
            visual = stack.enter_context(patch(MODULE + 'analyze_visual_track', side_effect=faces))
            stack.enter_context(patch(MODULE + 'bind_faces'))
            stack.enter_context(patch(MODULE + 'build_moodbook', return_value=None))
            stack.enter_context(patch(MODULE + 'attach_person_motion', return_value=False))
            stack.enter_context(patch(MODULE + 'object_layers', return_value={}))
            stack.enter_context(patch(MODULE + 'reconcile_names', return_value=False))
            scene = stack.enter_context(patch(MODULE + 'review_scene', side_effect=lambda *a, **k: {
                'status': 'ok', 'frames': [], 'shotColors': [], 'sampling': k.get('object_sampling')}))
            def run(**kwargs):
                checkpoint = AnalysisRevolver(root / 'cache')
                report = transcribe_media(root, 'en', source, speakers=True, quality=True, visual=True,
                                          scene=True, revolver=checkpoint, **kwargs)
                return report, checkpoint.counts['layers']
            fresh_events, cached_events = [], []
            first, _ = run(emit=fresh_events.append)
            warm, stats = run(emit=cached_events.append)
            self.assertEqual(first, warm)
            self.assertTrue(all(row['reused'] for row in stats.values()))
            reason = "Нет подходящих голосовых образцов или коротких реплик для проверки"
            self.assertEqual(first['speakerReview'], {'status': 'skipped', 'reason': reason})
            for events in (fresh_events, cached_events):
                skipped, = [event for event in events
                            if event.get('stage') == 'speaker-review' and event['state'] == 'skipped']
                self.assertEqual(skipped['reason'], reason)
            changed, stats = run(object_sampling='basic')
            self.assertEqual(changed['segments'], first['segments'])
            self.assertEqual({key for key, row in stats.items() if not row['reused']}, {'scene'})
            _, stats = run(audio_classifier='zipformer')
            self.assertEqual({key for key, row in stats.items() if not row['reused']}, {'audio'})
            self.assertEqual((asr.call_count, sound.call_count, visual.call_count, scene.call_count), (1, 2, 1, 2))
            self.assertEqual(ffmpeg.call_count, 2)
            self.assertTrue(asr.call_args.kwargs['quality'])
            self.assertTrue(asr.call_args.kwargs['defer_audio'])

    def test_failed_downstream_resume_and_upstream_change(self):
        """An interrupted stage resumes after a successful speech checkpoint; changed cues fork it."""
        with TemporaryDirectory() as folder:
            path = Path(folder)
            calls = []
            def operation():
                calls.append('computed')
                return {'value': 1}
            cache = IncrementalStages(AnalysisRevolver(path), 'revision')
            one = cache.run('speech', {}, None, operation, lambda row: True)
            with self.assertRaisesRegex(RuntimeError, 'late'):
                cache.run('visual', {}, one, lambda: (_ for _ in ()).throw(RuntimeError('late')), lambda row: True)
            cache = IncrementalStages(AnalysisRevolver(path), 'revision')
            self.assertEqual(cache.run('speech', {}, None, operation, lambda row: True), one)
            cache.run('visual', {}, one, operation, lambda row: True)
            cache.run('visual', {}, {'value': 2}, operation, lambda row: True)
            self.assertEqual(len(calls), 3)
            # A different code/model revision must not share any old result.
            IncrementalStages(AnalysisRevolver(path), 'new-revision').run('speech', {}, None, operation, lambda row: True)
            self.assertEqual(len(calls), 4)

    def test_scene_environment_does_not_invalidate_speech(self):
        with TemporaryDirectory() as folder:
            cache = IncrementalStages(AnalysisRevolver(Path(folder)), 'revision')
            def run(name):
                return cache.run(name, {}, None, lambda: {'ok': True}, lambda row: True)
            run('speech'); run('scene')
            with patch.dict('os.environ', {'VOICE_SCENE_OCR': 'alternate'}):
                run('speech'); run('scene')
            self.assertTrue(cache.revolver.counts['layers']['speech']['reused'])
            self.assertFalse(cache.revolver.counts['layers']['scene']['reused'])

    def test_unavailable_and_corrupt_results_retry(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)
            cache = IncrementalStages(AnalysisRevolver(path), 'revision')
            def run(value):
                return cache.run('audio', {}, None, lambda: value, lambda row: row.get('status') == 'done')
            run({'status': 'failed'})
            self.assertEqual(list(path.rglob('complete.json.gz')), [])
            run({'status': 'done'})
            filename = next(path.rglob('complete.json.gz'))
            filename.write_bytes(gzip.compress(json.dumps({'result': [], 'digest': 'bad'}).encode()))
            self.assertEqual(run({'status': 'done', 'retry': True})['retry'], True)
            self.assertFalse(cache.revolver.counts['layers']['audio']['reused'])
            self.assertEqual(run({'status': 'done'})['retry'], True)
            filename.write_bytes(b'broken gzip')
            self.assertEqual(run({'status': 'done', 'retry': 2})['retry'], 2)

    def test_mosaic_is_republished_without_running_worker(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = IncrementalStages(AnalysisRevolver(root / 'cache'), 'revision')
            calls, urls = [], []
            def operation(publish):
                calls.append('worker')
                image = root / 'worker.png'
                image.write_bytes(b'png-fixture')
                publish(0, image)
                image.unlink()
                return {'status': 'ok', 'interfaceScrolls': [{
                    'startMs': 1000, 'endMs': 2000, 'width': 100, 'height': 200,
                    'mosaicText': [{'text': 'Search', 'box': [.2, .2, .3, .1]}],
                }]}
            def publish(index, image, markdown):
                self.assertEqual(image.read_bytes(), b'png-fixture')
                self.assertIn('| Search | `20, 40, 30, 20` |', markdown)
                urls.append(f'generation/{len(urls)}')
                return urls[-1]
            first = cache.scene({}, 'basic', root, operation, publish, None)
            second = cache.scene({}, 'basic', root, operation, publish, None)
            self.assertEqual(calls, ['worker'])
            self.assertNotEqual(first['interfaceScrolls'][0]['previewUrl'], second['interfaceScrolls'][0]['previewUrl'])

    def test_only_subframe_partial_terminal_tail_is_reusable(self):
        base = {'videoEvidence': {'stages': {'light-asd': 'partial', 'yunet': 'ok'},
                                 'activityCoverage': {'windows': [{'startMs': 0, 'endMs': 1000, 'status': 'ok'},
                                                                 {'startMs': 1000, 'endMs': 1003, 'status': 'no-video'}]}}}
        self.assertTrue(usable_visual(base))
        base['videoEvidence']['activityCoverage']['windows'][-1]['status'] = 'failed'
        self.assertFalse(usable_visual(base))
        base['videoEvidence']['activityCoverage']['windows'][-1]['status'] = 'no-video'
        base['videoEvidence']['activityCoverage']['windows'][-1]['endMs'] = 1200
        self.assertFalse(usable_visual(base))

    def test_incomplete_scene_is_retried(self):
        for incomplete in ({'status': 'ok', 'stages': {'nanodet': 'partial'}},
                           {'status': 'ok', 'stages': {'apple-vision': 'unavailable'}},
                           {'status': 'ok', 'interfaceStatus': 'partial'}):
            with self.subTest(incomplete=incomplete), TemporaryDirectory() as folder:
                cache = IncrementalStages(AnalysisRevolver(Path(folder)), 'revision')
                cache.run('scene', {}, None, lambda: incomplete, usable_scene)
                healthy = cache.run('scene', {}, None, lambda: {'status': 'ok'}, usable_scene)
                self.assertEqual(healthy, {'status': 'ok'})
                self.assertFalse(cache.revolver.counts['layers']['scene']['reused'])

    def test_scene_with_all_ocr_crops_recovered_can_be_reused(self):
        scene = {'status': 'ok', 'interfaceStatus': 'ok',
                 'stages': {'nanodet': 'ok', 'apple-vision': 'partial', 'tesseract': 'ok'}}
        self.assertTrue(usable_scene(scene))
        scene['stages']['tesseract'] = 'partial'
        self.assertFalse(usable_scene(scene))

    def test_recovery_completion_is_required_even_without_unresolved(self):
        for status in ('failed', 'partial', 'skipped', None):
            self.assertFalse(usable_speech({'segments': [], 'recognitionRecovery': {'status': status}}))
        self.assertTrue(usable_speech({'segments': [], 'recognitionRecovery': {
            'status': 'done', 'unresolved': [{'startMs': 100, 'endMs': 200}]}}))

    def test_replay_uses_ui_stage_names_and_never_claims_pcm_decode(self):
        with TemporaryDirectory() as folder:
            cache = IncrementalStages(AnalysisRevolver(Path(folder)), 'revision')
            events = []
            report = {'segments': [], 'durationMs': 1234, 'recognitionRecovery': {'status': 'done'}}
            for _ in range(2):
                cache.run('speech', {'speakers': True}, None, lambda: report, usable_speech, emit=events.append)
                cache.run('audio', {}, report, lambda: {'ok': True}, lambda value: True, emit=events.append)
                cache.run('visual', {}, report, lambda: {'ok': True}, lambda value: True, emit=events.append)
            stages = {event['stage'] for event in events if 'stage' in event}
            self.assertEqual(stages, {'whisper', 'diarization', 'dialogue-pauses', 'whisper-recovery',
                                     'audio-layers', 'light-asd', 'yunet', 'speaker-review'})
            self.assertIn({'durationMs': 1234}, events)
            self.assertTrue(all(event['reused'] for event in events if 'stage' in event))
            self.assertTrue(all(event['cacheHit'] for event in events if 'stage' in event))
            self.assertEqual({event['cacheLayer'] for event in events if event.get('cacheHit') is False},
                             {'speech', 'audio', 'visual'})

    def test_scene_key_uses_measured_inputs_without_person_worker_metrics(self):
        """Changed figure timings must not rerun OCR, while changed face geometry must."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = IncrementalStages(AnalysisRevolver(root), 'revision')
            view = {'timeline': {'frames': [[1, 2]]}, 'editorTimeline': {'shots': [{'startMs': 0}]},
                    'personLayer': {'workerMs': 123, 'workerCpuMs': 100}}
            calls = []
            def operation(publish):
                calls.append(1)
                return {'status': 'ok'}
            cache.scene(view, 'small-detail', root, operation, None, None)
            view['personLayer'] = {'workerMs': 999, 'workerCpuMs': 901}
            cache.scene(view, 'small-detail', root, operation, None, None)
            self.assertTrue(cache.revolver.counts['layers']['scene']['reused'])
            self.assertEqual(calls, [1])
            view['timeline']['frames'] = [[1, 3]]
            cache.scene(view, 'small-detail', root, operation, None, None)
            self.assertFalse(cache.revolver.counts['layers']['scene']['reused'])
            self.assertEqual(calls, [1, 1])

    def test_replay_retains_skipped_visual_stages(self):
        events = []
        report = {'videoEvidence': {'stages': {'light-asd': 'no-phrases', 'yunet': 'ok'}}}
        self.assertTrue(usable_visual(report))
        replay_progress('visual', {}, report, events.append, 1)
        self.assertEqual({event['stage']: event['state'] for event in events},
                         {'light-asd': 'skipped', 'yunet': 'done', 'speaker-review': 'skipped'})
        self.assertEqual(events[-1]['reason'], 'Причина пропуска отсутствует в сохранённом анализе')

    def test_ui_people_layer_reuses_and_invalidates_independently_of_scene(self):
        """Default UI layers must reuse in either execution order; changing gaze only reruns people."""
        for parallel in ('0', '1'):
            with self.subTest(parallel=parallel), TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                calls = {'people': 0, 'scene': 0}
                def people(source, work, evidence, **kwargs):
                    calls['people'] += 1
                    evidence['personLayer'] = {'frames': [], 'workerMs': calls['people'] * 100,
                                               'gazeUnavailable': False}
                def scene(*args, **kwargs):
                    calls['scene'] += 1
                    return {'status': 'ok', 'stages': {'nanodet': 'ok'}}
                stack.enter_context(patch(MODULE + 'attach_person_evidence', side_effect=people))
                stack.enter_context(patch(MODULE + 'review_scene', side_effect=scene))
                stack.enter_context(patch.dict('os.environ', {'VOICE_VIDEO_PARALLEL': parallel}))
                for gaze in (True, True, False):
                    cache = IncrementalStages(AnalysisRevolver(root / 'cache'), 'revision')
                    evidence = {'timeline': {'frames': [[1, 2]], 'stepMs': 40}, 'editorTimeline': {'shots': []}}
                    analyze_detail_layers(root / 'source', root, evidence, body=True, gaze=gaze, scene=True,
                                          object_sampling='small-detail', emit=lambda event: None, cache=cache)
                self.assertEqual(calls, {'people': 2, 'scene': 1})
