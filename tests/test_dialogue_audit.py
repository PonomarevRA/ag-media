"""Contract tests for deterministic candidates and isolated bounded local retries."""

import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave

from voice_studio.speech.dialogue_audit import AuditOptions, audit_dialogue, clean_windows
from voice_studio.speech.dialogue_audit_runner import (AuditCancelled, decode_local, projected_suggestions,
                                                     run_audit, snippet_plan)


def row(start, end, speaker=None, text=None):
    """Build one original-clock fact for a synthetic contract fixture."""
    result = {'startMs': start, 'endMs': end, 'speaker': speaker}
    if text is not None:
        result['text'] = text
    return result


def audit(cues, turns, duration=10000, **kwargs):
    """Supply explicit source identity for pure detector tests."""
    return audit_dialogue({'durationMs': duration, 'segments': cues}, turns,
                          source={'sha256': 'a' * 64, 'durationMs': duration}, **kwargs)


class DialogueAuditTests(TestCase):
    """Check candidates independently of how Whisper happens to phrase a recording."""

    def test_transition_is_not_simultaneous_and_inputs_are_immutable(self):
        cues = [row(0, 10000, None, 'a full sentence and then another complete reply here')]
        turns = [row(100, 4000, 1), row(6000, 9800, 2)]
        before = copy.deepcopy((cues, turns))
        result = audit(cues, turns)
        candidate = result['candidates'][0]
        self.assertIn('speaker-transition-inside-cue', candidate['reasons'])
        self.assertNotIn('simultaneous-voices', candidate['reasons'])
        self.assertEqual(candidate['evidence']['simultaneousMs'], 0)
        self.assertEqual(candidate['evidence']['transitionMs'], [5000])
        self.assertEqual((cues, turns), before)
        self.assertEqual(audit(cues, turns), result)

    def test_real_overlap_is_removed_from_clean_snippets(self):
        turns = [row(0, 6000, 1), row(4000, 10000, 2)]
        result = audit([row(0, 10000, None, 'many words are in this longer sentence and reply')], turns)
        self.assertEqual(result['candidates'][0]['evidence']['simultaneousMs'], 2000)
        self.assertIn('simultaneous-voices', result['candidates'][0]['reasons'])
        windows = result['candidates'][0]['reviewWindows']
        self.assertEqual([(w['startMs'], w['endMs']) for w in windows], [(0, 4000), (6000, 10000)])

    def test_sparse_cue_is_a_missing_speech_candidate_without_claiming_error(self):
        result = audit([row(0, 10000, 1, 'question?')], [row(0, 10000, 1)])
        self.assertEqual(result['candidates'][0]['priority'], 'high')
        self.assertIn('sparse-text-over-voiced-audio', result['candidates'][0]['reasons'])
        self.assertIn('not confirmed', result['limits'][0])
        self.assertEqual(result['repairs']['suggestions'], [])

    def test_unknown_speaker_keeps_unresolved_identity_and_duplicates_do_not_inflate(self):
        unknown = row(0, 10000)
        result = audit([row(0, 10000, None, 'one two three four five six seven eight nine ten')],
                       [unknown, unknown])
        self.assertEqual(result['summary']['duplicateTurnsRemoved'], 1)
        self.assertEqual(result['candidates'][0]['evidence']['speechMs'], 10000)
        self.assertIn('unknown-acoustic-label', result['candidates'][0]['reasons'])
        self.assertTrue(all(w['identity'] == 'unresolved' for w in result['candidates'][0]['reviewWindows']))

    def test_silence_and_tiny_cluster_noise_do_not_make_mixed_voice_candidate(self):
        self.assertEqual(audit([row(0, 10000, None, 'question')], [])['candidates'], [])
        result = audit([row(0, 10000, 1, 'one two three four five six seven eight nine ten')],
                       [row(0, 10000, 1), row(5000, 5100, 2)])
        self.assertEqual(result['candidates'], [])

    def test_voiced_gaps_are_found_once_despite_duplicate_turns(self):
        result = audit([row(0, 2000, 1, 'some text')], [row(0, 8000, 1), row(0, 8000, 1)])
        missing = [c for c in result['candidates'] if 'voiced-audio-without-text' in c['reasons']]
        self.assertEqual([(c['startMs'], c['endMs']) for c in missing], [(2000, 8000)])

    def test_invalid_bounds_options_and_source_binding_fail_closed(self):
        for bad in [row(-1, 300), row(0, 10001), row(500, 100)]:
            with self.assertRaises(ValueError):
                audit([], [bad])
        with self.assertRaises(ValueError):
            AuditOptions(max_snippet_ms=500)
        with self.assertRaises(ValueError):
            audit_dialogue({'durationMs': 10000, 'sourceSha256': 'b' * 64, 'segments': []}, [],
                           source={'sha256': 'a' * 64, 'durationMs': 10000})
        with self.assertRaises(ValueError):
            audit_dialogue({'durationMs': 10000, 'segments': []}, [],
                           source={'sha256': 'a' * 64, 'durationMs': 5000})

    def test_visual_facts_are_provenance_never_identity(self):
        result = audit([row(0, 10000, None, 'question?')], [row(0, 10000, 1)],
                       visual_facts={'samples': [{'timeMs': 500, 'active': True, 'panelId': 'face-a',
                                                  'evidenceStrength': 'explicit'}]})
        visual = result['candidates'][0]['evidence']['visual']
        self.assertEqual(visual['authority'], 'explicit-video-first')
        self.assertEqual(visual['identity'], 'unresolved')
        self.assertNotIn('speakerNumber', result['candidates'][0])

    def test_plan_is_bounded_nonoverlapping_and_suggestions_are_not_alignment(self):
        result = audit([row(0, 10000, None, '?')], [row(0, 10000, 1)])
        windows = snippet_plan(result, 1, 7000)
        self.assertEqual(len(windows), 1)
        self.assertLessEqual(sum(w['endMs'] - w['startMs'] for w in windows), 7000)
        raw = {'transcription': [{'offsets': {'from': -1, 'to': 99999}, 'text': 'hello',
                                  'tokens': [{'t_dtw': -1}]}]}
        suggestion = projected_suggestions(raw, windows[0])[0]
        self.assertFalse(suggestion['forceAligned'])
        self.assertEqual(suggestion['timingBasis'], 'asr-segment-offsets')
        self.assertEqual(suggestion['startMs'], windows[0]['startMs'])
        self.assertEqual(suggestion['endMs'], windows[0]['endMs'])
        self.assertEqual(raw['transcription'][0]['tokens'][0]['t_dtw'], -1)


class DialogueAuditRunnerTests(TestCase):
    """Verify cache identities, budget, silence and prior-output preservation."""

    def fixture(self, folder, voiced=True):
        """Write a small PCM/report fixture with several seconds of supported audio."""
        root = Path(folder)
        source, report, turns = root / 'source.wav', root / 'report.json', root / 'turns.json'
        with wave.open(str(source), 'wb') as audio:
            audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            sample = b'\xe8\x03' if voiced else b'\x00\x00'
            audio.writeframes(sample * 16000 * 8)
        report.write_text(json.dumps({'durationMs': 8000, 'language': 'ru',
                                      'segments': [row(0, 8000, 1, 'question?')]}))
        turns.write_text(json.dumps({'turns': [row(0, 8000, 1)]}))
        return report, turns, source, root / 'out'

    def test_complete_repeat_hits_cache_and_input_changes_invalidate(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            before = [path.read_bytes() for path in args[:3]]
            first = run_audit(*args)
            second = run_audit(*args)
            self.assertFalse(first['cacheHit'])
            self.assertTrue(second['cacheHit'])
            self.assertEqual(first['runKey'], second['runKey'])
            self.assertEqual([path.read_bytes() for path in args[:3]], before)
            changed = run_audit(*args, options=AuditOptions(edge_drift_ms=2000))
            self.assertNotEqual(changed['runKey'], first['runKey'])
            data = json.loads(args[0].read_text())
            data['segments'][0]['text'] = 'changed'
            args[0].write_text(json.dumps(data))
            report_changed = run_audit(*args)
            self.assertNotEqual(report_changed['runKey'], changed['runKey'])
            with args[2].open('ab') as source:
                source.write(b'extra')
            self.assertNotEqual(run_audit(*args)['runKey'], report_changed['runKey'])

    def test_malformed_cache_is_rebuilt_and_input_collision_is_rejected(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            first = run_audit(*args)
            cache = args[3] / 'cache' / (first['runKey'] + '.json')
            cache.write_text('[]')
            self.assertFalse(run_audit(*args)['cacheHit'])
            target = args[3] / 'audit.json'
            target.write_bytes(args[0].read_bytes())
            with self.assertRaisesRegex(ValueError, 'overwrite an input'):
                run_audit(target, args[1], args[2], args[3])

    def test_source_duration_mismatch_and_output_input_collision_rejected(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            with self.assertRaises(ValueError):
                run_audit(*args[:3], args[0].parent, timeout_seconds=-1)
            data = json.loads(args[0].read_text())
            data['durationMs'] += 1000
            args[0].write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'duration mismatch'):
                run_audit(*args)

    def test_silence_is_pruned_before_decode(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder, voiced=False)
            with patch('voice_studio.speech.recognition.ready', return_value=True), patch(
                    'voice_studio.speech.dialogue_audit_runner.file_hash', return_value='a' * 64), patch(
                    'voice_studio.speech.dialogue_audit_runner.decode_local') as decode:
                result = run_audit(*args, rerun=True, max_windows=1)
            decode.assert_not_called()
            self.assertEqual(result['repairs']['windows'][0]['status'], 'silence-pruned')
            self.assertEqual(result['repairs']['budget']['decodedWindows'], 0)

    def test_timeout_cancel_and_decode_failure_preserve_last_success(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            run_audit(*args)
            successful = (args[3] / 'audit.json').read_bytes()
            for failure, expected in [(AuditCancelled('stop'), 'cancelled'),
                                      (TimeoutError('timeout'), 'timeout'), (ValueError('bad decode'), 'failed')]:
                with patch('voice_studio.speech.recognition.ready', return_value=True), patch(
                        'voice_studio.speech.dialogue_audit_runner.file_hash', return_value='a' * 64), patch(
                        'voice_studio.speech.dialogue_audit_runner.decode_local', side_effect=failure):
                    result = run_audit(*args, rerun=True, max_windows=1)
                self.assertEqual(result['status'], expected)
                self.assertEqual((args[3] / 'audit.json').read_bytes(), successful)
                self.assertTrue(Path(result['artifactPath']).is_file())
            with self.assertRaises(AuditCancelled):
                run_audit(*args, cancelled=lambda: True)
            self.assertEqual((args[3] / 'audit.json').read_bytes(), successful)

    def test_effective_cpu_budget_is_pinned_to_decode_and_cache_identity(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            native = {'transcription': []}
            with patch('voice_studio.speech.recognition.ready', return_value=True), patch(
                    'voice_studio.speech.dialogue_audit_runner.file_hash', return_value='a' * 64), patch(
                    'voice_studio.speech.dialogue_audit_runner.decode_local', return_value=native) as decode:
                with patch.dict(os.environ, {'VOICE_RESOURCE_PROFILE': 'throughput', 'VOICE_CPU_BUDGET': '1'}):
                    limited = run_audit(*args, rerun=True, max_windows=1)
                self.assertEqual(decode.call_args.kwargs['threads'], 1)
                self.assertEqual(limited['runOptions']['effectiveThreads'], 1)
                self.assertEqual(limited['runOptions']['decodeOptions'][-2:], ['-t', '1'])
                with patch('voice_studio.speech.dialogue_audit_runner.stage_threads', return_value=4):
                    normal = run_audit(*args, rerun=True, max_windows=1)
                self.assertEqual(decode.call_args.kwargs['threads'], 4)
                self.assertEqual(normal['runOptions']['effectiveThreads'], 4)
                self.assertNotEqual(limited['runKey'], normal['runKey'])

    def test_direct_local_decode_respects_throughput_cpu_ceiling(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / 'argument-whisper'
            executable.write_text(
                f'#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n'
                'Path(sys.argv[sys.argv.index("-of")+1]+".json").write_text(json.dumps({"argv":sys.argv[1:]}))\n')
            executable.chmod(0o755)
            with patch('voice_studio.speech.recognition.CLI', executable), patch.dict(
                    os.environ, {'VOICE_RESOURCE_PROFILE': 'throughput', 'VOICE_CPU_BUDGET': '1'}):
                result = decode_local(root / 'in.wav', root / 'out', 'ru', root / 'model', lambda: None)
            self.assertEqual(result['argv'][result['argv'].index('-t') + 1], '1')

    def test_successful_retry_has_separate_raw_asr_and_original_clock(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            native = {'transcription': [{'text': 'answer', 'offsets': {'from': 120, 'to': 1000},
                                          'tokens': [{'t_dtw': -1}]}]}
            with patch('voice_studio.speech.recognition.ready', return_value=True), patch(
                    'voice_studio.speech.dialogue_audit_runner.file_hash', return_value='a' * 64), patch(
                    'voice_studio.speech.dialogue_audit_runner.decode_local', return_value=native):
                result = run_audit(*args, rerun=True, max_windows=1, max_audio_ms=7000)
            repairs = result['repairs']
            self.assertEqual(repairs['windows'][0]['rawASR'], native)
            self.assertEqual(repairs['suggestions'][0]['startMs'], 120)
            self.assertLessEqual(repairs['budget']['decodedAudioMs'], 7000)
            self.assertEqual(json.loads(args[0].read_text())['segments'][0]['text'], 'question?')


    def test_original_source_metadata_binds_mov_and_pcm_identities(self):
        with TemporaryDirectory() as folder:
            args = self.fixture(folder)
            data = json.loads(args[0].read_text())
            data['sourceDigest'] = 'b' * 64
            args[0].write_text(json.dumps(data))
            metadata = {'sha256': 'b' * 64, 'path': '/local/original.mov', 'durationMs': 8000}
            result = run_audit(*args, source_metadata=metadata)
            self.assertEqual(result['source']['sha256'], 'b' * 64)
            self.assertNotEqual(result['source']['pcmSha256'], 'b' * 64)
            self.assertEqual(result['source']['pcmPath'], str(args[2].resolve()))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                run_audit(*args, source_metadata={**metadata, 'sha256': 'c' * 64})
            with self.assertRaisesRegex(ValueError, 'clockOffsetMs'):
                run_audit(*args, source_metadata={**metadata, 'clockOffsetMs': 1000})

    def test_input_candidate_and_local_turn_budgets_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'input row budget'):
            audit([row(0, 10000, 1, '?')], [row(0, 10000, 1)], options=AuditOptions(max_input_rows=1))
        with self.assertRaisesRegex(ValueError, 'per-cue turn budget'):
            audit([row(0, 10000, 1, '?')], [row(0, 10000, 1), row(0, 10000, 2)],
                  options=AuditOptions(max_turns_per_cue=1))
        with self.assertRaisesRegex(ValueError, 'candidate budget'):
            audit([row(0, 5000, 1, '?'), row(5000, 10000, 1, '?')], [row(0, 10000, 1)],
                  options=AuditOptions(max_candidates=1))

    def test_real_subprocess_cancel_kills_child_and_releases_lock(self):
        """Cancellation must terminate an actual process, not only a mocked decoder."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / 'sleep-whisper'
            executable.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(30)\n')
            executable.chmod(0o755)
            calls = {'count': 0}

            def cancel_after_spawn():
                """Allow process creation, then interrupt the poll loop."""
                calls['count'] += 1
                if calls['count'] >= 2:
                    raise AuditCancelled('test')

            with patch('voice_studio.speech.recognition.CLI', executable):
                with self.assertRaises(AuditCancelled):
                    decode_local(root / 'in.wav', root / 'out', 'ru', root / 'model', cancel_after_spawn)
            from voice_studio.speech.recognition import LOCK
            self.assertTrue(LOCK.acquire(blocking=False))
            LOCK.release()
