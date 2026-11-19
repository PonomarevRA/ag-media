"""Exercise service validation, clock preservation, deadlines and cache writers."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from time import sleep
from unittest import TestCase, skipUnless
from unittest.mock import patch
import wave

from voice_studio.app.dialogue_audit_service import audit_media, cache_writer, prepare_pcm
from voice_studio.speech.dialogue_audit_runner import AuditCancelled
from voice_studio.core.storage_cache import cache_lock, fcntl as storage_fcntl

MODULE = 'voice_studio.app.dialogue_audit_service.'


def write_pcm(path, duration=1000, sample=b'\xe8\x03'):
    """Generate tiny mono16k PCM without native inference or external media."""
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        audio.writeframes(sample * (duration * 16))


class DialogueAuditServiceTests(TestCase):
    """Keep expensive decoding downstream of all source/report/budget validation."""

    def fixture(self, folder):
        """Create a source-bound report and controlled cache location."""
        root = Path(folder).resolve()
        source, report, turns = root / 'source.wav', root / 'report.json', root / 'turns.json'
        write_pcm(source)
        digest = sha256(source.read_bytes()).hexdigest()
        report.write_text(json.dumps({'durationMs': 1000, 'language': 'ru', 'sourceDigest': digest,
                                      'segments': [{'startMs': 0, 'endMs': 1000, 'text': 'hello there'}],
                                      'speechTurns': [{'startMs': 0, 'endMs': 1000, 'speaker': 1}]}))
        turns.write_text(json.dumps({'turns': [{'startMs': 0, 'endMs': 1000, 'speaker': 1}]}))
        media = {'durationMs': 1000, 'streams': [{'kind': 'audio', 'index': 0, 'startMs': 0}]}
        return root, source, report, turns, media

    def execute(self, fixture, **kwargs):
        """Use real hashing/runner and synthetic PCM, with isolated native admission."""
        root, source, report, turns, media = fixture
        def decode(original, info, output, **options):
            """Produce the same valid PCM a native decoder would publish."""
            options['check']()
            write_pcm(output, info['durationMs'])
        with patch(MODULE + 'workspace_temp', return_value=root / 'cache'), patch(
                MODULE + 'ensure_temporary_root'), patch(MODULE + 'probe_source', return_value=media), patch(
                MODULE + 'run_bounded_batch_stage', side_effect=lambda operation, **options: operation()), patch(
                MODULE + 'prepare_pcm', side_effect=decode) as prepare:
            result = audit_media(root, source, report, **kwargs)
        return result, prepare.call_count

    def test_invalid_budget_does_not_read_hash_probe_or_decode(self):
        invalid = [{'max_windows': -1}, {'max_windows': True}, {'max_audio_ms': 600001},
                   {'timeout_seconds': float('nan')}, {'timeout_seconds': 0}, {'rerun': 'true'}]
        with TemporaryDirectory() as folder:
            root, source, report, _, _ = self.fixture(folder)
            for options in invalid:
                with self.subTest(options=options), patch(MODULE + 'read_report') as read, patch(
                        MODULE + 'identify_source') as identify, patch(MODULE + 'prepare_pcm') as prepare:
                    with self.assertRaises(ValueError):
                        audit_media(root, source, report, **options)
                    read.assert_not_called()
                    identify.assert_not_called()
                    prepare.assert_not_called()

    def test_matching_native_eof_copy_reaches_runner_and_cache_without_changing_inputs(self):
        """Service preflight and runner apply the same policy to captured native evidence."""
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            report_path = fixture[2]
            report = json.loads(report_path.read_text())
            report['speechTurns'][0]['endMs'] = 1024
            report_path.write_text(json.dumps(report))
            captured = report_path.read_bytes()
            first, decodes = self.execute(fixture)
            second, repeats = self.execute(fixture)
            self.assertEqual(first['status'], 'complete')
            self.assertFalse(first['cacheHit'])
            self.assertTrue(second['cacheHit'])
            self.assertEqual((decodes, repeats), (1, 0))
            self.assertEqual(report_path.read_bytes(), captured)
            self.assertEqual(first['nativeTurnBounds']['adjustmentCount'], 1)
            self.assertNotEqual(first['nativeTurnBounds']['rawTurnsHash'], first['inputHashes']['turns'])
            report['speechTurns'][0]['endMs'] = 1025
            report_path.write_text(json.dumps(report))
            changed, _ = self.execute(fixture)
            self.assertFalse(changed['cacheHit'])
            self.assertNotEqual(changed['runKey'], first['runKey'])

    def test_repeated_cue_time_budget_prevents_unbounded_derived_windows(self):
        with TemporaryDirectory() as folder:
            root, source, report, _, _ = self.fixture(folder)
            value = json.loads(report.read_text())
            value['durationMs'] = 7200000
            value['segments'] = [{'startMs': 0, 'endMs': 7200000, 'text': '?'}] * 5
            value['speechTurns'] = [{'startMs': 0, 'endMs': 7200000, 'speaker': 1}]
            report.write_text(json.dumps(value))
            with patch(MODULE + 'identify_source') as identify, patch(MODULE + 'prepare_pcm') as prepare:
                with self.assertRaisesRegex(ValueError, 'бюджет перекрытий'):
                    audit_media(root, source, report)
                identify.assert_not_called()
                prepare.assert_not_called()

    @skipUnless(storage_fcntl is not None, 'Cross-process cache cleanup lease requires POSIX flock')
    def test_cleanup_exclusive_lock_observes_service_deadline_before_pcm(self):
        with TemporaryDirectory() as folder:
            root, source, report, _, media = self.fixture(folder)
            group = root / 'cache' / 'analysis-checkpoints'
            with cache_lock(group, exclusive=True) as acquired:
                self.assertTrue(acquired)
                with patch(MODULE + 'workspace_temp', return_value=root / 'cache'), patch(
                        MODULE + 'ensure_temporary_root'), patch(MODULE + 'probe_source', return_value=media), patch(
                        MODULE + 'run_bounded_batch_stage', side_effect=lambda operation, **options: operation()), patch(
                        MODULE + 'prepare_pcm') as prepare:
                    with self.assertRaises(TimeoutError):
                        audit_media(root, source, report, timeout_seconds=.05)
                    prepare.assert_not_called()

    def test_invalid_turns_report_or_conflicting_hash_fails_before_source_work(self):
        with TemporaryDirectory() as folder:
            root, source, report, turns, _ = self.fixture(folder)
            original = json.loads(report.read_text())
            for bad in ({'turns': 'not a list'}, {'turns': [{'startMs': -1, 'endMs': 100, 'speaker': 1}]},
                        {'turns': [{'startMs': 0, 'endMs': 1001, 'speaker': 1}]}):
                turns.write_text(json.dumps(bad))
                with patch(MODULE + 'identify_source') as identify, patch(MODULE + 'prepare_pcm') as prepare:
                    with self.assertRaises(ValueError):
                        audit_media(root, source, report, turns_path=turns)
                    identify.assert_not_called()
                    prepare.assert_not_called()
            for change in ({'durationMs': True}, {'sourceSha256': 'b' * 64},
                           {'segments': [{'startMs': 0, 'endMs': 1000, 'text': None}]}):
                report.write_text(json.dumps({**original, **change}))
                with patch(MODULE + 'identify_source') as identify, patch(MODULE + 'prepare_pcm') as prepare:
                    with self.assertRaises(ValueError):
                        audit_media(root, source, report)
                    identify.assert_not_called()
                    prepare.assert_not_called()

    def test_external_turn_hash_and_nonzero_clock_are_rejected_before_source_work(self):
        with TemporaryDirectory() as folder:
            root, source, report, turns, _ = self.fixture(folder)
            original = json.loads(turns.read_text())
            for change in ({'sourceSha256': 'b' * 64}, {'clockOffsetMs': 500},
                           {'source': {'clockOffsetMs': 500}}):
                turns.write_text(json.dumps({**original, **change}))
                with patch(MODULE + 'identify_source') as identify, patch(MODULE + 'prepare_pcm') as prepare:
                    with self.assertRaises(ValueError):
                        audit_media(root, source, report, turns_path=turns)
                    identify.assert_not_called()
                    prepare.assert_not_called()

    def test_historical_47ms_rounding_pads_pcm_to_report_clock(self):
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            value = json.loads(fixture[2].read_text())
            value['durationMs'] = 1047
            value['segments'][0]['endMs'] = 1047
            value['speechTurns'][0]['endMs'] = 1047
            fixture[2].write_text(json.dumps(value))
            result, count = self.execute(fixture)
            self.assertEqual(count, 1)
            self.assertEqual(result['source']['durationMs'], 1047)
            with wave.open(result['source']['pcmPath'], 'rb') as pcm:
                self.assertEqual(pcm.getnframes(), 1047 * 16)

    def test_source_mismatch_wrong_duration_or_audio_offset_avoids_pcm(self):
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            root, source, report, _, media = fixture
            original = json.loads(report.read_text())
            report.write_text(json.dumps({**original, 'sourceDigest': 'b' * 64}))
            with patch(MODULE + 'probe_source') as probe, patch(MODULE + 'prepare_pcm') as prepare:
                with self.assertRaisesRegex(ValueError, 'другому исходнику'):
                    audit_media(root, source, report)
                probe.assert_not_called()
                prepare.assert_not_called()
            report.write_text(json.dumps(original))
            for changed in ({**media, 'durationMs': 1200},
                            {**media, 'streams': [{'kind': 'audio', 'index': 0, 'startMs': -1}]}):
                with patch(MODULE + 'probe_source', return_value=changed), patch(MODULE + 'prepare_pcm') as prepare:
                    with self.assertRaises(ValueError):
                        audit_media(root, source, report)
                    prepare.assert_not_called()

    def test_empty_native_turns_are_unavailable_and_do_not_decode(self):
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            root, source, report, _, _ = fixture
            value = json.loads(report.read_text())
            value['speechTurns'] = []
            report.write_text(json.dumps(value))
            result, count = self.execute(fixture)
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(count, 0)

    def test_repeat_reuses_validated_pcm_and_complete_runner_result(self):
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            first, count = self.execute(fixture)
            second, again = self.execute(fixture)
            self.assertEqual((count, again), (1, 0))
            self.assertFalse(first['cacheHit'])
            self.assertTrue(second['cacheHit'])
            self.assertEqual(first['runKey'], second['runKey'])
            self.assertEqual(first['source']['sha256'], json.loads(fixture[2].read_text())['sourceDigest'])
            self.assertIn('pcmSha256', first['source'])

    def test_external_turns_are_captured_and_content_changes_get_distinct_invocations(self):
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            first, _ = self.execute(fixture, turns_path=fixture[3])
            fixture[3].write_text(json.dumps({'turns': [{'startMs': 100, 'endMs': 900, 'speaker': 2}]}))
            second, count = self.execute(fixture, turns_path=fixture[3])
            self.assertEqual(count, 0)
            self.assertNotEqual(first['runKey'], second['runKey'])
            self.assertNotEqual(Path(first['artifactPath']).parent, Path(second['artifactPath']).parent)

    def test_turn_file_edits_during_pcm_do_not_change_captured_input(self):
        with TemporaryDirectory() as folder:
            root, source, report, turns, media = self.fixture(folder)
            captured = json.loads(turns.read_text())
            def decode(original, info, output, **kwargs):
                """Alter the external document after service validation captured it."""
                write_pcm(output)
                turns.write_text(json.dumps({'turns': [{'startMs': 100, 'endMs': 900, 'speaker': 9}]}))
            with patch(MODULE + 'workspace_temp', return_value=root / 'cache'), patch(
                    MODULE + 'ensure_temporary_root'), patch(MODULE + 'probe_source', return_value=media), patch(
                    MODULE + 'run_bounded_batch_stage', side_effect=lambda operation, **options: operation()), patch(
                    MODULE + 'prepare_pcm', side_effect=decode):
                result = audit_media(root, source, report, turns_path=turns)
            invocation = Path(result['artifactPath']).parent.parent
            self.assertEqual(json.loads((invocation / 'input-turns.json').read_text()), captured)

    def test_corrupt_pcm_and_receipt_are_rebuilt(self):
        with TemporaryDirectory() as folder:
            fixture = self.fixture(folder)
            result, _ = self.execute(fixture)
            pcm = Path(result['source']['pcmPath'])
            original = pcm.read_bytes()
            pcm.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            _, count = self.execute(fixture)
            self.assertEqual(count, 1)
            pcm.write_bytes(b'RIFF')
            _, count = self.execute(fixture)
            self.assertEqual(count, 1)
            receipt = pcm.with_suffix('.json')
            data = json.loads(receipt.read_text())
            data['durationMs'] = 900
            receipt.write_text(json.dumps(data))
            _, count = self.execute(fixture)
            self.assertEqual(count, 1)

    def test_source_changed_during_decode_stops_before_runner(self):
        with TemporaryDirectory() as folder:
            root, source, report, _, media = self.fixture(folder)
            def changed(original, info, output, **kwargs):
                """Simulate external replacement while native PCM is being generated."""
                write_pcm(output)
                write_pcm(source, sample=b'\xff\x00')
            with patch(MODULE + 'workspace_temp', return_value=root / 'cache'), patch(
                    MODULE + 'ensure_temporary_root'), patch(MODULE + 'probe_source', return_value=media), patch(
                    MODULE + 'run_bounded_batch_stage', side_effect=lambda operation, **options: operation()), patch(
                    MODULE + 'prepare_pcm', side_effect=changed), patch(MODULE + 'run_audit') as runner:
                with self.assertRaises(ValueError):
                    audit_media(root, source, report)
                runner.assert_not_called()

    def test_remaining_deadline_reaches_probe_pcm_and_runner_callback(self):
        with TemporaryDirectory() as folder:
            root, source, report, _, media = self.fixture(folder)
            clock = [0.0]
            def probe(original, *, timeout):
                """Consume part of a shared caller budget."""
                self.assertEqual(timeout, 10)
                clock[0] += 2
                return media
            def decode(original, info, output, *, timeout, check):
                """Require the unused budget rather than a fresh fixed native timeout."""
                self.assertEqual(timeout, 8)
                write_pcm(output)
                clock[0] += 3
            def run(*args, **kwargs):
                """A stable cache budget and a shared absolute-deadline callback coexist."""
                self.assertEqual(kwargs['timeout_seconds'], 10)
                clock[0] = 11
                kwargs['cancelled']()
                self.fail('expired runner callback did not stop')
            with patch(MODULE + 'monotonic', side_effect=lambda: clock[0]), patch(
                    MODULE + 'workspace_temp', return_value=root / 'cache'), patch(
                    MODULE + 'ensure_temporary_root'), patch(MODULE + 'probe_source', side_effect=probe), patch(
                    MODULE + 'run_bounded_batch_stage', side_effect=lambda operation, **options: operation()), patch(
                    MODULE + 'prepare_pcm', side_effect=decode), patch(MODULE + 'run_audit', side_effect=run):
                with self.assertRaises(TimeoutError):
                    audit_media(root, source, report, timeout_seconds=10)

    def test_missing_rerun_model_stops_before_hashing_or_pcm(self):
        with TemporaryDirectory() as folder:
            root, source, report, _, _ = self.fixture(folder)
            with patch('voice_studio.speech.recognition.ready', return_value=False), patch(
                    MODULE + 'identify_source') as identify, patch(MODULE + 'prepare_pcm') as prepare:
                with self.assertRaisesRegex(ValueError, 'Whisper'):
                    audit_media(root, source, report, rerun=True)
                identify.assert_not_called()
                prepare.assert_not_called()

    def test_cache_writer_wait_observes_deadline_and_releases_owned_lock(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            calls = [0]
            def expired():
                """Expire while another cache owner holds the writer lease."""
                calls[0] += 1
                if calls[0] >= 2:
                    raise TimeoutError('expired')
            with cache_writer(directory, lambda: None):
                with self.assertRaises(TimeoutError):
                    with cache_writer(directory, expired):
                        self.fail('second cache writer entered')
            with cache_writer(directory, lambda: None):
                pass

    def test_failed_native_prepare_preserves_previous_pcm_and_cleans_staging(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'source.wav', root / 'existing.wav'
            write_pcm(source)
            write_pcm(output)
            before = output.read_bytes()
            media = {'durationMs': 1000, 'streams': [{'kind': 'audio', 'index': 0, 'startMs': 0}]}
            with patch(MODULE + 'ffmpeg', side_effect=TimeoutError('decode expired')):
                with self.assertRaises(TimeoutError):
                    prepare_pcm(source, media, output, timeout=.1)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual({path.name for path in root.iterdir()}, {'source.wav', 'existing.wav'})

    def test_cancelled_request_stops_before_json_or_source_work(self):
        with patch(MODULE + 'read_report') as read, patch(MODULE + 'prepare_pcm') as prepare:
            with self.assertRaises(AuditCancelled):
                audit_media('/workspace', '/source', '/report', cancelled=lambda: True)
            read.assert_not_called()
            prepare.assert_not_called()

    def test_concurrent_same_source_requests_decode_once_and_reuse_result(self):
        with TemporaryDirectory() as folder:
            root, source, report, _, media = self.fixture(folder)
            barrier, count_lock, count = Barrier(2), Lock(), [0]
            def probe(*args, **kwargs):
                """Admit both requests together to exercise the actual cache-writer lock."""
                barrier.wait(timeout=3)
                return deepcopy(media)
            def decode(original, info, output, **kwargs):
                """Model a short native phase while the second writer must wait."""
                with count_lock:
                    count[0] += 1
                sleep(.05)
                write_pcm(output)
            with patch(MODULE + 'workspace_temp', return_value=root / 'cache'), patch(
                    MODULE + 'ensure_temporary_root'), patch(MODULE + 'probe_source', side_effect=probe), patch(
                    MODULE + 'run_bounded_batch_stage', side_effect=lambda operation, **options: operation()), patch(
                    MODULE + 'prepare_pcm', side_effect=decode), ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(audit_media, root, source, report) for _ in range(2)]
                results = [future.result(timeout=5) for future in futures]
            self.assertEqual(count[0], 1)
            self.assertEqual(sorted(result['cacheHit'] for result in results), [False, True])
            self.assertEqual(results[0]['runKey'], results[1]['runKey'])

    @skipUnless(which('ffmpeg'), 'Requires installed local FFmpeg')
    def test_native_tiny_pcm_preserves_leading_audio_offset(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'source.wav', root / 'full-clock.wav'
            write_pcm(source, duration=200)
            media = {'durationMs': 700, 'streams': [{'kind': 'audio', 'index': 0, 'startMs': 500}]}
            prepare_pcm(source, media, output, timeout=5)
            with wave.open(str(output), 'rb') as audio:
                self.assertEqual(audio.getnframes(), 11200)
                self.assertEqual(audio.readframes(8000), bytes(16000))
                self.assertNotEqual(audio.readframes(3200), bytes(6400))
