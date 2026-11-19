"""Independent regressions for observed cache, public-export and admission failures."""

from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from mcp.server.fastmcp import FastMCP

from voice_studio.app.media_pipeline import MAX_MEDIA_SECONDS, transcribe_uploaded_media
from voice_studio.core.media_resources import IN_BATCH
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_graph_ingest import SPEECH_WINDOW_MS, analyze_source
from voice_studio.mcp.dialogue_audit_mcp import register_dialogue_tools
from voice_studio.mcp.voice_mcp import StudioClient
from voice_studio.speech.media_transcript import transcribe_media
from voice_studio.speech.voice_profiles import VoiceProfileRegistry


class VoiceResumeCriticTests(TestCase):
    def test_existing_public_checkpoint_ahead_of_private_registry_rebuilds_consistent_tail(self):
        """Recover an interrupted old cache without orphaning a recurring voice's cue ID."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.wav'
            source.write_bytes(b'immutable source fixture')
            duration = 2 * SPEECH_WINDOW_MS + 3000
            metadata = {'kind': 'audio', 'hasAudio': True, 'durationMs': duration,
                        'streams': [{'kind': 'audio', 'index': 0, 'startMs': 0, 'durationMs': duration}]}

            def recognize(*args, **kwargs):
                """Supply clean native turns without invoking ASR or interpreting a model label."""
                return {'language': 'en', 'segments': [
                    {'startMs': 100, 'endMs': 1900, 'text': 'Raw text', 'speaker': 1}],
                    'speechTurns': [{'startMs': 100, 'endMs': 1900, 'speaker': 1}]}

            def attach(layer, wav, offset, source_id, *, registry, window_id, **kwargs):
                """Exercise the real registry with one first voice and one recurring second voice."""
                cue = layer['segments'][0]
                features = [1, 0] if window_id == 0 else [0, 1]
                decision = registry.observe(features, {
                    'key': str(window_id), 'sourceId': source_id, 'windowId': window_id,
                    'startMs': cue['startMs'], 'endMs': cue['endMs'], 'rawSpeaker': cue['speaker']})
                profile = next(row for row in registry.index() if row['id'] == decision['profileId'])
                cue.update(voiceProfileId=profile['id'], profileSpeakerNumber=profile['number'],
                           voiceProfileEvidence=decision)
                return {'profiles': registry.index()}

            original_save = VoiceProfileRegistry.save
            writes = 0

            def legacy_incomplete_write(registry, path):
                """Represent a legacy public window published before its private references reached disk."""
                nonlocal writes
                writes += 1
                if writes == 1:
                    original_save(registry, path)
                elif writes >= 3:
                    raise OSError('Controlled interruption before the third window committed')

            with (patch('voice_studio.graph.media_graph_ingest.probe_source', return_value=metadata),
                  patch('voice_studio.graph.media_graph_ingest.extract_speech',
                        side_effect=lambda source, work, start, end, stream: (work / 'unused.wav', 0, end - start)),
                  patch('voice_studio.graph.media_graph_ingest.transcribe_wav', side_effect=recognize),
                  patch('voice_studio.graph.media_graph_ingest.analyze_audio_scene',
                        return_value={'soundAnalysisStatus': 'too-short', 'audioLayers': []}),
                  patch('voice_studio.speech.voice_profiles.local_encoder_id', return_value='fixed-test-encoder'),
                  patch('voice_studio.speech.speaker_pipeline.attach_voice_profiles', side_effect=attach)):
                with patch.object(VoiceProfileRegistry, 'save', new=legacy_incomplete_write):
                    with self.assertRaises(OSError):
                        analyze_source(root, source, 'audio', 'en', visual=False, scene=False,
                                       revolver=AnalysisRevolver(root / 'cache'))
                report = analyze_source(root, source, 'audio', 'en', visual=False, scene=False,
                                        revolver=AnalysisRevolver(root / 'cache'))

            cues = report['segments']
            identities = {row['id'] for row in report['voiceProfiles']['profiles']}
            self.assertEqual([cue['text'] for cue in cues], ['Raw text'] * 3)
            self.assertTrue(all(cue['voiceProfileId'] in identities for cue in cues))
            self.assertEqual(cues[1]['voiceProfileId'], cues[2]['voiceProfileId'])
            self.assertNotEqual(cues[0]['voiceProfileId'], cues[1]['voiceProfileId'])


class AudioExportCriticTests(IsolatedAsyncioTestCase):
    async def test_audio_transcription_saves_analysis_consumable_by_public_profile_tool(self):
        """Connect audio-only transcription output to its advertised profile read workflow."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'audio.wav'
            source.write_bytes(b'controlled original audio')
            identity = 'a' * 32
            raw = {'language': 'en', 'durationMs': 2000, 'srt': '',
                   'segments': [{'startMs': 0, 'endMs': 1800, 'speaker': 9, 'text': 'Raw text',
                                 'voiceProfileId': identity, 'profileSpeakerNumber': 1,
                                 'voiceProfileEvidence': {'state': 'matched'}}],
                   'speechTurns': [{'startMs': 0, 'endMs': 1800, 'speaker': 9}],
                   'voiceProfiles': {'profiles': [{'id': identity, 'number': 1, 'references': []}]}}
            response = io.BytesIO(json.dumps(raw).encode())
            response.status = 200
            connection = Mock()
            connection.getresponse.return_value = response
            with patch('voice_studio.mcp.voice_mcp.http.client.HTTPConnection', return_value=connection):
                result = StudioClient(8766, root / 'out').transcript(str(source), 'en', speakers=True)
            analysis = Path(result['analysisPath'])
            saved = json.loads(analysis.read_text())
            self.assertEqual(saved['segments'], raw['segments'])
            self.assertEqual(saved['speechTurns'], raw['speechTurns'])
            server = FastMCP('critic-audio-profile-workflow')
            register_dialogue_tools(server, root)
            exported = await server.call_tool('speaker_profiles', {'analysis_path': str(analysis)})
            if isinstance(exported, tuple):
                public = exported[1]
            else:
                public = json.loads(next(block.text for block in exported if block.type == 'text'))
            self.assertEqual(public['profiles'][0]['id'], identity)
            self.assertEqual(public['cues'][0]['rawSpeaker'], 9)


class LongSettingsCriticTests(TestCase):
    def test_long_upload_preserves_explicit_quality_merge_body_and_gaze_options(self):
        """Crossing the short-duration boundary must not silently replace selected options."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'long.mov'
            source.write_bytes(b'fixed long source')
            metadata = {'durationMs': MAX_MEDIA_SECONDS * 1000 + 1, 'kind': 'video'}
            result = {'segments': []}
            with (patch('voice_studio.app.media_pipeline.probe_source', return_value=metadata),
                  patch('voice_studio.app.media_pipeline.workspace_temp', return_value=root / 'scratch'),
                  patch('voice_studio.app.media_pipeline.ensure_temporary_root'),
                  patch('voice_studio.app.media_pipeline.use_cache', side_effect=lambda *args: nullcontext()),
                  patch('voice_studio.app.media_pipeline.analyze_source', return_value=result) as analyze):
                transcribe_uploaded_media(root, 'en', source, quality=False, merge_similar=True,
                                           body=True, gaze=False, visual=True)
            for name, expected in {'quality': False, 'merge_similar': True, 'body': True, 'gaze': False}.items():
                self.assertIs(analyze.call_args.kwargs[name], expected)
            self.assertEqual(analyze.call_args.kwargs["revolver"].directory.parent.name, "bounded-v2")


class ShortStageAdmissionCriticTests(TestCase):
    def test_short_person_and_scene_workers_enter_shared_native_admission(self):
        """Short-file native stages must obey the throughput policy just like long-file stages."""
        entered = []

        class Cache:
            """Exercise cold production stages while avoiding durable fixture caches."""
            def run(self, name, settings, upstream, operation, valid, **kwargs):
                """Run the measured operation instead of replaying a synthetic result."""
                return operation()

            def scene(self, view, sampling, work, operation, publish_mosaic, emit):
                """Invoke the existing scene worker without mosaic persistence."""
                return operation(None)

        def people(*args, **kwargs):
            """Record admission at the actual figure/gaze worker boundary."""
            entered.append(('people', IN_BATCH.get()))
            return {'frames': []}

        def scene(*args, **kwargs):
            """Record admission at the actual scene worker boundary."""
            entered.append(('scene', IN_BATCH.get()))
            return {'status': 'ok', 'frames': []}

        def visual(source, wav, offset, result, **kwargs):
            """Supply a measured timeline so the actual person stage is eligible."""
            result['videoEvidence'] = {'timeline': {'frames': [[]], 'stepMs': 40}}

        base = {'segments': [{'startMs': 0, 'endMs': 1000, 'speaker': 1, 'text': 'Raw text'}],
                'durationMs': 1000, 'language': 'en'}

        with (TemporaryDirectory() as folder,
              patch.dict('os.environ', {'VOICE_RESOURCE_PROFILE': 'throughput', 'VOICE_VIDEO_PARALLEL': '0'}),
              patch('voice_studio.core.media_resources.batch_slot', side_effect=lambda **kwargs: nullcontext()),
              patch('voice_studio.graph.media_stage_cache.incremental_stages',
                    side_effect=lambda *args: nullcontext(Cache())),
              patch('voice_studio.speech.media_transcript.audio_offset', return_value=0),
              patch('voice_studio.speech.media_transcript.select_sound_classifier', return_value='auto'),
              patch('voice_studio.speech.media_transcript.ffmpeg',
                    side_effect=lambda args, **kwargs: Path(args[-1]).write_bytes(b'pcm')),
              patch('voice_studio.speech.media_transcript.transcribe_wav', return_value=deepcopy(base)),
              patch('voice_studio.speech.media_transcript.analyze_audio_scene',
                    return_value={'audioLayers': [], 'soundAnalysisStatus': 'done'}),
              patch('voice_studio.speech.media_transcript.analyze_visual_track', side_effect=visual),
              patch('voice_studio.speech.media_transcript.bind_faces'),
              patch('voice_studio.speech.media_transcript.build_moodbook', return_value=None),
              patch('voice_studio.speech.media_transcript.attach_person_motion', return_value=False),
              patch('voice_studio.speech.media_transcript.object_layers', return_value={}),
              patch('voice_studio.speech.media_transcript.reconcile_names', return_value=False),
              patch('voice_studio.speech.media_transcript.preview_people', side_effect=people),
              patch('voice_studio.speech.media_transcript.review_scene', side_effect=scene)):
            source = Path(folder) / 'source.mov'
            source.write_bytes(b'original source')
            transcribe_media(Path(folder), 'en', source, visual=True, body=True, gaze=True, scene=True)
        self.assertEqual(entered, [('people', True), ('scene', True)])

    def test_short_sound_worker_enters_shared_native_admission(self):
        """The short-file sound classifier must not bypass a held heavy-media allocation."""
        entered = []

        class Cache:
            """Run genuine cold stage operations without persisting a test cache."""
            def run(self, name, settings, upstream, operation, valid, **kwargs):
                """Exercise each production stage boundary exactly once."""
                return operation()

        def sound(*args, **kwargs):
            """Observe the sound-classifier allocation without invoking native inference."""
            entered.append(IN_BATCH.get())
            return {'audioLayers': [], 'soundAnalysisStatus': 'done'}

        base = {'segments': [{'startMs': 0, 'endMs': 1000, 'speaker': 1, 'text': 'Raw text'}],
                'durationMs': 1000, 'language': 'en'}
        with (TemporaryDirectory() as folder,
              patch.dict('os.environ', {'VOICE_RESOURCE_PROFILE': 'throughput'}),
              patch('voice_studio.core.media_resources.batch_slot', side_effect=lambda **kwargs: nullcontext()),
              patch('voice_studio.graph.media_stage_cache.incremental_stages',
                    side_effect=lambda *args: nullcontext(Cache())),
              patch('voice_studio.speech.media_transcript.audio_offset', return_value=0),
              patch('voice_studio.speech.media_transcript.select_sound_classifier', return_value='auto'),
              patch('voice_studio.speech.media_transcript.ffmpeg', side_effect=lambda args, **kwargs: Path(args[-1]).write_bytes(b'pcm')),
              patch('voice_studio.speech.media_transcript.transcribe_wav', return_value=deepcopy(base)),
              patch('voice_studio.speech.media_transcript.analyze_audio_scene', side_effect=sound)):
            source = Path(folder) / 'source.wav'
            source.write_bytes(b'original source')
            transcribe_media(Path(folder), 'en', source)
        self.assertEqual(entered, [True])
