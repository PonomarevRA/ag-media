"""QA regressions for long-upload integration, voice identity boundaries and public exports."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from mcp.server.fastmcp import FastMCP

from voice_studio.app import media_pipeline
from voice_studio.graph.media_analysis_revolver import AnalysisRevolver
from voice_studio.graph.media_graph_ingest import ID_BLOCK, SPEECH_WINDOW_MS, analyze_source
from voice_studio.graph.media_graph_model import build_graph
from voice_studio.mcp.dialogue_audit_mcp import register_dialogue_tools
from voice_studio.speech.speaker_pipeline import project_voice_index
from voice_studio.speech.voice_profiles import VERSION as VOICE_VERSION, VoiceProfileConfig


VOICE_ID = 'a'*32
SOURCE_ID = 'b'*64


def metadata(*, audio=True, duration=SPEECH_WINDOW_MS+1000):
    """Describe a long source while native probe/ASR/model work is controlled by fixtures."""
    streams = [{'kind':'video', 'index':0, 'startMs':0, 'durationMs':duration}]
    if audio:
        streams.append({'kind':'audio', 'index':1, 'startMs':0, 'durationMs':duration})
    return {'kind':'video', 'durationMs':duration, 'hasAudio':audio, 'streams':streams}


def local_speech(*args, **options):
    """Supply raw ASR labels and native turns without introducing a voice-profile decision."""
    return {'language':'en', 'segments':[{'startMs':100, 'endMs':900, 'text':'Raw transcript', 'speaker':1}],
            'speechTurns':[{'startMs':100, 'endMs':900, 'speaker':1}]}


def decode_audio(source, work, start, end, stream):
    """Represent a bounded decoded interval; tests never run an acoustic encoder."""
    return Path(work)/'controlled.wav', 0, end-start


def public_profile():
    """Return one valid vector-free canonical identity for graph and MCP boundary tests."""
    return {'id':VOICE_ID, 'number':1, 'status':'provisional', 'sampleCount':1,
            'visualSubjects':[], 'references':[{'key':'reference', 'sourceId':SOURCE_ID,
                'startMs':0, 'endMs':1500, 'rawSpeaker':9, 'explicitVideo':False}]}


def sdk_data(result):
    """Read actual SDK structured output, with its JSON text fallback."""
    if isinstance(result, tuple):
        return result[1]
    return json.loads(next(block.text for block in result if block.type=='text'))


class MediaUpgradeIntegrationTests(TestCase):
    def analyze_fixture(self, root, *, audio=True, speakers=True, revolver=None):
        """Exercise real long-window orchestration while replacing external decoders/models."""
        source = root/'source.mov'
        source.write_bytes(b'controlled-source-for-integration')
        with (patch('voice_studio.graph.media_stage_cache.analysis_fingerprint', return_value='qa-revision'),
              patch('voice_studio.graph.media_graph_ingest.probe_source', return_value=metadata(audio=audio)),
              patch('voice_studio.graph.media_graph_ingest.extract_speech', side_effect=decode_audio),
              patch('voice_studio.graph.media_graph_ingest.transcribe_wav', side_effect=local_speech),
              patch('voice_studio.graph.media_graph_ingest.collect_visual', return_value=({}, {})),
              patch('voice_studio.graph.media_graph_ingest.scan_shots', return_value=[]),
              patch('voice_studio.graph.media_graph_ingest.analyze_audio_scene',
                    return_value={'soundAnalysisStatus':'too-short', 'audioLayers':[]})):
            return analyze_source(root, source, 'video', 'en', visual=False, scene=False,
                                  speakers=speakers, revolver=revolver)

    def test_speakers_disabled_does_not_initialize_encoder_or_drop_raw_asr(self):
        """Speaker-free long transcription must remain usable without any encoder weights."""
        with TemporaryDirectory() as folder, patch('voice_studio.speech.voice_profiles.local_encoder_id',
                side_effect=AssertionError('Encoder initialization is forbidden')) as encoder:
            report = self.analyze_fixture(Path(folder), speakers=False)
        encoder.assert_not_called()
        self.assertEqual([cue['text'] for cue in report['segments']], ['Raw transcript']*2)
        self.assertNotIn('voiceProfiles', report)

    def test_no_audio_does_not_initialize_encoder_even_when_speakers_requested(self):
        """A silent video track does not create an acoustic model or source voice identity."""
        with TemporaryDirectory() as folder, patch('voice_studio.speech.voice_profiles.local_encoder_id',
                side_effect=AssertionError('Encoder initialization is forbidden')) as encoder:
            report = self.analyze_fixture(Path(folder), audio=False, speakers=True)
        encoder.assert_not_called()
        self.assertEqual(report['segments'], [])
        self.assertNotIn('voiceProfiles', report)

    def test_missing_registry_and_encoder_weights_preserve_raw_transcript(self):
        """An unavailable optional voice model cannot abort measured ASR text or raw labels."""
        with TemporaryDirectory() as folder, patch('voice_studio.speech.voice_profiles.local_encoder_id',
                side_effect=FileNotFoundError('controlled missing embedding weights')) as encoder:
            root = Path(folder)
            report = self.analyze_fixture(root, revolver=AnalysisRevolver(root/'checkpoints'))
        self.assertEqual(encoder.call_count, 1)
        self.assertEqual([cue['speaker'] for cue in report['segments']], [1, ID_BLOCK+1])
        self.assertEqual([cue['text'] for cue in report['segments']], ['Raw transcript']*2)
        self.assertIn('voiceProfileWarning', report)

    def test_corrupt_registry_json_and_schema_preserve_raw_transcript(self):
        """Invalid JSON and structurally corrupt valid JSON are cache failures, not ASR failures."""
        for corruption in ('{', '[]', 'missing-profile-fields', 'unhashable-subject', 'nested-vector'):
            with self.subTest(corruption=corruption), TemporaryDirectory() as folder:
                root = Path(folder)
                with (patch('voice_studio.speech.voice_profiles.local_encoder_id', return_value='qa-encoder'),
                      patch('voice_studio.speech.speaker_pipeline.attach_voice_profiles', return_value={'profiles': []})):
                    self.analyze_fixture(root, revolver=AnalysisRevolver(root/'checkpoints'))
                registries = list((root/'checkpoints'/'voice-registry').rglob(VOICE_VERSION+'.json'))
                self.assertTrue(registries)
                registry = registries[0]
                if corruption in ('missing-profile-fields', 'unhashable-subject', 'nested-vector'):
                    source_digest = sha256(b'controlled-source-for-integration').hexdigest()
                    payload = {'version':VOICE_VERSION, 'sourceId':source_digest, 'encoderId':'qa-encoder',
                               'config':asdict(VoiceProfileConfig()), 'dimension':2, 'profiles':[{}]}
                    if corruption != 'missing-profile-fields':
                        payload['profiles'] = [{'id':VOICE_ID, 'number':1, 'subjectId':None,
                            'visualSubjects':[], 'samples':[]}]
                    if corruption == 'unhashable-subject':
                        payload['profiles'][0]['subjectId'] = {'corrupt':'nested identity'}
                    elif corruption == 'nested-vector':
                        payload['profiles'][0]['samples'] = [{'vector':[{'corrupt':'vector'},1]}]
                    for registry in registries:
                        registry.write_text(json.dumps(payload))
                else:
                    for registry in registries:
                        registry.write_text(corruption)
                with patch('voice_studio.speech.voice_profiles.local_encoder_id', return_value='qa-encoder'):
                    report = self.analyze_fixture(root, revolver=AnalysisRevolver(root/'checkpoints'))
                self.assertEqual([cue['text'] for cue in report['segments']], ['Raw transcript']*2)
                self.assertIn('voiceProfileWarning', report)
                self.assertNotIn('voiceProfiles', report)

    def test_raw_unknown_cluster_one_does_not_bind_to_canonical_voice_one(self):
        """Numerically equal raw cluster/profile labels never create a false identity edge."""
        report = {'language':'en', 'durationMs':5000,
                  'voiceProfiles':{'version':VOICE_VERSION, 'profiles':[public_profile()]},
                  'segments':[
                      {'startMs':0, 'endMs':1500, 'text':'Confirmed canonical voice', 'speaker':9,
                       'voiceProfileId':VOICE_ID, 'profileSpeakerNumber':1,
                       'voiceProfileEvidence':{'state':'matched','basis':'audio-profile','profileId':VOICE_ID}},
                      {'startMs':1700, 'endMs':2500, 'text':'Unknown raw cluster one', 'speaker':1,
                       'voiceProfileId':None, 'voiceProfileEvidence':{'state':'abstained','basis':'ambiguous-audio'}},
                      {'startMs':2700, 'endMs':3500, 'text':'Raw overlap', 'speaker':1, 'overlap':True,
                       'overlaps':[{'startMs':2700,'endMs':3500,'speakers':[1,2]}],
                       'voiceProfileId':None, 'voiceProfileEvidence':{'state':'abstained','basis':'overlap'}}],
                  'dialogueAudit':{'candidates':[{'id':'issue-one','startMs':1700,'endMs':2500,
                       'reasons':['speaker-boundary'], 'evidence':{'cueIndices':[1]}}]}}
        project_voice_index(report)
        graph = build_graph(report, kind='video', duration_ms=5000)
        nodes = {row['id']:row for row in graph['nodes']}
        edges = set(graph['edges'])
        self.assertIn(('utterance:0:0','spoken_by','speaker:1'), edges)
        self.assertFalse(any(a=='utterance:1700:0' and relation=='spoken_by' for a,relation,b in edges))
        self.assertEqual(nodes['utterance:1700:0']['data']['rawSpeaker'], 1)
        overlap_edges = [row for row in edges if row[0].startswith('overlap:')]
        self.assertTrue(any(relation=='raw_participant' and target.startswith('raw_audio_cluster:')
                            for source,relation,target in overlap_edges))
        self.assertFalse(any(relation=='participant' and target=='speaker:1'
                             for source,relation,target in overlap_edges))
        issue = nodes['dialogue_issue:issue-one']
        self.assertTrue(issue['data']['notConfirmedError'])
        self.assertIn('voice_profile:'+VOICE_ID, nodes)

    def test_long_upload_uses_shared_analyze_source_and_propagates_settings(self):
        """The HTTP upload path must reuse window orchestration instead of short-file truncation."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'long.mov'; source.write_bytes(b'long upload fixture')
            duration = media_pipeline.MAX_MEDIA_SECONDS*1000+1
            result = {'segments':[], 'language':'en', 'durationMs':duration}
            with (patch.object(media_pipeline, 'probe_source', return_value=metadata(duration=duration)),
                  patch.object(media_pipeline, 'workspace_temp', return_value=root/'temporary'),
                  patch.object(media_pipeline, 'ensure_temporary_root'),
                  patch.object(media_pipeline, 'use_cache', side_effect=lambda group,directory:nullcontext()),
                  patch.object(media_pipeline, 'transcribe_media', side_effect=AssertionError('Short route forbidden')),
                  patch.object(media_pipeline, 'analyze_source', return_value=result) as windows):
                report = media_pipeline.transcribe_uploaded_media(root, 'en', source,
                    speakers=False, visual=True, scene=True, speaker_count=None,
                    object_sampling='every-third', audio_classifier='auto', people_count=False)
            self.assertEqual(windows.call_count, 1)
            args, options = windows.call_args
            self.assertEqual(args, (root, source, 'video', 'en'))
            self.assertIs(options['speakers'], False)
            self.assertTrue(options['visual'] and options['scene'])
            self.assertEqual(options['object_sampling'], 'every-third')
            self.assertIsInstance(options['revolver'], AnalysisRevolver)
            self.assertEqual(options['revolver'].directory, root/'temporary'/'analysis-checkpoints'/'bounded-v2'/sha256(source.read_bytes()).hexdigest())
            self.assertEqual(report['sourceDigest'], sha256(source.read_bytes()).hexdigest())


    def test_http_upload_handler_reaches_shared_long_source_pipeline(self):
        """The actual upload handler remains wired to the common long-file coordinator."""
        from voice_studio.app.server import Handler

        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'long-upload.mov'; source.write_bytes(b'controlled uploaded source')
            duration = media_pipeline.MAX_MEDIA_SECONDS*1000+1
            report = {'segments':[], 'language':'en', 'durationMs':duration}
            handler = SimpleNamespace(server=SimpleNamespace(studio=SimpleNamespace(root=root)),
                uploaded_media=lambda limit:nullcontext(source), reply=Mock())
            with (patch.object(media_pipeline, 'probe_source', return_value=metadata(duration=duration)),
                  patch.object(media_pipeline, 'workspace_temp', return_value=root/'temporary'),
                  patch.object(media_pipeline, 'ensure_temporary_root'),
                  patch.object(media_pipeline, 'use_cache', side_effect=lambda group,directory:nullcontext()),
                  patch.object(media_pipeline, 'analyze_source', return_value=report) as windows,
                  patch('voice_studio.app.server.transcribe_media',
                        side_effect=AssertionError('Legacy short-file handler route forbidden'))):
                Handler.receive_media(handler, 'en', speakers=False, visual=False, scene=False)
            windows.assert_called_once()
            self.assertEqual(handler.reply.call_args.args[0], 200)
            self.assertEqual(handler.reply.call_args.args[1]['sourceDigest'], sha256(source.read_bytes()).hexdigest())


class PublicVoiceExportIntegrationTests(IsolatedAsyncioTestCase):
    async def export(self, report):
        """Call the public tool through a fresh real SDK server and an actual local JSON file."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            analysis = root/'analysis.json'; analysis.write_text(json.dumps(report))
            server = FastMCP('voice-public-boundary-qa')
            register_dialogue_tools(server, root)
            return sdk_data(await server.call_tool('speaker_profiles', {'analysis_path':str(analysis)}))

    async def test_profile_and_nested_evidence_vectors_are_not_public(self):
        """Allowlisted profile/reference/video facts survive while arbitrary nested model data is omitted."""
        poison = {'secretVectorMarker':[.17,.29,.43]}
        profile = public_profile()
        profile.update(privateEmbedding=poison)
        profile['references'][0]['modelPayload'] = poison
        profile['visualSubjects'] = [{'id':'visual-subject','scope':'source','modelPayload':poison}]
        evidence = {'state':'matched','basis':'audio-profile','modelPayload':poison,
                    'quality':{'accepted':True,'reasons':[],'modelPayload':poison},
                    'video':{'subjectId':'visual-subject','evidence':[{'faceId':2,'modelPayload':poison}]}}
        report = {'sourceDigest':SOURCE_ID, 'voiceProfiles':{'version':VOICE_VERSION,'profiles':[profile]},
                  'segments':[{'startMs':0,'endMs':1500,'speaker':9,'voiceProfileId':VOICE_ID,
                               'voiceProfileEvidence':evidence}]}
        result = await self.export(report)
        self.assertEqual(result['profiles'][0]['id'], VOICE_ID)
        self.assertEqual(result['profiles'][0]['references'][0]['startMs'], 0)
        self.assertTrue(result['cues'][0]['evidence']['quality']['accepted'])
        self.assertNotIn('secretVectorMarker', json.dumps(result))
        self.assertNotIn('modelPayload', json.dumps(result))

    async def test_unhashable_voice_profile_ids_and_invalid_times_are_skipped_safely(self):
        """Untrusted cue identifiers and timestamps cannot raise or carry nested private payloads."""
        poison = {'secretVectorMarker':[.17,.29,.43]}
        report = {'sourceDigest':SOURCE_ID, 'voiceProfiles':{'version':VOICE_VERSION,'profiles':[public_profile()]},
                  'segments':[
                      {'startMs':0,'endMs':1000,'voiceProfileId':deepcopy(poison)},
                      {'startMs':0,'endMs':1000,'voiceProfileId':[VOICE_ID]},
                      {'startMs':deepcopy(poison),'endMs':1000,'voiceProfileId':VOICE_ID},
                      {'startMs':0,'endMs':1000,'speaker':9,'voiceProfileId':VOICE_ID,
                       'voiceProfileEvidence':{'basis':'audio-profile'}}]}
        result = await self.export(report)
        self.assertEqual(len(result['cues']), 1)
        self.assertEqual(result['cues'][0]['rawSpeaker'], 9)
        self.assertNotIn('secretVectorMarker', json.dumps(result))

    async def test_poisoned_outer_metadata_and_raw_label_do_not_leak_nested_vectors(self):
        """Known public metadata fields must reject nested values as strictly as profile evidence."""
        poison = {'secretVectorMarker':[.17,.29,.43]}
        report = {'sourceDigest':deepcopy(poison),
                  'voiceProfiles':{'version':deepcopy(poison),'profiles':[public_profile()]},
                  'segments':[{'startMs':0,'endMs':1500,'speaker':deepcopy(poison),
                               'voiceProfileId':VOICE_ID,'voiceProfileEvidence':{'basis':'audio-profile'}}]}
        result = await self.export(report)
        self.assertNotIn('secretVectorMarker', json.dumps(result))
        self.assertEqual(result['profiles'][0]['id'], VOICE_ID)
