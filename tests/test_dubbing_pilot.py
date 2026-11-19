"""Exercise the pilot's rejected-stage recovery without invoking a model or live tracker."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock,patch

from voice_studio.synthesis.prosody_alignment import write_pcm
from tests.test_prosody_alignment import tone


def load_script(name):
    """Load a repository CLI as a module without running its argument parser."""
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class DubbingPilotTests(TestCase):
    def test_failed_phrase_evaluation_cannot_promote_a_leftover_wav(self):
        pilot=load_script('run-dubbing-prosody-pilot.py');client=MagicMock()
        client.create_run.return_value=SimpleNamespace(info=SimpleNamespace(run_id='synthetic-test'))
        def audio_file(*args):write_pcm(args[-1],tone(1000))
        def generate(engine,mode,text,reference,seed,output):write_pcm(output,tone(1000));return 0.01
        def score(source,candidate,text,root):
            if candidate.name.endswith('phrases.wav'):raise ValueError('evaluation rejected')
            return {'whisper_word_error_rate':0.,'eres2net_encoder_cosine':.8,'duration_error_ms':0.}
        with TemporaryDirectory() as temp:
            p=Path(temp);source=p/'source.wav';source.write_bytes(b'local source marker')
            metadata=p/'input.json';metadata.write_text(json.dumps({'sourceLanguage':'ru','cues':[{'startMs':0,'endMs':1000,'speaker':1,'sourceText':'Исходник','englishText':'Source'}]}))
            args=SimpleNamespace(output=p/'out',report=metadata,source=source,background=p/'bed.wav',cue=0,reference_cue=None,phrase=None,seed=[421],engine=['qwen'])
            with patch.object(pilot,'source_streams',return_value=(1000,'h264','aac')),patch.object(pilot,'audio_offset',return_value=0),patch.object(pilot,'decode_original',side_effect=lambda source,target,rate:write_pcm(target,tone(1000))),patch.object(pilot,'isolate_reference_audio',side_effect=lambda source,bed,target,offset:write_pcm(target,tone(1000))),patch.object(pilot,'create_tracker',return_value=(client,'test')),patch.object(pilot,'generate_take',side_effect=generate),patch.object(pilot,'metrics_for',side_effect=score):
                report=pilot.run(args)
            self.assertEqual([r['status'] for r in report['runs']],['ok','ok','rejected','rejected'])
            self.assertIn('No accepted phrase',report['runs'][-1]['error'])
            self.assertTrue((args.output/'qwen-421-phrases.wav').is_file())

    def test_package_refuses_to_replace_an_existing_export(self):
        package=load_script('package-prosody-ab.py')
        with TemporaryDirectory() as temp:
            p=Path(temp);(p/'report.json').write_text(json.dumps({'cue':{}}))
            out=p/'existing';out.mkdir();(out/'keep').write_text('original')
            args=SimpleNamespace(pilot=p,output=out)
            with self.assertRaises(FileExistsError):package.package(args)
            self.assertEqual((out/'keep').read_text(),'original')
