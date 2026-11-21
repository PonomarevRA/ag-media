"""Exercise real PCM pause placement and bounded pronunciation controls."""
from array import array
from math import sin, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from voice_studio.synthesis.prosody_alignment import RATE, align_phrases, read_pcm, reference_rhythm, validate_rhythm, write_pcm
from voice_studio.synthesis.voice_controls import generation_options, pronunciation_mode


def tone(ms):
    """Make a voiced test signal at known sample boundaries."""
    return array('h', (round(8000*sin(2*pi*170*i/RATE)) for i in range(ms*24))).tobytes()


class ProsodyAlignmentTests(TestCase):
    def test_measured_internal_pause_is_preserved_at_exact_source_samples(self):
        with TemporaryDirectory() as temp:
            p=Path(temp); source=p/'source.wav'; a=p/'a.wav'; b=p/'b.wav'; output=p/'out.wav'
            write_pcm(source, tone(500)+bytes(240*24*2)+tone(600))
            write_pcm(a,tone(500)); write_pcm(b,tone(600))
            plan=reference_rhythm(source)
            self.assertEqual(len(plan['phrases']),2)
            audit=align_phrases([a,b],plan,output)
            result=read_pcm(output)
            self.assertEqual(len(result),len(read_pcm(source)))
            gap=plan['pauses'][0]
            self.assertFalse(any(result[gap['startSample']*2:gap['endSample']*2]))
            self.assertEqual(audit['frames'],1340*24)
            self.assertTrue(any(result[:500*24*2]))

    def test_failed_fit_preserves_previous_result_without_truncating_speech(self):
        with TemporaryDirectory() as temp:
            p=Path(temp); source=p/'source.wav'; output=p/'out.wav'
            write_pcm(source,tone(2000));output.write_bytes(b'previous')
            plan={'sampleRate':RATE,'frames':400*24,'phrases':[{'startSample':0,'endSample':400*24}]}
            with self.assertRaisesRegex(ValueError,'tempo'):
                align_phrases([source],plan,output)
            self.assertEqual(output.read_bytes(),b'previous')

    def test_unmeasured_silence_and_overlapping_slots_are_rejected(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'silent.wav';write_pcm(p,bytes(RATE*2))
            with self.assertRaises(ValueError):reference_rhythm(p)
        for phrases in [[{'startSample':0,'endSample':100},{'startSample':99,'endSample':150}], [{'startSample':False,'endSample':150}]]:
            with self.assertRaises(ValueError):validate_rhythm({'sampleRate':RATE,'frames':150,'phrases':phrases})

    def test_legacy_and_neutral_modes_pass_only_supported_backend_controls(self):
        job={'language':'en','references':[{'text':'Russian reference'}]}
        self.assertEqual(generation_options('qwen',job),{'ref_text':'Russian reference'})
        self.assertEqual(generation_options('chatterbox',job),{})
        self.assertEqual(generation_options('qwen',{**job,'pronunciationMode':'neutral'}),{'ref_text':None})
        self.assertEqual(generation_options('chatterbox',{**job,'pronunciationMode':'neutral'}),{'cfg_weight':0.0})
        self.assertEqual(generation_options('qwen',{**job,'language':'ru','pronunciationMode':'neutral'}),{'ref_text':None})
        with self.assertRaises(ValueError):pronunciation_mode('neutral','de')
        with self.assertRaises(ValueError):pronunciation_mode([], 'en')
        with self.assertRaises(ValueError):generation_options('qwen',{**job,'maxTokens':True})

    def test_quiet_terminal_voice_is_not_discarded_by_envelope_trimming(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);source=p/'quiet-tail.wav';output=p/'out.wav'
            low=array('h',(round(500*sin(2*pi*170*i/RATE)) for i in range(300*24))).tobytes()
            write_pcm(source,tone(500)+low)
            plan=reference_rhythm(source)
            self.assertEqual(plan['phrases'][-1]['endSample'],800*24)
            align_phrases([source],plan,output)
            tail=array('h',read_pcm(output)[600*24*2:750*24*2])
            self.assertGreater(sum(x*x for x in tail)/len(tail),100**2)

    def test_pitch_correction_preserves_declared_zero_gap_and_pitch_centre(self):
        from unittest.mock import patch
        from voice_studio.synthesis.prosody_alignment import centred_corrections, transfer_intonation
        correction=centred_corrections([-6,12,-6],2)
        self.assertAlmostEqual(sum(correction),0,places=8)
        self.assertTrue(all(abs(x)<=2 for x in correction))
        with TemporaryDirectory() as temp:
            p=Path(temp);source=p/'source.wav';output=p/'out.wav'
            write_pcm(source,tone(600)+bytes(240*24*2)+tone(600))
            plan=reference_rhythm(source)
            with patch('voice_studio.synthesis.prosody_alignment.measured_contour',side_effect=[{'pitchHz':[120,150,110]},{'pitchHz':[110,110,110]},{'pitchHz':[120,140,110]}]):
                result=transfer_intonation(source,source,output,rhythm=plan)
            self.assertEqual(result['status'],'applied')
            gap=plan['pauses'][0]
            self.assertFalse(any(read_pcm(output)[gap['startSample']*2:gap['endSample']*2]))
            self.assertEqual(len(read_pcm(output)),len(read_pcm(source)))

    def test_pitch_clock_drift_abstains_without_replacing_previous_audio(self):
        from unittest.mock import patch
        from voice_studio.synthesis.prosody_alignment import transfer_intonation
        with TemporaryDirectory() as temp:
            p=Path(temp);source=p/'source.wav';output=p/'out.wav'
            write_pcm(source,tone(1000));output.write_bytes(b'previous')
            def drifting(args,**kwargs):write_pcm(Path(args[-1]),tone(1200))
            with patch('voice_studio.synthesis.prosody_alignment.measured_contour',return_value={'pitchHz':[120,130,110]}),patch('voice_studio.synthesis.prosody_alignment.ffmpeg',side_effect=drifting):
                result=transfer_intonation(source,source,output)
            self.assertEqual(result['status'],'skipped')
            self.assertEqual(output.read_bytes(),b'previous')

    def test_exact_source_edge_silence_stays_on_its_original_clock(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);source=p/'source.wav';candidate=p/'voice.wav';output=p/'out.wav'
            write_pcm(source,bytes(300*24*2)+tone(500)+bytes(300*24*2))
            write_pcm(candidate,tone(500));plan=reference_rhythm(source)
            self.assertEqual(plan['phrases'],[{'startSample':300*24,'endSample':800*24}])
            align_phrases([candidate],plan,output)
            raw=read_pcm(output)
            self.assertFalse(any(raw[:300*24*2]))
            self.assertFalse(any(raw[800*24*2:]))
            self.assertTrue(any(raw[300*24*2:800*24*2]))

    def test_pitch_refinement_rolls_back_on_quality_regression_or_unknown(self):
        from voice_studio.synthesis.prosody_evaluation import pitch_promotion
        before={'whisper_word_error_rate':.1,'eres2net_encoder_cosine':.7}
        improved={**before,'f0_shape_before_semitones':3,'f0_shape_after_semitones':1}
        self.assertEqual(pitch_promotion(before,improved)['decision'],'candidate')
        for after in [{**improved,'whisper_word_error_rate':.2}, {**improved,'eres2net_encoder_cosine':.65}, {**improved,'f0_shape_after_semitones':4}, {}]:
            self.assertEqual(pitch_promotion(before,after)['decision'],'rollback')
