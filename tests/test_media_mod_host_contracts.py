"""Check host-enforced canonical results after real model-format adapters."""
from unittest import TestCase
from unittest.mock import patch
import wave

import test_media_mod_integration as fixtures
from voice_studio.mod_api import mod_operation
from voice_studio.mods import ModExecutionError
from voice_studio.speech import recognition
from voice_studio.speech.media_transcript import transcribe_media


class HostContractIntegrationTests(TestCase):
    """Reuse isolated package helpers without inheriting their separately-run cases."""

    setUp = fixtures.MediaModIntegrationTests.setUp
    tearDown = fixtures.MediaModIntegrationTests.tearDown
    package = fixtures.MediaModIntegrationTests.package

    def test_prepared_vendor_words_convert_into_real_transcript(self):
        """A copied worker adapts input, runs the vendor model, then converts output to ASR v1."""
        adapter = '''def prepare_input(payload, settings):
    return {'source': payload['jobs'][0][0], 'locale': payload['language']}

def convert_output(raw, settings):
    return [{'transcription': [{'text': word['value'], 'offsets': {'from': round(word['begin'] * 1000), 'to': round(word['finish'] * 1000)}} for word in raw['words']]}]
'''
        declaration = {'contract': {'id': 'studio.asr', 'version': 1},
                       'inputContract': {'id': 'studio.named-input', 'version': 1},
                       'prepare_input': 'adapter.py:prepare_input', 'convert_output': 'adapter.py:convert_output',
                       'rawOutputSchema': {'type': 'object', 'properties': {'words': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': True}}}, 'required': ['words'], 'additionalProperties': False}}
        model = "assert payload['locale'] == 'en'\nassert payload['source'].endswith('.wav')\nreturn {'words': [{'value': 'adapted format', 'begin': 0.05, 'finish': 0.7}]}"
        self.package('test.converted', 'speech.asr', model, declaration=declaration, assets={'adapter.py': adapter})
        self.registry.select('speech.asr', 'test.converted')
        source = self.root / 'source.wav'
        with wave.open(str(source), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\x40\x00' * 16000)
        result = transcribe_media(self.root, 'en', source, allow_empty=True)
        self.assertEqual(result['text'], 'adapted format')
        self.assertEqual((result['segments'][0]['startMs'], result['segments'][0]['endMs']), (50, 700))
        self.assertEqual(result['recognitionProvider'], 'test.converted')

    def test_undeclared_legacy_provider_still_cannot_return_invalid_known_output(self):
        """Known host consumers reject malformed values before indexing them or attempting fallback."""
        self.package('test.invalid-asr', 'speech.asr', "return [{'words': []}]")
        self.registry.select('speech.asr', 'test.invalid-asr')
        with patch.object(recognition, 'ready') as native, self.assertRaisesRegex(ModExecutionError, 'transcription'):
            recognition.run_whisper([(self.root / 'source.wav', self.root / 'output')], 'en', [], None, 2, 10, 0)
        native.assert_not_called()

    def test_known_batch_cardinality_is_enforced_after_canonical_validation(self):
        """A structurally valid ASR result cannot silently omit one input job."""
        self.package('test.short-asr', 'speech.asr', "return []")
        self.registry.select('speech.asr', 'test.short-asr')
        with self.assertRaisesRegex(ModExecutionError, 'one canonical result'):
            recognition.run_whisper([(self.root / 'source.wav', self.root / 'output')], 'en', [], None, 2, 10, 0)

    def test_contract_discovery_and_builtin_references_are_detached_metadata(self):
        """Authors can discover compatible versions and machine-readable schemas without executing code."""
        contracts = mod_operation('contracts')
        self.assertTrue(any(item['id'] == 'studio.asr' and item['version'] == 1 for item in contracts['contracts']))
        rows = self.registry.catalog()['mods']
        for row in rows:
            if row['id'].startswith('builtin.whisper'):
                self.assertEqual(row['manifest']['capabilities'][0]['contract'], {'id': 'studio.asr', 'version': 1})
        contracts['contracts'].clear()
        self.assertTrue(mod_operation('contracts')['contracts'])
