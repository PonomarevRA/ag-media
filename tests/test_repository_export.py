"""Safety contracts for source-only repository preparation using synthetic trees."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/export_repository.py'
if not MODULE_PATH.exists():
    MODULE_PATH = Path(__file__).with_name('export_repository.py')
SPEC = importlib.util.spec_from_file_location('repository_export', MODULE_PATH)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class RepositoryExportTests(unittest.TestCase):
    """Exercise failure boundaries without copying user material or creating Git state."""

    def setUp(self):
        """Create a tiny reviewed source tree in a disposable directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'source'
        self.root.mkdir()
        self.write('app/main.py', '"""Synthetic app."""\nprint("hello")\n')
        self.write('README.md', '# Example\n')
        self.write('internal/design.md', 'Internal synthetic architecture.\n')
        self.write('packaging/export-profiles.json', json.dumps({
            'schema': 'ag-media-source-export-v1',
            'common': {'files': ['README.md'], 'trees': [{'path': 'app', 'extensions': ['.py']} ]},
            'profiles': {'public': {}, 'private': {'files': ['internal/design.md']}}}))

    def tearDown(self):
        """Remove only the fixture directory owned by this test."""
        self.temporary.cleanup()

    def write(self, name, text):
        """Write a synthetic input under the owned fixture root."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def destination(self, name):
        """Return a fresh output next to the fixture source, never within it."""
        return self.root.parent / name

    def configuration(self):
        """Read the fixture allowlist for deliberate invalid configuration tests."""
        return json.loads((self.root / 'packaging/export-profiles.json').read_text())

    def save_configuration(self, configuration):
        """Persist an intentionally adjusted synthetic allowlist."""
        self.write('packaging/export-profiles.json', json.dumps(configuration))

    def test_profiles_are_source_only_and_source_is_unchanged(self):
        """Verify profile differences, forbidden-file omissions and unchanged source hashes."""
        for name in ('data/private.txt', '.env', '.git/config', '.venv/lib/secret.py',
                     'app/models/model.onnx', 'app/reports/report.py', 'app/cache/state.py'):
            self.write(name, 'DO NOT EXPORT\n')
        before = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob('*') if p.is_file()}
        for profile in ('public', 'private'):
            destination = self.destination(profile)
            result = EXPORT.prepare_export(self.root, destination, profile)
            files = set(result_path.relative_to(destination / 'repository').as_posix()
                        for result_path in (destination / 'repository').rglob('*') if result_path.is_file())
            expected = {'app/main.py', 'README.md', 'EXPORT_MANIFEST.json'}
            if profile == 'private':
                expected.add('internal/design.md')
            self.assertEqual(expected, files)
            self.assertEqual([], result['findings'])
            with zipfile.ZipFile(destination / 'source.zip') as archive:
                self.assertEqual(expected, set(archive.namelist()))
                self.assertTrue(all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()))
        after = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_only_exact_reviewed_brand_bytes_bypass_text_scanning(self):
        """Reviewed icons export intact; replacing one with arbitrary binary fails the existing scanner."""
        source = MODULE_PATH.parents[1]
        name = 'web/brand/icon-32.png'
        approved = source / name
        self.assertEqual(hashlib.sha256(approved.read_bytes()).hexdigest(), EXPORT.REVIEWED_BRAND_ASSETS[name])
        target = self.root / name
        target.parent.mkdir(parents=True)
        target.write_bytes(approved.read_bytes())
        profile = {'files': [name], 'trees': [], 'exclude': []}
        self.assertEqual(EXPORT.capture_sources(self.root, profile)[name], approved.read_bytes())
        target.write_bytes(b'\x89PNG\x00not-reviewed')
        with self.assertRaises(EXPORT.ExportError):
            EXPORT.capture_sources(self.root, profile)

    def test_stable_archives_and_manifests(self):
        """Compare repeated export bytes and verify a selected file hash."""
        first = EXPORT.prepare_export(self.root, self.destination('one'), 'public')
        second = EXPORT.prepare_export(self.root, self.destination('two'), 'public')
        self.assertEqual(first, second)
        self.assertEqual((self.destination('one') / 'source.zip').read_bytes(),
                         (self.destination('two') / 'source.zip').read_bytes())
        manifest = json.loads((self.destination('one') / 'repository/EXPORT_MANIFEST.json').read_text())
        self.assertEqual(hashlib.sha256((self.root / 'app/main.py').read_bytes()).hexdigest(),
                         manifest['files']['app/main.py'])

    def test_file_and_directory_symlink_escapes_are_rejected(self):
        """Reject both file links and directory links before any output is published."""
        for directory in (False, True):
            with self.subTest(directory=directory):
                link = self.root / ('app/outside' if directory else 'app/linked.py')
                link.symlink_to(self.root.parent if directory else self.root / 'README.md', target_is_directory=directory)
                destination = self.destination('blocked-link')
                with self.assertRaises(EXPORT.ExportError) as caught:
                    EXPORT.prepare_export(self.root, destination, 'public')
                self.assertEqual('symlink', caught.exception.findings[0]['ruleId'])
                self.assertFalse(destination.exists())
                link.unlink()

    def test_selected_file_through_symlink_parent_is_rejected(self):
        """Reject a declared file whose ancestor is a symlink."""
        (self.root / 'linked').symlink_to(self.root / 'internal', target_is_directory=True)
        configuration = self.configuration()
        configuration['common']['files'].append('linked/design.md')
        self.save_configuration(configuration)
        with self.assertRaises(EXPORT.ExportError):
            EXPORT.prepare_export(self.root, self.destination('blocked-parent'), 'public')

    def test_secrets_and_host_paths_fail_without_echoing_matched_text(self):
        """Block synthetic sensitive text and keep diagnostics content-free."""
        samples = {
            'private-key': '-----BEGIN ' + 'PRIVATE KEY-----',
            'access-token': 'ghp_' + 'a' * 30,
            'credential-assignment': 'password = "' + 'a' * 24 + '"',
            'credential-url': 'https://' + 'operator:private-value' + '@' + 'invalid.example',
            'host-home-path': '/' + 'Users/' + 'synthetic-user/project',
            'email-address': 'synthetic-person' + '@' + 'invalid.example',
        }
        for rule, value in samples.items():
            with self.subTest(rule=rule):
                self.write('app/main.py', value)
                destination = self.destination('blocked-' + rule)
                with self.assertRaises(EXPORT.ExportError) as caught:
                    EXPORT.prepare_export(self.root, destination, 'public')
                self.assertIn(rule, [finding['ruleId'] for finding in caught.exception.findings])
                self.assertNotIn(value, str(caught.exception))
                self.assertFalse(destination.exists())

    def test_denied_paths_cannot_be_added_to_either_profile(self):
        """Keep mandatory exclusions stronger than either profile allowlist."""
        for profile in ('public', 'private'):
            for name in ('data/private.txt', '.env.production', 'model.onnx', 'private/notes.md'):
                with self.subTest(profile=profile, name=name):
                    configuration = self.configuration()
                    configuration['profiles'][profile]['files'] = [name]
                    self.save_configuration(configuration)
                    self.write(name, 'synthetic\n')
                    with self.assertRaises(EXPORT.ExportError) as caught:
                        EXPORT.prepare_export(self.root, self.destination('denied'), profile)
                    self.assertIn('forbidden-path', [finding['ruleId'] for finding in caught.exception.findings])
                    configuration['profiles'][profile]['files'] = []
                    self.save_configuration(configuration)

    def test_overlapping_existing_and_traversal_destinations_are_rejected(self):
        """Protect existing outputs and reject source overlap and unsafe source paths."""
        for target in (self.root / 'output', self.root, self.root.parent):
            with self.subTest(target=target.name):
                with self.assertRaises(EXPORT.ExportError):
                    EXPORT.prepare_export(self.root, target, 'public')
        destination = self.destination('occupied')
        destination.mkdir()
        marker = destination / 'preserve.txt'
        marker.write_text('keep')
        with self.assertRaises(EXPORT.ExportError):
            EXPORT.prepare_export(self.root, destination, 'public')
        self.assertEqual('keep', marker.read_text())
        for unsafe in ('../outside.py', '/absolute.py', 'C:\\absolute.py'):
            configuration = self.configuration()
            configuration['common']['files'] = [unsafe]
            self.save_configuration(configuration)
            with self.assertRaises(EXPORT.ExportError):
                EXPORT.prepare_export(self.root, self.destination('traversal'), 'public')

    def test_reviewed_readme_mapping_preserves_the_original(self):
        """Export the reviewed README mapping without modifying the original file."""
        self.write('packaging/public-README.md', '# Exported source README\n')
        configuration = self.configuration()
        configuration['common']['files'] = [{'source': 'packaging/public-README.md', 'target': 'README.md'}]
        self.save_configuration(configuration)
        destination = self.destination('mapped')
        EXPORT.prepare_export(self.root, destination, 'public')
        self.assertEqual('# Exported source README\n', (destination / 'repository/README.md').read_text())
        self.assertEqual('# Example\n', (self.root / 'README.md').read_text())
        configuration['common']['files'].append('README.md')
        self.save_configuration(configuration)
        with self.assertRaises(EXPORT.ExportError) as caught:
            EXPORT.prepare_export(self.root, self.destination('collision'), 'public')
        self.assertEqual('duplicate-target', caught.exception.findings[0]['ruleId'])

    def test_manifest_and_case_colliding_targets_are_rejected(self):
        """Protect generated metadata and portable case-insensitive target identity."""
        for target in ('EXPORT_MANIFEST.json', 'export_manifest.json', 'readme.md'):
            with self.subTest(target=target):
                configuration = self.configuration()
                configuration['common']['files'] = ['README.md', {'source': 'app/main.py', 'target': target}]
                self.save_configuration(configuration)
                with self.assertRaises(EXPORT.ExportError):
                    EXPORT.prepare_export(self.root, self.destination('reserved'), 'public')

    def test_windows_paths_and_personal_accounts_are_detected(self):
        """Cover source-escaped Windows home paths and generic CI account handling."""
        for slash_count in (1, 2):
            value = 'C:' + '\\' * slash_count + 'Users' + '\\' * slash_count + 'synthetic-person'
            findings = EXPORT.scan_bytes('fixture.py', value.encode())
            self.assertIn('host-home-path', [finding['ruleId'] for finding in findings])
        findings = EXPORT.scan_bytes('fixture.py', b'syntheticperson', 'syntheticperson')
        self.assertEqual('host-account-name', findings[0]['ruleId'])
        self.assertEqual([], EXPORT.scan_bytes('fixture.py', b'runner.os', 'runner'))

    def test_concurrent_destination_creation_is_preserved(self):
        """Reserve output without replacing a target created after source capture begins."""
        destination = self.destination('concurrent')
        original_capture = EXPORT.capture_sources

        def capture_then_claim(root, profile, rejected_references):
            """Simulate another writer claiming the requested target during the scan."""
            sources = original_capture(root, profile, rejected_references)
            destination.mkdir()
            (destination / 'preserve.txt').write_text('keep')
            return sources

        with patch.object(EXPORT, 'capture_sources', side_effect=capture_then_claim):
            with self.assertRaises(EXPORT.ExportError):
                EXPORT.prepare_export(self.root, destination, 'public')
        self.assertEqual('keep', (destination / 'preserve.txt').read_text())
        self.assertEqual([], list(destination.parent.glob('.ag-media-export-*')))

    def test_external_reference_scan_does_not_publish_or_echo_matches(self):
        """Reject a synthetic reference found in content or a selected path."""
        reference = 'Épisode de synthèse 521'
        lower_decomposed = 'e\u0301pisode de synthe\u0300se 521'
        cases = [('app/main.py', lower_decomposed, 'app/main.py'),
                 (f'app/{lower_decomposed}.py', 'synthetic source', '<selection>')]
        for index, (name, content, reported_path) in enumerate(cases):
            with self.subTest(name=name):
                source = self.write(name, content)
                destination = self.destination(f'reference-{index}')
                with self.assertRaises(EXPORT.ExportError) as caught:
                    EXPORT.prepare_export(self.root, destination, 'public',
                                          rejected_references=(reference,))
                self.assertEqual([EXPORT.issue(reported_path, 'excluded-reference')],
                                 caught.exception.findings)
                self.assertNotIn(lower_decomposed, str(caught.exception))
                self.assertFalse(destination.exists())
                source.write_text('synthetic source', encoding='utf-8')

    def test_shipped_profiles_exclude_private_research_tests(self):
        """Keep interview-derived research hooks out of both profiles using synthetic stand-ins."""
        config_path = Path(__file__).resolve().parents[1] / 'packaging/export-profiles.json'
        research_paths = {'tests/test_facial_expressions.py', 'tests/test_voice_profiles.py',
                          'tests/test_media_graph_interval.py'}
        for name in research_paths:
            self.write(name, '"""Synthetic stand-in; no original interview content."""\n')
        self.write('tests/test_synthetic_allowed.py', '"""Independent synthetic regression."""\n')
        for name in ('public', 'private'):
            with self.subTest(profile=name):
                shipped = EXPORT.read_profile(config_path, name)
                profile = {'files': [], 'trees': [{'path': 'tests', 'extensions': ['.py']}],
                           'exclude': shipped['exclude']}
                selected = {str(path) for path in EXPORT.selected_files(self.root, profile)}
                self.assertFalse(selected & research_paths)
                self.assertIn('tests/test_synthetic_allowed.py', selected)

    def test_binary_bytes_are_rejected_even_with_a_source_extension(self):
        """Reject binary input disguised as a source file."""
        (self.root / 'app/main.py').write_bytes(b'\xff\x00')
        with self.assertRaises(EXPORT.ExportError) as caught:
            EXPORT.prepare_export(self.root, self.destination('binary'), 'public')
        self.assertEqual('non-utf8-source', caught.exception.findings[0]['ruleId'])


if __name__ == '__main__':
    unittest.main()
