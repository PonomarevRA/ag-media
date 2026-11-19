#!/usr/bin/env python3
"""Prepare allowlisted, scanned source archives; never initialize Git or publish."""
from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile

# Source-only exports deliberately omit installed packages and downloadable resources.
DENIED_PARTS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.engines',
                '.cache', 'data', 'models', 'weights', 'caches', 'cache', 'private',
                'reports', 'screenshots', 'transcripts', 'testmedia', 'training-data',
                'raw-training-data', 'userdata', 'user-data', 'user_data', 'tmp', 'temp',
                'history', 'logs', 'keys', 'secrets', 'uploads', 'recordings', 'bin', 'obj',
                'build', 'dist', '.pytest_cache', '.mypy_cache', '.ruff_cache',
                '.ssh', '.aws', '.codex', '.agents'}
DENIED_SUFFIXES = {'.pyc', '.pyo', '.onnx', '.pt', '.pth', '.ckpt', '.safetensors',
                   '.bin', '.db', '.sqlite', '.sqlite3', '.mp4', '.mov', '.wav',
                   '.mp3', '.flac', '.pem', '.key', '.p12', '.pfx', '.zip', '.tgz',
                   '.gz', '.dmg', '.so', '.dylib', '.dll', '.exe', '.csv', '.tsv', '.jsonl',
                   '.ndjson', '.parquet', '.arrow', '.h5', '.hdf5'}
TEXT_SUFFIXES = {'.py', '.js', '.mjs', '.css', '.html', '.json', '.md', '.txt', '.toml',
                 '.yaml', '.yml', '.sh', '.ps1', '.swift', '.cs', '.csproj', '.spec', '.svg'}
# Only these reviewed generated brand files may bypass the UTF-8 source scanner.
# Regenerating the SVG derivatives requires reviewing and updating their exact hashes.
REVIEWED_BRAND_ASSETS = {
    "web/brand/icon-32.png": "290b91e04419f8129411086ee4f025b19f6c808767260619398a470748d2e525",
    "web/brand/icon-180.png": "26c77af9ce35c56689f44696c148b7f177bafc1a89f2232ec754d93655243dc2",
    "web/brand/icon-512.png": "2f46b36b7cbd8a5a521cb3105fd2cc5e2a8e6b0241ea56c8268532d814953349",
    "web/brand/favicon.ico": "37f372371dba8b2508dcf560889bd64ccc0e7c1e846287a130873c3d331678dd",
    "packaging/icons/AgMediaStudio.icns": "9da1bf0b8acb873ed66fef80cd84c463372d56904bccce4a2078759492b703c1",
    "packaging/icons/AgMediaStudio.ico": "37f372371dba8b2508dcf560889bd64ccc0e7c1e846287a130873c3d331678dd"
}

# This upstream notice contains public attribution addresses. Keep them only
# while the complete file matches the reviewed ONNX Runtime 1.28.2 bytes.
REVIEWED_NOTICE_EMAILS = {
    'voice_studio/vendor/onnxruntime/ThirdPartyNotices.md':
        '0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2',
}

RULES = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'access-token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b'),
    'credential-assignment': re.compile(r'''(?im)\b(?:password|api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}["']'''),
    'credential-url': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
    'host-home-path': re.compile(r'/Users/[A-Za-z0-9._-]+(?:/|\b)|/home/[A-Za-z0-9._-]+(?:/|\b)|[A-Za-z]:\\+Users\\+[A-Za-z0-9._-]+'),
    'email-address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}


class ExportError(ValueError):
    """Carry only reviewed relative paths and rule IDs, never matching text."""

    def __init__(self, findings: list[dict[str, str]]):
        """Store only sanitized findings in the public exception message."""
        self.findings = findings
        super().__init__(json.dumps({'findings': findings}, sort_keys=True))


def issue(path: str, rule: str) -> dict[str, str]:
    """Build a content-free finding suitable for logs and audit files."""
    return {'path': path, 'ruleId': rule}


def relative_path(value: str) -> PurePosixPath:
    """Reject traversal, absolute paths and ambiguous platform separators."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ExportError([issue('<configuration>', 'invalid-relative-path')])
    return path


def denied(path: PurePosixPath) -> bool:
    """Apply non-overridable exclusions to every exported relative path."""
    if any(part.lower() in DENIED_PARTS for part in path.parts):
        return True
    reviewed_hidden = {'.github', '.gitignore', '.gitattributes'}
    if any(part.startswith('.') and part not in reviewed_hidden for part in path.parts):
        return True
    name = path.name.lower()
    return name.startswith('.env') or name in {'.ds_store', 'id_rsa', 'id_ed25519'} or path.suffix.lower() in DENIED_SUFFIXES


def read_profile(config_path: Path, profile: str) -> dict:
    """Load one explicit profile; inheritance is shallow and reviewed in configuration."""
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('schema') != 'ag-media-source-export-v1' or profile not in {'public', 'private'}:
        raise ExportError([issue('<configuration>', 'unknown-profile')])
    selected = config['profiles'][profile]
    base = config.get('common', {})
    return {key: list(base.get(key, [])) + list(selected.get(key, [])) for key in ('files', 'trees', 'exclude')}


def ensure_regular(root: Path, relative: PurePosixPath) -> Path:
    """Reject a symlink at any path component and require one regular source file."""
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ExportError([issue(str(relative), 'symlink')])
    if not current.is_file() or not stat.S_ISREG(current.stat().st_mode):
        raise ExportError([issue(str(relative), 'missing-regular-file')])
    return current


def selected_files(root: Path, profile: dict) -> dict[PurePosixPath, PurePosixPath]:
    """Enumerate only declared files and source trees without following links."""
    result = {}
    for value in profile['files']:
        relative = relative_path(value['source'] if isinstance(value, dict) else value)
        target = relative_path(value['target'] if isinstance(value, dict) else value)
        if denied(relative) or denied(target):
            raise ExportError([issue(str(target), 'forbidden-path')])
        ensure_regular(root, relative)
        if target in result and result[target] != relative:
            raise ExportError([issue(str(target), 'duplicate-target')])
        result[target] = relative
    for tree in profile['trees']:
        relative = relative_path(tree['path'])
        tree_root = root.joinpath(*relative.parts)
        if denied(relative) or tree_root.is_symlink() or not tree_root.is_dir():
            raise ExportError([issue(str(relative), 'invalid-source-tree')])
        # Validate parent directories too; Path.is_dir alone follows symlinks.
        for parent in [relative, *relative.parents]:
            if root.joinpath(*parent.parts).is_symlink():
                raise ExportError([issue(str(relative), 'symlink')])
        extensions = set(tree['extensions'])
        if not extensions <= TEXT_SUFFIXES:
            raise ExportError([issue(str(relative), 'unsupported-source-extension')])
        for folder, directories, files in os.walk(tree_root, followlinks=False):
            kept = []
            for name in sorted(directories):
                item = Path(folder, name)
                item_relative = PurePosixPath(item.relative_to(root).as_posix())
                if denied(item_relative):
                    continue
                if item.is_symlink():
                    raise ExportError([issue(str(item_relative), 'symlink')])
                kept.append(name)
            directories[:] = kept
            for name in sorted(files):
                item = Path(folder, name)
                item_relative = PurePosixPath(item.relative_to(root).as_posix())
                if denied(item_relative) or item.suffix.lower() not in extensions:
                    continue
                ensure_regular(root, item_relative)
                if item_relative in result and result[item_relative] != item_relative:
                    raise ExportError([issue(str(item_relative), 'duplicate-target')])
                result[item_relative] = item_relative
    filtered = {}
    for target, source in sorted(result.items()):
        if not any(fnmatch.fnmatchcase(str(target), pattern) for pattern in profile['exclude']):
            filtered[target] = source
    targets_by_case = {}
    for target in filtered:
        if str(target).casefold() == 'export_manifest.json':
            raise ExportError([issue(str(target), 'reserved-target')])
        key = str(target).casefold()
        if key in targets_by_case:
            raise ExportError([issue(str(target), 'case-colliding-target')])
        targets_by_case[key] = target
    return filtered


def reference_key(value: str) -> str:
    """Normalize literal review inputs for case-insensitive Unicode matching."""
    return unicodedata.normalize('NFC', value).casefold()


def scan_bytes(path: str, data: bytes, username: str = '',
               reference_keys: tuple[str, ...] = ()) -> list[dict[str, str]]:
    """Scan UTF-8 source and paths, returning no matched value or source line."""
    if reference_keys:
        normalized_path = reference_key(path)
        if any(key in normalized_path for key in reference_keys):
            return [issue('<selection>', 'excluded-reference')]
    findings = []
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return [issue(path, 'non-utf8-source')]
    if '\x00' in text:
        findings.append(issue(path, 'binary-source'))
    for rule_id, pattern in RULES.items():
        if pattern.search(path):
            findings.append(issue(path, rule_id))
            continue
        if pattern.search(text):
            reviewed_digest = REVIEWED_NOTICE_EMAILS.get(path) if rule_id == 'email-address' else None
            if reviewed_digest and reviewed_digest == hashlib.sha256(data).hexdigest():
                continue
            findings.append(issue(path, rule_id))
    # Host account names must not leak even without an accompanying home path.
    # Common CI account words are code vocabulary rather than personal account identifiers.
    generic_accounts = {'root', 'runner', 'ubuntu', 'user', 'admin', 'builder', 'build', 'test'}
    if len(username) >= 4 and username.lower() not in generic_accounts and re.search(r'(?<![A-Za-z0-9])' + re.escape(username) + r'(?![A-Za-z0-9])', text + '\n' + path, re.I):
        findings.append(issue(path, 'host-account-name'))
    if reference_keys:
        normalized_text = reference_key(text)
        if any(key in normalized_text for key in reference_keys):
            findings.append(issue(path, 'excluded-reference'))
    return findings


def capture_sources(root: Path, profile: dict,
                    rejected_references: tuple[str, ...] = ()) -> dict[str, bytes]:
    """Read and scan selected bytes once so published content equals inspected content."""
    sources = {}
    findings = []
    username = getpass.getuser()
    references = tuple(reference_key(value.strip()) for value in rejected_references if value.strip())
    for target, relative in selected_files(root, profile).items():
        path = str(target)
        source = ensure_regular(root, relative)
        data = source.read_bytes()
        approved_digest = REVIEWED_BRAND_ASSETS.get(path)
        approved_binary = approved_digest is not None and hashlib.sha256(data).hexdigest() == approved_digest
        findings.extend(scan_bytes(path, b'' if approved_binary else data, username, references))
        sources[path] = data
    if findings:
        raise ExportError(findings)
    if not sources:
        raise ExportError([issue('<selection>', 'empty-export')])
    return sources


def json_bytes(value: dict) -> bytes:
    """Serialize deterministic UTF-8 audit metadata without host identity or timestamps."""
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def write_archive(path: Path, sources: dict[str, bytes]) -> None:
    """Write a stable ZIP with fixed dates and portable source permissions."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(sources.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if name.endswith('.sh') else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def publish_output(temporary: Path, destination: Path) -> None:
    """Claim a fresh target atomically, then move completed output into that owned directory."""
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise ExportError([issue('<destination>', 'destination-exists')]) from None
    try:
        for item in sorted(temporary.iterdir()):
            item.rename(destination / item.name)
    except BaseException:
        # This directory was created by this invocation, so rollback never removes an older target.
        shutil.rmtree(destination)
        raise


def prepare_export(root: Path, destination: Path, profile_name: str, config_path: Path | None = None,
                   *, rejected_references: tuple[str, ...] = ()) -> dict:
    """Create a fresh scanned tree and ZIP outside the source; failures leave no target."""
    root = root.resolve(strict=True)
    proposed = destination.absolute()
    if proposed.exists() or proposed.is_symlink():
        raise ExportError([issue('<destination>', 'destination-exists')])
    destination = proposed.parent.resolve() / proposed.name
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ExportError([issue('<destination>', 'destination-overlaps-source')])
    config_path = config_path or root / 'packaging/export-profiles.json'
    profile = read_profile(config_path, profile_name)
    sources = capture_sources(root, profile, rejected_references)
    manifest = {'schema': 'ag-media-source-manifest-v1', 'profile': profile_name,
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(sources.items())},
                'licensePolicy': 'Original project source: MIT; retain upstream rights/notices; no weights or dependencies bundled.'}
    sources['EXPORT_MANIFEST.json'] = json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.ag-media-export-', dir=destination.parent))
    try:
        tree = temporary / 'repository'
        for name, data in sorted(sources.items()):
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if name.endswith('.sh') else 0o644)
        archive = temporary / 'source.zip'
        write_archive(archive, sources)
        scan_rules = [*RULES, 'host-account-name', 'binary-source', 'non-utf8-source']
        if rejected_references:
            scan_rules.append('excluded-reference')
        audit = {'schema': 'ag-media-source-audit-v1', 'profile': profile_name, 'findings': [],
                 'fileCount': len(sources), 'archiveSha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                 'manifestSha256': hashlib.sha256(sources['EXPORT_MANIFEST.json']).hexdigest(),
                 'scanRuleIds': sorted(scan_rules)}
        (temporary / 'audit.json').write_bytes(json_bytes(audit))
        publish_output(temporary, destination)
        return audit
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    """Run a preparation or read-only source scan and print sanitized JSON only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--profile', required=True, choices=['private', 'public'])
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--reject-references-file', type=Path,
                        help='Read excluded literal references from an external UTF-8 file, one per line')
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        references = ()
        if args.reject_references_file is not None:
            reference_file = args.reject_references_file.resolve(strict=True)
            if reference_file.is_relative_to(root):
                raise ExportError([issue('<configuration>', 'reference-list-must-be-external')])
            references = tuple(line.strip() for line in reference_file.read_text(encoding='utf-8').splitlines()
                               if line.strip())
            if not references:
                raise ExportError([issue('<configuration>', 'empty-reference-list')])
        if args.check_only:
            profile = read_profile(root / 'packaging/export-profiles.json', args.profile)
            sources = capture_sources(root, profile, references)
            result = {'profile': args.profile, 'fileCount': len(sources), 'findings': []}
        else:
            if args.destination is None:
                parser.error('--destination is required unless --check-only is selected')
            result = prepare_export(root, args.destination, args.profile, rejected_references=references)
        print(json.dumps(result, sort_keys=True))
        return 0
    except ExportError as error:
        print(str(error))
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({'findings': [issue('<operation>', 'configuration-or-io-error')]}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
