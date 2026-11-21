#!/usr/bin/env python3
"""Install or safely synchronize agent-memory project copies without keeping a script there."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


SKILL_NAME = "agent-memory"
SKILL_VERSION = "3.4.0"
MANIFEST_NAME = ".agent-memory-managed.json"
TARGET_ROOTS = (
    Path(".agents") / "skills",
    Path(".claude") / "skills",
    Path(".cursor") / "skills",
)
MEMORY_ROOT = Path("docs") / "agent-memory"
ROOT_REGISTRY = MEMORY_ROOT / "registry.a2a.yaml"
JSON_ROOT_REGISTRY = MEMORY_ROOT / "registry.a2a.json"
LEGACY_MEMORY_INDEX = MEMORY_ROOT / "index.md"
PROJECT_SKILL_SECTION = """## Project installation

This is a managed project copy of `agent-memory`. AG Media Studio can install a copy for Codex, Claude Code or Cursor. Update it using Studio’s skill installer; the synchronizer is not part of an installed project copy.

"""


def repository_root() -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 else None


def target_paths(root: Path) -> list[Path]:
    return [root / relative / SKILL_NAME for relative in TARGET_ROOTS]


def project_skill_content(source: Path) -> bytes:
    content = (source / "SKILL.md").read_text(encoding="utf-8")
    before, found, remainder = content.partition("## Install and update the shared project skill\n")
    if not found:
        raise RuntimeError("Cannot build project skill: installation section is missing.")
    _, next_heading, after = remainder.partition("## Store A2A objects by functional area\n")
    if not next_heading:
        raise RuntimeError("Cannot build project skill: memory layout section is missing.")
    return (before + PROJECT_SKILL_SECTION + "## Store A2A objects by functional area\n" + after).encode("utf-8")


def expected_files(source: Path) -> dict[Path, bytes]:
    """Build the portable package, retaining legacy references used by existing clients."""
    files = {
        Path("SKILL.md"): project_skill_content(source),
        Path("references") / "a2a-memory-schema.yaml": (
            source / "references" / "a2a-memory-schema.yaml"
        ).read_bytes(),
        Path("agents") / "openai.yaml": (source / "agents" / "openai.yaml").read_bytes(),
    }
    for relative in ("references/compatibility.md", "references/memory-records.md", "references/retrieval-and-validity.md",
                     "references/ag-media-analysis.md",
                     "scripts/validate_memory.py", "requirements-validation.txt"):
        path = Path(relative)
        files[path] = (source / path).read_bytes()
    return files


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def manifest_for(files: dict[Path, bytes]) -> dict[str, object]:
    return {
        "managed_by": SKILL_NAME,
        "version": SKILL_VERSION,
        "files": {path.as_posix(): digest(content) for path, content in files.items()},
    }


def read_manifest(target: Path) -> dict[str, object] | None:
    """Treat malformed or unsafe manifests as unmanaged; never use them to authorize writes."""
    try:
        data = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("managed_by") != SKILL_NAME:
        return None
    file_hashes = data.get("files")
    if not isinstance(file_hashes, dict) or not file_hashes:
        return None
    for raw_path, expected_hash in file_hashes.items():
        path = Path(raw_path)
        if (path.is_absolute() or ".." in path.parts or "\\" in raw_path or
                not isinstance(expected_hash, str) or len(expected_hash) != 64 or
                any(character not in "0123456789abcdef" for character in expected_hash)):
            return None
        try:
            (target / path).resolve().relative_to(target.resolve())
        except ValueError:
            return None
    return data


def matches_manifest(target: Path, manifest: dict[str, object]) -> bool:
    file_hashes = manifest["files"]
    assert isinstance(file_hashes, dict)
    for raw_path, expected_hash in file_hashes.items():
        path = target / str(raw_path)
        if not path.is_file() or digest(path.read_bytes()) != expected_hash:
            return False
    return True


def write_target(target: Path, files: dict[Path, bytes]) -> None:
    """Publish known package files while retaining unrelated files and legacy records."""
    for relative, content in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest_for(files), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def root_registry() -> str:
    """Create ordinary YAML without requiring a parser or converting existing memory."""
    return 'schema: agent-memory-root-registry/v1\nversion: 1\nareas: []\n'


def install_memory_layout(root: Path, check: bool) -> str:
    """Initialize only a contained ordinary directory; never follow layout symlinks."""
    registry = root / ROOT_REGISTRY
    registries = [registry, root / JSON_ROOT_REGISTRY]
    boundary = root.resolve()
    for entry in registries:
        for candidate in (entry, *entry.parents):
            if candidate == boundary:
                break
            if candidate.is_symlink():
                return "conflict-layout"
        if not entry.resolve().is_relative_to(boundary):
            return "conflict-layout"
    legacy_index = root / LEGACY_MEMORY_INDEX
    existing = [entry for entry in registries if entry.exists()]
    if len(existing) > 1:
        return 'conflict-layout'
    if existing:
        return "existing"
    if legacy_index.exists() or any((root / MEMORY_ROOT).rglob("*.md")):
        return "legacy-layout"
    if not check:
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(root_registry(), encoding="utf-8")
    return "would-create" if check else "created"


def synchronize_target(target: Path, files: dict[Path, bytes], adopt: bool, check: bool) -> str:
    """Synchronize only unchanged managed files; preview and conflict paths never write."""
    if target.is_symlink():
        return "conflict"
    for relative in [*files, Path(MANIFEST_NAME)]:
        candidate = target / relative
        if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents):
            return "conflict"
    if not target.exists():
        if not check:
            write_target(target, files)
        return "would-create" if check else "created"

    manifest = read_manifest(target)
    if manifest is None:
        if not adopt:
            return "legacy"
        if not check:
            # Adoption is explicit and reversible, including locally extended legacy helpers.
            backup = target.with_name(target.name + ".before-adoption")
            if backup.exists():
                return "adoption-backup-exists"
            shutil.copytree(target, backup, symlinks=True)
            write_target(target, files)
        return "would-adopt" if check else "adopted"

    if not matches_manifest(target, manifest):
        return "conflict"
    # A new package file must not silently replace an extension absent from the old manifest.
    for relative, content in files.items():
        candidate = target / relative
        if relative.as_posix() not in manifest["files"] and candidate.exists():
            if not candidate.is_file() or candidate.read_bytes() != content:
                return "conflict"

    desired = manifest_for(files)
    if manifest == desired:
        return "current"
    if not check:
        write_target(target, files)
    return "would-update" if check else "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or safely synchronize agent-memory project skills.")
    parser.add_argument("--check", action="store_true", help="Report state without writing files.")
    parser.add_argument("--adopt", action="store_true", help="Replace a legacy copy after explicit owner approval.")
    args = parser.parse_args()

    root = repository_root()
    if root is None:
        print("Not inside a Git repository; no project skill was changed.", file=sys.stderr)
        return 2

    files = expected_files(Path(__file__).resolve().parent.parent)
    statuses = {target: synchronize_target(target, files, args.adopt, args.check) for target in target_paths(root)}
    layout_status = install_memory_layout(root, args.check)

    for target, status in statuses.items():
        print(f"{status} {target.relative_to(root)}")
    layout_path = ROOT_REGISTRY
    if not (root / ROOT_REGISTRY).exists() and (root / JSON_ROOT_REGISTRY).exists():
        layout_path = JSON_ROOT_REGISTRY
    print(f"{layout_status} {layout_path}")

    blocked = {"legacy", "conflict", "adoption-backup-exists"}
    if any(status in blocked for status in statuses.values()) or layout_status in {"legacy-layout", "conflict-layout"}:
        print("Some project skills or the memory layout were not changed; resolve legacy status explicitly.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # A diagnostic must never remove even a project-local copy of itself.
    raise SystemExit(main())
