#!/usr/bin/env python3
"""Read-only A2A YAML validation with automatic PyYAML runtime selection and JSON compatibility."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


KINDS = {"fact", "constraint", "decision", "observation", "failed_attempt", "outcome", "handoff"}
PAYLOADS = {
    "fact": ({"statement", "consequence"}, {"statement", "consequence"}),
    "constraint": ({"requirement", "consequence"}, {"requirement", "consequence"}),
    "decision": ({"problem", "decision", "rationale", "consequences"},
                 {"problem", "context", "options", "decision", "rationale", "consequences"}),
    "observation": ({"observation", "follow_up"}, {"observation", "follow_up"}),
    "failed_attempt": ({"attempt", "result", "cause", "preferred_alternative"},
                       {"attempt", "result", "cause", "preferred_alternative"}),
    "outcome": ({"change", "result", "next_action"}, {"change", "measurement", "result", "next_action"}),
    "handoff": ({"objective", "owner", "state", "expected_result"},
                {"objective", "owner", "participants", "state", "constraints", "expected_result", "outcome", "blocker"}),
}
REQUIRED = {"schema", "id", "kind", "area", "status", "title", "summary", "recorded", "updated", "tags", "sources", "payload"}
OBJECT_ID = re.compile(r"(?:[a-z][a-z0-9-]{0,47}|_cross-cutting)\.(?:fact|constraint|decision|observation|failed_attempt|outcome|handoff)\.[a-z][a-z0-9-]{0,63}")


class ParserUnavailable(ValueError):
    """Automatic YAML checks need a parser; agents can still review files directly."""


def unique_mapping(pairs) -> dict:
    """Reject duplicate/non-string keys in either format instead of silently losing evidence."""
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError('Memory mapping keys must be strings')
        if key in result:
            raise ValueError(f'duplicate mapping key: {key}')
        result[key] = value
    return result


def reject_constant(value: str):
    """Non-finite numbers are not JSON and must not depend on Python's permissive decoder default."""
    raise ValueError(f'Non-finite JSON constant: {value}')


def memory_document(text: str, *, json_only: bool = False) -> Any:
    """Parse JSON or YAML with duplicate-key rejection; YAML uses only safe constructors."""
    try:
        return json.loads(text, object_pairs_hook=unique_mapping, parse_constant=reject_constant)
    except json.JSONDecodeError:
        if json_only:
            raise
    if yaml is None:
        raise ParserUnavailable('PyYAML is unavailable. Read/edit YAML directly using the skill manual checklist; parser validation was not performed.')

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any, deep: bool = False) -> dict:
        pairs = []
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            pairs.append((key, loader.construct_object(value_node, deep=deep)))
        return unique_mapping(pairs)

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        return yaml.load(text, Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise ValueError('Invalid YAML') from error


class MemoryValidator:
    """Follow bounded registry pointers, keeping source contents out of diagnostic output."""

    def __init__(self, root: Path, strict: bool = False):
        self.root = root.resolve()
        self.strict = strict
        self.findings: list[dict] = []
        self.records: dict[str, dict] = {}
        self.visited: set[Path] = set()
        self.legacy_count = 0
        self.read_bytes = 0

    def report(self, code: str, path: Path, detail: str, shape: bool = False) -> None:
        """Historical shape differences warn in compatible mode; corruption remains an error."""
        severity = "warning" if shape and not self.strict else "error"
        self.findings.append({"severity": severity, "code": code,
                              "path": str(path), "detail": detail})

    def read(self, path: Path, schema: str, byte_limit: int) -> dict | None:
        """Read only contained, bounded YAML/JSON mappings with a supported v1 schema."""
        if not path.resolve().is_relative_to(self.root):
            self.report("unsafe_path", path, "Registry reference escapes the memory root.")
            return None
        if path in self.visited:
            self.report("duplicate_path", path, "Registry path is referenced more than once.")
            return None
        self.visited.add(path)
        if len(self.visited) > 10000:
            self.report("inspection_limit", path, "More than 10000 references; narrow the audit.")
            return None
        try:
            size = path.stat().st_size
            self.read_bytes += size
            if self.read_bytes > 64 * 1024 * 1024:
                self.report("inspection_limit", path, "Audit exceeds the 64 MiB total read bound.")
                return None
            if size > 1024 * 1024:
                self.report("read_limit", path, "Document exceeds the 1 MiB read safety bound.")
                return None
            if size > byte_limit:
                self.report("size_limit", path, f"{size} bytes exceeds {byte_limit}.", shape=True)
            value = memory_document(path.read_text(encoding="utf-8"), json_only=path.suffix == '.json')
        except ParserUnavailable as error:
            self.report('parser_unavailable', path, str(error))
            return None
        except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
            code = 'unreadable_json' if path.suffix == '.json' else 'unreadable_yaml'
            self.report(code, path, type(error).__name__)
            return None
        if not isinstance(value, dict):
            self.report("invalid_document", path, "Expected a memory object mapping.")
            return None
        if value.get("schema") != schema or value.get("version", 1) != 1:
            self.report("unsupported_schema", path, "Document schema/version is not supported; left intact.")
            return None
        if schema != "agent-memory-object/v1" and "version" not in value:
            self.report("missing_version", path, "Registry version is required.", shape=True)
        return value

    def children(self, document: dict, key: str, path: Path, maximum: int) -> list[dict]:
        """Check collection shapes before following arbitrary registry-supplied values."""
        values = document.get(key)
        if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
            self.report("invalid_collection", path, f"{key} must be a list of mappings.")
            return []
        if len(values) > maximum:
            self.report("collection_limit", path, f"{key} contains {len(values)} entries; limit {maximum}.", shape=True)
        return values[:10000]

    def pointer(self, parent: Path, descriptor: dict) -> Path | None:
        """Resolve relative registry paths; absolute and parent traversal never become reads."""
        relative = descriptor.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            self.report("invalid_path", parent, "Missing or invalid relative path.")
            return None
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            self.report("unsafe_path", parent, "Absolute paths and parent traversal are forbidden.")
            return None
        return parent.parent / path

    def object(self, path: Path, area: str, kind: str, descriptor: dict) -> None:
        """Validate v1 identities and links without inventing approval or factual truth."""
        value = self.read(path, "agent-memory-object/v1", 4096)
        if value is None:
            return
        identifier = value.get("id")
        if not isinstance(identifier, str) or not OBJECT_ID.fullmatch(identifier):
            self.report("invalid_id", path, "Object ID does not match the v1 grammar.")
            return
        if identifier in self.records:
            self.report("duplicate_id", path, identifier)
            return
        self.records[identifier] = value
        if (value.get("area") != area or value.get("kind") != kind or descriptor.get("id") != identifier
                or not identifier.startswith(area + "." + kind + ".")):
            self.report("identity_mismatch", path, "Object and registry identities differ.")
        missing = REQUIRED - value.keys()
        extra = value.keys() - REQUIRED - {"supersedes", "related"}
        if missing or extra:
            self.report("object_fields", path, f"Missing: {sorted(missing)}; additional: {sorted(extra)}.", shape=True)
        for field, maximum in (("title", 96), ("summary", 160)):
            if not isinstance(value.get(field), str) or len(value[field]) > maximum:
                self.report("text_field", path, f"{field} must be a string <= {maximum} characters.", shape=True)
        payload = value.get("payload")
        if not isinstance(payload, dict):
            self.report("invalid_payload", path, "Payload must be a mapping.")
        elif kind in PAYLOADS:
            required, allowed = PAYLOADS[kind]
            if required - payload.keys() or payload.keys() - allowed:
                self.report("payload_fields", path, f"Missing: {sorted(required - payload.keys())}; additional: {sorted(payload.keys() - allowed)}.", shape=True)
            for field, content in payload.items():
                if (isinstance(content, str) and len(content) > 600 or
                        isinstance(content, list) and (len(content) > 6 or
                            any(isinstance(item, str) and len(item) > 600 for item in content))):
                    self.report("payload_limit", path, f"{field} exceeds the compact v1 limit.", shape=True)
        for field, maximum in (("tags", 8), ("sources", 6), ("supersedes", 3), ("related", 6)):
            entries = value.get(field, [])
            if not isinstance(entries, list) or len(entries) > maximum:
                self.report("list_field", path, f"{field} must be a list <= {maximum} entries.", shape=True)
        self.metadata(value, path)
        for field in ("supersedes", "related"):
            entries = value.get(field, [])
            if isinstance(entries, list) and any(not isinstance(item, str) for item in entries):
                self.report("invalid_link", path, f"{field} must contain object IDs.")

    def metadata(self, value: dict, path: Path) -> None:
        """Check portable source/date/tag fields without interpreting their truth."""
        for field in ("recorded", "updated"):
            try:
                text = str(value.get(field))
                date.fromisoformat(text)
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                    raise ValueError("date shape")
            except ValueError:
                self.report("invalid_date", path, field, shape=True)
        tags = value.get("tags")
        if isinstance(tags, list):
            if any(not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9-]+", tag) for tag in tags):
                self.report("invalid_tag", path, "Tags must be lowercase short labels.", shape=True)
            elif len(set(tags)) != len(tags):
                self.report("duplicate_tag", path, "Repeated tags.", shape=True)
        sources = value.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    self.report("invalid_source", path, "Source must be a mapping.", shape=True)
                    continue
                reference = source.get("ref")
                source_kind = source.get("kind")
                if (not isinstance(source_kind, str) or source_kind not in {"code", "test", "config", "document", "issue", "task", "external"}
                        or not isinstance(reference, str) or not reference.strip() or len(reference) > 180):
                    self.report("invalid_source", path, "Unsupported source kind/ref; left intact.", shape=True)
        status = value.get("status")
        if not isinstance(status, str) or status not in {"active", "proposed", "approved", "superseded", "archived", "blocked", "completed", "cancelled"}:
            self.report("invalid_status", path, "Unknown lifecycle status.", shape=True)
        payload = value.get("payload")
        if value.get("kind") == "handoff" and isinstance(payload, dict):
            state = payload.get("state")
            if not isinstance(state, str) or state not in {"planned", "active", "blocked", "completed", "cancelled"}:
                self.report("invalid_handoff_state", path, "Unknown handoff state.", shape=True)

    def registry(self, path: Path, area: str | None = None, kind: str | None = None) -> None:
        """Walk the prescribed root/area/kind hierarchy; never execute linked material."""
        level = "root" if area is None else "area" if kind is None else "kind"
        limits = {"root": 8192, "area": 2048, "kind": 12288}
        value = self.read(path, f"agent-memory-{level}-registry/v1", limits[level])
        if value is None:
            return
        if area is not None and value.get("area") != area or kind is not None and value.get("kind") != kind:
            self.report("identity_mismatch", path, "Registry area/kind differs from its parent.")
        field, maximum = ("areas", 40) if level == "root" else ("kinds", 7) if level == "area" else ("objects", 40)
        for descriptor in self.children(value, field, path, maximum):
            required = ({"id", "summary", "path"} if level == "root" else
                        {"kind", "summary", "path"} if level == "area" else
                        {"id", "title", "status", "summary", "path", "tags", "recorded", "updated"})
            if required - descriptor.keys():
                self.report("descriptor_fields", path, f"Missing: {sorted(required - descriptor.keys())}.", shape=True)
            if level == "kind":
                self.metadata(descriptor, path)
                title = descriptor.get("title")
                if not isinstance(title, str) or len(title) > 96:
                    self.report("descriptor_title", path, "Missing or oversized descriptor title.", shape=True)
                tags = descriptor.get("tags")
                if not isinstance(tags, list) or len(tags) > 8:
                    self.report("descriptor_tags", path, "Descriptor tags must be a list <= 8.", shape=True)
            summary = descriptor.get("summary")
            if not isinstance(summary, str) or len(summary) > (160 if level == "kind" else 120):
                self.report("descriptor_summary", path, "Missing or oversized registry summary.", shape=True)
            child = self.pointer(path, descriptor)
            if child is None:
                continue
            if level == "root":
                child_area = descriptor.get("id")
                if not isinstance(child_area, str):
                    self.report("invalid_area", path, "Area ID must be a string.")
                else:
                    self.registry(child, child_area)
            elif level == "area":
                child_kind = descriptor.get("kind")
                if not isinstance(child_kind, str) or child_kind not in KINDS:
                    self.report("unsupported_kind", path, "Unknown kind; not reinterpreted.")
                else:
                    self.registry(child, area, child_kind)
            else:
                self.object(child, area, kind, descriptor)

    def links(self) -> None:
        """Detect dangling links and supersession cycles with an iterative bounded walk."""
        graph = {}
        for identifier, record in self.records.items():
            graph[identifier] = []
            for field in ("supersedes", "related"):
                entries = record.get(field, [])
                if not isinstance(entries, list):
                    continue
                for target in entries:
                    if not isinstance(target, str):
                        continue
                    if target not in self.records:
                        self.report("dangling_link", self.root, f"{identifier}: {field} target {target} was not loaded.")
                    elif field == "supersedes":
                        graph[identifier].append(target)
        state = {}
        for start in graph:
            stack = [(start, False)]
            while stack:
                node, complete = stack.pop()
                if complete:
                    state[node] = 2
                    continue
                if state.get(node) == 1:
                    self.report("supersession_cycle", self.root, node)
                    continue
                if state.get(node) == 2:
                    continue
                state[node] = 1
                stack.append((node, True))
                for target in graph[node]:
                    stack.append((target, False))

    def validate(self) -> dict:
        """Return diagnostics only; compatible Markdown is counted without schema claims."""
        registries = []
        for name in ('registry.a2a.json', 'registry.a2a.yaml'):
            path = self.root / name
            if path.exists() or path.is_symlink():
                registries.append(path)
        if len(registries) > 1:
            self.report('ambiguous_registry', self.root, 'Both JSON and YAML root registries exist; choose one explicitly.')
        elif registries:
            self.registry(registries[0])
            self.links()
        # Names only: legacy content remains governed by the project's approved format.
        for position, path in enumerate(self.root.rglob("*.md")):
            if position == 10000:
                self.report("inspection_limit", self.root, "Legacy inventory exceeded 10000 paths.")
                break
            if path.is_file() and path.resolve().is_relative_to(self.root):
                self.legacy_count += 1
        if not registries and not self.legacy_count:
            self.report("memory_not_found", self.root, "No A2A root registry or legacy Markdown records found.")
        return {"mode": "strict" if self.strict else "compatible", "objects": len(self.records),
                "legacy_markdown": self.legacy_count, "findings": self.findings,
                "errors": sum(item["severity"] == "error" for item in self.findings),
                "warnings": sum(item["severity"] == "warning" for item in self.findings)}


def runtime_candidates(memory_root: Path) -> list[Path]:
    """Find only conventional local/active environments; never interpret paths from memory records.

    Keep symlink spellings: two virtualenvs can share one binary but have different site packages.
    Candidates are probed in isolated mode, so a repository's yaml.py cannot satisfy the check.
    """
    project = memory_root.resolve()
    if project.name == 'agent-memory' and project.parent.name == 'docs':
        project = project.parent.parent
    candidates = [Path(sys.executable)]
    environments = []
    active = os.environ.get('VIRTUAL_ENV')
    if active:
        environments.append(Path(active))
    environments.extend([project / '.venv', project / 'venv', Path(__file__).resolve().parents[1] / '.venv'])
    for environment in environments:
        candidates.append(environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
    found, seen = [], set()
    for candidate in candidates:
        candidate = candidate.absolute()
        key = os.path.normcase(str(candidate))
        if key not in seen and candidate.is_file():
            seen.add(key)
            found.append(candidate)
    return found


def validate_with_available_yaml(memory_root: Path, arguments: list[str]) -> int | None:
    """Retry once with an existing PyYAML environment; preserve stdout, flags and validation exit code.

    Probes are bounded to three seconds each and the read-only validation to sixty seconds.
    No package installation, shell evaluation or fallback parser is performed.
    """
    for candidate in runtime_candidates(memory_root):
        try:
            probe = subprocess.run([str(candidate), '-I', '-B', '-c', 'import yaml'],
                                   capture_output=True, timeout=3, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode:
            continue
        print(f'agent-memory: validating with PyYAML from {candidate}', file=sys.stderr)
        try:
            result = subprocess.run([str(candidate), '-I', '-B', str(Path(__file__).resolve()),
                                     *arguments, '--no-runtime-discovery'], timeout=60, check=False)
            return result.returncode
        except (OSError, subprocess.TimeoutExpired):
            print('agent-memory: automatic runtime failed; parser validation remains unavailable.', file=sys.stderr)
            return None
    return None


def main() -> int:
    """Select a repository's memory directory and report validation without mutations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("compatible", "strict"), default="compatible")
    parser.add_argument("--json", action="store_true")
    parser.add_argument('--no-runtime-discovery', action='store_true',
                        help='Check only this interpreter; do not probe existing PyYAML environments.')
    args = parser.parse_args()
    root = args.root / "docs" / "agent-memory" if (args.root / "docs" / "agent-memory").is_dir() else args.root
    result = MemoryValidator(root, args.mode == "strict").validate()
    unavailable = any(row['code'] == 'parser_unavailable' for row in result['findings'])
    if unavailable and not args.no_runtime_discovery:
        exit_code = validate_with_available_yaml(root, sys.argv[1:])
        if exit_code is not None:
            return exit_code
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else
          f"{result['objects']} A2A objects; {result['legacy_markdown']} legacy Markdown; "
          f"{result['errors']} errors; {result['warnings']} warnings")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
