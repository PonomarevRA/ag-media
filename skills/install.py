"""Install bundled Studio skills into an explicitly selected project folder."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

# The installer can run directly from an installed, signed application bundle.
sys.dont_write_bytecode = True
SKILLS = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILLS / "agent-memory" / "scripts"))
from bootstrap_project_skill import (  # noqa: E402
    expected_files,
    install_memory_layout,
    root_registry,
    synchronize_target,
)

CLIENTS = {
    "codex": Path(".agents/skills"),
    "claude": Path(".claude/skills"),
    "cursor": Path(".cursor/skills"),
}
SCENARIOS = ("setup", "interval-review", "ui-review", "dialogue-audit")


def safe_path(project, path):
    """Reject redirected directories or files anywhere below the project."""
    relative = path.relative_to(project)
    for part in relative.parts:
        project /= part
        if project.is_symlink():
            return False
    return True


def install_scenario(project, target, source, check):
    """Copy a scenario only if missing; keep all local edits intact."""
    entry = target / "SKILL.md"
    if not safe_path(project, entry):
        return "conflict"
    content = source.read_bytes()
    if target.exists():
        return "current" if entry.is_file() and entry.read_bytes() == content else "preserved"
    if not check:
        target.mkdir(parents=True)
        with entry.open("xb") as output:
            output.write(content)
    return "would-create" if check else "created"


def install_agent_memory(project, target, files, check):
    """Install a missing copy; another distribution may own an existing one."""
    if not safe_path(project, target):
        return "conflict"
    status = synchronize_target(target, files, False, True)
    if status == "would-create" and not check:
        return synchronize_target(target, files, False, False)
    return "preserved" if status == "would-update" else status


def local_memory(project, check):
    """Create a Git-ignored registry for local analysis references."""
    root = project / ".ag-media-memory"
    ignore = root / ".gitignore"
    yaml = root / "registry.a2a.yaml"
    alternate = root / "registry.a2a.json"
    paths = (ignore, yaml, alternate)
    if not all(safe_path(project, path) for path in paths):
        return "conflict"
    if not root.exists():
        if not check:
            root.mkdir()
            ignore.write_text("*\n", encoding="utf-8")
            yaml.write_text(root_registry(), encoding="utf-8")
        return "would-create" if check else "created"
    protected = ignore.is_file() and ignore.read_text(encoding="utf-8") == "*\n"
    registries = [path for path in (yaml, alternate) if path.exists()]
    return "current" if protected and len(registries) == 1 and registries[0].is_file() else "conflict"


def install(project, clients, check):
    """Install skills and initialize empty registries without modifying existing data."""
    files = expected_files(SKILLS / "agent-memory")
    statuses = {}
    for client in clients:
        target = project / CLIENTS[client]
        memory = target / "agent-memory"
        statuses[str(memory.relative_to(project))] = install_agent_memory(project, memory, files, check)
        for suffix in SCENARIOS:
            name = "ag-media-" + suffix
            scenario = target / name
            statuses[str(scenario.relative_to(project))] = install_scenario(
                project, scenario, SKILLS / name / "SKILL.md", check
            )
    statuses["docs/agent-memory"] = install_memory_layout(project, check)
    statuses[".ag-media-memory"] = local_memory(project, check)
    return statuses


def main():
    """Select the target project and report each destination without concealing conflicts."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Existing project directory")
    parser.add_argument("--client", choices=(*CLIENTS, "all"), default="codex")
    parser.add_argument("--check", action="store_true", help="Preview without changing files")
    options = parser.parse_args()
    try:
        project = options.project.expanduser().resolve(strict=True)
        if not project.is_dir():
            parser.error("--project must be a directory")
        clients = CLIENTS if options.client == "all" else (options.client,)
        statuses = install(project, clients, options.check)
    except (OSError, ValueError) as error:
        print(f"Install stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(statuses, indent=2))
    blocked = {"conflict", "legacy", "legacy-layout", "conflict-layout"}
    return int(any(status in blocked for status in statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())
