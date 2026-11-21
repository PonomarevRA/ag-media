# Compatibility and evidence

Version 3.4 keeps the skill name, project locations, A2A v1 record shapes and existing
bootstrap flags. It does not require AgMemory, MCP, embeddings or a model provider.

## Existing projects

- Keep approved Markdown layouts, `index.md`, decision records and A2A v1. Read and
  update them in their established format. The retained `memory-records.md` has the
  legacy templates. Never automatically convert, delete or move existing records.
- When legacy Markdown has no index and no A2A registry, list Markdown filenames
  under `docs/agent-memory` (`rg --files docs/agent-memory -g '*.md'`), select
  task-related files, and inspect at most 20 matching titles/headings before
  opening relevant records. State the inspected scope on a miss; do not create
  an index or migrate records merely to retrieve them.
- In mixed projects, use the A2A registry and the legacy index for the relevant area;
  report search coverage. Avoid making a second copy of the same knowledge.
- New A2A records use `.a2a.yaml` and the v1 shape in `a2a-memory-schema.yaml`. Existing JSON remains supported; no helper converts it automatically. Missing PyYAML never requires a format migration. Existing unknown fields and historical
  payload aliases are warnings in compatible validation, not a migration demand.
- Keep an unsupported future schema intact and report it as unsupported. Do not claim
  its records passed validation or reinterpret it as v1.

## Updating copies

The global synchronizer updates only files matching a valid previous managed manifest.
Changed files, symlinks and collisions with newly introduced package paths are conflicts.
Extra project files remain in place. `--check` changes no file, and exit 0 may mean
`would-create` or `would-update`; inspect the statuses rather than treating exit 0 as
proof that everything is current.

Unmanaged copies remain `legacy`. Explicit `--adopt` takes a sibling
`agent-memory.before-adoption` backup first and refuses to overwrite an existing backup.
It adopts the skill package only; existing memory is never migrated. The old global
bootstrap flags continue to work. No project-local helper is deleted.

## Read-only validation

`scripts/validate_memory.py` uses PyYAML for automated YAML checks. If unavailable, the CLI probes known current/active/project/skill environments with isolated imports and bounded timeouts, then reruns this same read-only script once. It never installs packages or uses executable paths from memory content. `--no-runtime-discovery` disables retry. Runtime diagnostics go to stderr; stdout/JSON and the validation exit code are preserved. Library callers remain explicit: `MemoryValidator` reports parser availability and never launches a process.

Agents may read and edit YAML directly without PyYAML. Follow the manual checklist in SKILL.md: preserve fields, indentation, IDs, references and limits; inspect the diff; report automated validation as unavailable. JSON still works with standard Python. Duplicate keys, non-finite JSON values and unsafe pointers are rejected; two roots remain an ambiguity error.

Run it against a repository or its memory directory:

```sh
python3 scripts/validate_memory.py /path/to/repository --mode compatible --json
```

Compatible mode validates JSON or legacy YAML syntax, core identities, registry paths and links while
warning about historical field/size differences. Strict mode enforces the current v1
shape and limits. Legacy Markdown is discovered and reported as readable text, not
claimed to satisfy a YAML schema. Neither mode writes records or proves their truth.

## Reuse and procedures without a schema break

Use existing source references to identify the relevant revision or supporting artifact.
In payload text distinguish an observation from an inference and preserve constraints.
Do not infer an approval from `status: approved` without its actual decision source.
Recheck changed source/version/environment before treating a historical result as current.

For a reusable procedure, keep its detailed prerequisites, steps and verification in
the existing runbook; a v1 `fact` points to it and states applicability. Link an `outcome`
for an actual attempt, including failures and unknown results. Success is specific to
the tested conditions. Keep skill quality, discovery and transfer evaluation separate.

## Authorized format or area migration

A migration requires the actual task to authorize it. Snapshot the source documents, preserve business payloads, dates and statuses, rewrite all affected paths/IDs and related/supersedes references, and record a relative old→new mapping with preservation checks. Do not normalize historical fields merely to eliminate compatibility warnings. Read old references through that migration map.
