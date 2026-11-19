# Architecture and ownership

[Русский](Architecture-ru.md) · [ag-media-studio](../../README.md)

The browser owns workspace presentation. HTTP and MCP accept requests. Orchestrators select stages and launch workers. The graph stores compact references, timestamps and provenance; it does not replace the source file.

```text
web/ → HTTP app/server.py → speech/ + vision/ + audio/
                         → graph/ → timeline / MCP
synthesis/ ← speaker samples + prosody/ → WAV / dubbing
core/ → budgets, locks, stages, cache lifetime
workers/ → isolated model processes
```

## Cross-module invariants

- Root compatibility launchers forward to canonical `voice_studio/` modules.
- Preview and export should reuse the same technical-layer model.
- Face visibility, active speaking and speaker identity are different evidence types.
- An object profile retains observations, original labels, confidence and geometry.
- Chunk-local times must be shifted onto the source clock.
- Model environments, weights, user media and temporary caches are outside the documentation build.

- [Functional modules](Modules.md)
- [Graph and MCP](Agents.md)
- [Analysis stages](Pipeline.md)
