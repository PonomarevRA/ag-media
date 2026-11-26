# Architecture

AG Media Studio runs a local HTTP service for the browser and desktop interfaces. The
MCP server uses that service for interactive operations and can read saved media
graphs from the workspace. The desktop host owns the HTTP process and its lifecycle.

| Location | Responsibility |
| --- | --- |
| `voice_studio/app/` | HTTP routing and transport, workspace lifecycle, media upload and UI bridge |
| `voice_studio/core/` | Stage scheduling, subprocess commands, resource budgets and shared cache coordination |
| `voice_studio/speech/`, `audio/`, `vision/` | ASR and diarization, audio analysis, and video or screenshot analysis |
| `voice_studio/synthesis/` and `prosody/` | Voice generation, metrics, dubbing and text preparation |
| `voice_studio/graph/` | Saved media graph, incremental analysis and interval queries |
| `voice_studio/mcp/` | Agent tool catalog and adapters for the studio and graphs |
| `voice_studio/workers/` | Model processes that run in a provisioned interpreter |
| `web/`, `scripts/`, `packaging/` | Browser code, operator utilities and desktop or source builds |

The four root launchers `ag_media_studio.py`, `ag_media_studio_mcp.py`,
`server.py` and `voice_mcp.py` preserve existing commands and MCP configurations.
`voice_paths.py` is the installed path utility used by the application and
provisioning scripts. Run optional setup modules from a source checkout with
`python -m voice_studio.synthesis.setup_commercial_voices` or
`python -m voice_studio.synthesis.setup_dub_background` in the corresponding
environment. See [Getting started](GETTING-STARTED.md) for prerequisites.

## HTTP and worker boundaries

`app/server.py` dispatches routes and calls workspace and analysis services.
`app/http_transport.py` handles HTTP responses, loopback write checks, body limits
and byte-range streaming for both the browser and native server. It translates
application exceptions into HTTP errors.

Model subprocesses keep their own interpreter, arguments and environment. They are
launched by package module name from the studio's source root, including when the
host was started from a different directory. Audio and OCR Swift helpers live
alongside their Python callers under `audio/native/` and `vision/native/`.
Native staging and source exports include both files. The incremental analysis
fingerprint includes their paths and contents.

An application update preserves explicit activation, settings and selections for
built-in mods. A changed user mod requires activation against its current
fingerprint. Worker snapshots still verify the exact code and model declarations
captured for the run. A change in reported model availability alone does not
invalidate that snapshot.

Workspace data, installed dependencies and model weights live outside the
application bundle. [Storage and model paths](wiki/Models-and-storage.md) documents
the paths and environment overrides.

## Object comparison

The optional final object review measures background-masked crop similarity
alongside the original box-interior similarity. It runs GrabCut on up to 32
existing crops, with a four-second budget checked between crops. It checks mask
size and stability before comparing overlapping foreground pixels in normalized
coordinates. A stable mask can still include a neighboring object within the
detector box. An uncertain mask yields no foreground comparison.
The graph keeps the raw candidate pairs in
`objectForegroundMatches` and the pairs within a profile in
`foregroundComparisons`. These scores are experimental diagnostics. Track
and profile identity use the existing evidence and safeguards.
