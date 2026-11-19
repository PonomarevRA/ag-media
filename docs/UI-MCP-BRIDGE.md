# Live interface MCP Bridge

The bridge connects a local MCP assistant to a specific open Studio page. It uses
the existing WebMCP tool definitions, argument validation and UI controllers. It
does not require experimental browser WebMCP support or a platform-specific native
automation API. Browser and rebuilt native hosts load the same web module. Native
installers created before this source change retain their earlier bundled code;
this change does not silently replace released archives or installed apps.

## Agent workflow

1. Connect a fresh stdio MCP client to the running updated Studio. Use the existing
   `ag_media_studio_mcp.py --port PORT` configuration, or native instance discovery
   from `docs/NATIVE-DISTRIBUTION.md`. An old MCP process/schema needs reconnecting.
2. `media_catalog(category="ui", query="live")` discovers the selected-window bridge and read-only measurement tool.
3. `ui_windows()` lists connected window IDs, last published state, observation
   age and available tool schemas. There is no implicit active-window selection.
4. `ui_inspect(window_id=...)` reads fresh state and returns `revision`.
5. `ui_execute(window_id=..., revision=..., name="ui_navigate",
   arguments={"page":"transcript"})` uses that exact window and context.
   The response includes the resulting state and the next revision.
6. `ui_measure(window_id=..., revision=..., targets=["image-detail", "image-run"],
   pairs=[{"from":"image-detail","to":"image-run"}])` measures a batch. No browser WebMCP support is needed.
   Omit targets/pairs to discover rendered element IDs.
7. On a context error, inspect again. A timeout or cancellation has an **unknown
   outcome**: inspect instead of automatically replaying the action. Cancellation
   does not roll back an action already executed.

Available commands: measure, inspect, navigate Studio sections, read/set the people-count
preference, read paginated transcript/pauses/facial-expression statistics and
question windows, and seek the current preview. `ui_measure` is also a dedicated read-only ordinary stdio MCP tool;
its measurement output appends only the new revision, never the full UI inspect payload. Their schemas come from
`web/web-mcp.js`. Text and model observations are untrusted user content; expression
estimates do not establish a person's internal emotional state. Heavy transcription,
synthesis, recording, deletion, filesystem commands, arbitrary JavaScript and DOM
selectors are not accepted by this bridge. Existing domain MCP tools remain separate.

## Human controls and lifecycle

Open the existing **MCP** icon. The labelled **MCP Bridge · control this window**
checkbox connects/revokes only this page; its status shows the actual session ID
prefix. New tabs enable the local bridge by default. Disabling is remembered in
that tab's session storage across reloads; storage-disabled browsers keep the
choice only for the current document. Other tabs are independent.

Sessions are ephemeral and are never written to project data or the media graph.
Page hide/disable revokes the channel; a resumed page creates a fresh session.
Server restart loses sessions and pending operations. The page reconnects with
bounded backoff, without replaying an uncertain operation. A frozen/background
browser may not respond; the caller receives a bounded unknown result rather than
a success claim. State revisions fence document/route, speech workspace, report
replacement and an available report revision; they are not database transactions
or a universal counter of every DOM change.

## Companion gateway for an already running Studio

When an existing Studio must stay running, serve the current UI and bridge from
an independent companion process, keeping the existing backend as the only owner
of its database and models:

```sh
.venv/bin/python -B -m voice_studio.app.ui_bridge_gateway --port 8767 --backend-port 8766
```

Open `http://127.0.0.1:8767/` and connect the fresh MCP process with `--port 8767`.
Use the same `--studio-data` and output configuration as the existing workspace
for domain graph tools. The companion itself has no workspace argument, database
or model imports. It owns only its socket and ephemeral bridge. Stopping the
companion does not stop the backend. Existing browser tabs on `8766` are independent;
the companion cannot control them until they connect to its own origin.

The gateway shares the Studio's allowlisted static assets and CSP. It checks its
own exact write origin before relaying domain requests to the fixed loopback
backend, and streams response chunks/ranges without waiting for a full analysis.
This is not another analysis pipeline or a remote proxy. Existing direct browser
capture integrations can continue using their original backend address.
Closing the companion wakes broker callers and drains short HTTP handlers for up
to two seconds. Long upstream requests remain daemon workers rather than keeping
the companion alive; shutdown disconnect diagnostics are suppressed only during
teardown, so they cannot race interpreter-finalization locks on stderr. The
existing backend process and its analysis workers are never signalled.

## Local boundary and resources

The server binds loopback. Page registration/polling requires the existing exact
Host/Origin and `X-Voice-Studio` boundary plus a private per-page channel token.
MCP enumeration/commands reject browser Origin and Sec-Fetch-Site headers and use
a separate local transport marker. The marker is **not authentication**: this is
the same trusted-local-process boundary as the other Studio MCP tools, not a
multi-user or remotely exposed service. The private page token is never listed
for agents. Neither schemas nor tool descriptions grant additional permission.

At most eight live pages, one outstanding command per page, 60-second stale-session
expiry, 15-second call deadline and a 20-second idle long poll. Inputs are limited
to 64 KiB; page results above 48 KiB return an explicit size error (request a
smaller transcript page). There is no screenshot loop, model invocation, durable
queue, command replay or server-side browser evaluation. Server shutdown wakes
long polls before joining its HTTP threads. Requests do not use proxy environment
settings or follow redirects.

## Validation

`python -m unittest tests.test_ui_bridge_gateway tests.test_ui_bridge tests.test_desktop_runtime tests.test_mcp_runtime_info`
and `node --test tests/test_mcp_bridge.mjs tests/test_browser_ml.mjs tests/test_interface_locale.mjs` cover shared
validation, correlation, stale context, expiry, invalid ownership, capacity,
revocation, unknown timeouts, shutdown and browser-versus-MCP HTTP boundaries.
Live verification uses an isolated workspace and the companion's ordinary browser
page with fresh stdio MCP clients; no new recordings or inference are needed. Browser evidence
does not qualify interaction with old native binaries or every renderer.

## Names, categories and compatibility

Canonical discovery uses intent prefixes: `ui_*` for connected-window operations,
`screen_*` for local screenshots/OCR, `audio_*` for stems/cleanup, `speech_*` for
synthesis, `media_transcribe*` for full source analysis, `transcript_*` for saved
editorial material, `media_mods_*` for providers/settings, and `media_graph_*` for
timed media evidence. `browser_ml_capabilities` describes browser execution support.
Media graph names are unchanged. Internal HTTP routes, saved data and model IDs are unchanged.

`media_catalog(include_graph=true)` returns current registrations and a normalized
tool/category graph. Every tool has explicit categories and a scope such as
`live-window`, `local-screenshot` or `timed-media-graph`; overlapping categories
do not make recorded-screen analysis a live UI operation. Provider management is
`models`, not the old implicit `sound` fallback. Unknown extensions are listed
in `uncategorized`. The category IDs sound/face/picture/graph/ui are retained,
with models/runtime/transcript added. Schema and graph payloads are opt-in.

Only canonical names are exposed and accepted. Earlier `voice_*` tool names, page aliases,
legacy naming modes and the old catalogue resource URI are removed. Reconnect existing
clients to obtain the new43-tool inventory. This is an intentional API change; old
calls return unknown-tool errors. Historical reports remain historical records.

## Measurement contract

Use up to32distinct exact DOM IDs and32pairs per call. Pair endpoints are included
automatically; combined IDs must fit32. Coordinates are viewport CSS pixels, not
device pixels. The snapshot records viewport/scroll/font state and elapsed DOM time.
Results contain border boxes, computed padding/border/margin, CSSOM content boxes,
DOM Range text-layout boxes/insets, up to4SVG layout boxes, pair gaps/overlap,
unrounded alignment decisions and rounded0.01px reported differences. A default
1px comparison tolerance is diagnostic, not an AG/WCAG quality certification.

No text, input values, accessible-name strings, screenshots or script results are
exported by the measurement response. Native form text, transformed/clipped text,
fragmented inline boxes, duplicate IDs, hidden/missing elements and bounded-scan
limits have explicit states/reasons. Partial text never has complete-text insets.
Native text/pseudo-content, painted glyph ink, occlusion and true pointer hit
regions are not guessed. Measurement does not scroll, mutate layout, invoke models
or persist snapshots. Client-size content dimensions retain CSSOM integer rounding.
For more than32targets, use several calls; each is a new live snapshot.
