# MCP tools and UI measurements

Use [Agent setup](AGENT-SETUP.md) to connect a local stdio client. The current catalogue is generated from the running MCP server's registrations.

## Discovery

Use `media_catalog(category="ui", query="measure", include_schema=true)` to get the measurement schema. Use `media_catalog(include_graph=true)` for the tool and category graph. The `scope` field distinguishes live UI, screenshot, timed media, model management and saved transcript operations. Tools without a category appear in `uncategorized`.

## Local image recognition

Use `screen_analyze` with a local PNG/JPEG path and `mode="overview"` for offline OCR and
candidate UI regions. Use `full` on the specific area that needs detail. For general object
recognition in still images, use Studio's image analysis with **Objects** enabled. See the
[local-first skill workflow](../skills/ag-media-ui-review/SKILL.md#local-image-recognition-first)
for scrolling captures and when to ask a vision model to interpret remaining details.

## Speech and screen evidence

`media_graph_interval` returns `screenEvidence` on timed speech when visible text
matches a nearby utterance or a reviewable ASR draft. Each match has the UI time,
text, normalized source box and node ID. `actionCandidate` requires both spoken
click wording and a measured screen state change. It remains a review cue.

Call `media_graph_frame(snapshot_id=..., node_id=..., crop=true)` for the evidence
PNG. This crops the checked original at the node's time before scaling. A detailed
`screen_region` can also provide `regionKind`, `previousStateMs` and `lastObservedMs`.

## Voice quality and restoration

Start `audio_voice_quality` with an absolute path to a dialogue stem or media file.
Pass verified speaking ranges as `speech_intervals_ms: [[start, end], …]` to estimate
steady noise in the gaps. Poll `audio_job` for levels in dBFS, relative band levels,
and `recommendedModes`. Without speaking ranges, the tool treats the selected interval
as dialogue and leaves the noise estimate empty.

Call `audio_voice_restore` with the same range and any combination of `clarity`,
`quiet-speech`, and `denoise`. Leaving `modes` out uses the recommendation.
The job returns aligned reference and processed WAV paths. An explicit denoise
request uses a conservative preset when no stable noise was measured.

## Ordinary MCP measurements

1. `ui_windows()` lists connected native/browser pages.
2. `ui_inspect(window_id="...")` gives the current revision.
3. Call `ui_measure` with that window/revision and a batch:

```json
{"window_id":"...","revision":"...","targets":["image-detail","image-run"],"pairs":[{"from":"image-detail","to":"image-run"}],"tolerance_px":1}
```

Omit targets and pairs to discover rendered IDs with limit/offset pagination. One request can address up to 32 distinct IDs and 32 pairs. The response reports viewport boxes in CSS pixels, including padding, borders, margins, text bounds, icon boxes, gaps, overlap and alignment. Partial, hidden and platform-rendered measurements are explicitly marked. Text bounds measure DOM layout. The same operation is available through WebMCP and ordinary stdio MCP.

## Canonical-only API

Refresh the MCP connection after updating Studio to load the current names and schemas. Earlier `voice_*` tool calls return an unknown-tool error. Call `media_catalog` to find the current equivalent by purpose and category.

The catalogue includes `sound`, `face`, `picture`, `graph`, `ui`, `models`, `runtime` and `transcript`. Each tool has one ID and may link to multiple categories.
