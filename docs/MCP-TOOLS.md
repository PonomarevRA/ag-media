# MCP tools and UI measurements

Use [Agent setup](AGENT-SETUP.md) to connect a local stdio client. The current catalogue is generated from the running MCP server's registrations.

## Discovery

Use `media_catalog(category="ui", query="measure", include_schema=true)` to get the measurement schema. Use `media_catalog(include_graph=true)` for the tool and category graph. The `scope` field distinguishes live UI, screenshot, timed media, model management and saved transcript operations. Tools without a category appear in `uncategorized`.

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
