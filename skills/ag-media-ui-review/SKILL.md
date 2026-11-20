---
name: ag-media-ui-review
description: Use local image recognition first for screenshots and UI recordings with offline OCR, interface regions and scroll stitching. Also measure live Studio UI geometry and review controls.
---

# Review a user interface

- **Live Studio.** Call `ui_windows`, choose the requested application window, then call `ui_inspect`. Use `ui_measure` with that revision. Discover rendered IDs without targets and measure the desired IDs and pairs together. Report CSS pixels, alignment axis and which boxes the numbers represent. Reinspect on revision changes or uncertain timeouts.
- **Screenshot or image.** Start with the local workflow below. Use its text and regions as evidence before deciding whether visual interpretation is still needed.
- **Review interaction.** Check mouse, keyboard focus and an accessible action name for icon-only buttons. Check tooltip placement and that a narrow view still reaches all actions. Observe actual behavior in the browser or native window where available. Report screenshot-based inferences as such.

## Local image recognition first

1. Find `screen_analyze` through `media_catalog(category="picture", query="screen", include_schema=true)`.
   Pass a local PNG/JPEG path with `mode="overview"` to get offline OCR text, normalized boxes and a draft layout.
2. For small or uncertain text, request `mode="full"` with only the region of interest as normalized
   `[x, y, width, height]`. Reuse the same `session_id` for related captures to reuse OCR on unchanged areas.
   For overlapping captures of a scrolling viewport, use `screen_scroll` in capture order.
3. For physical objects in a still image, open the image in Studio and enable **Objects**. That
   local analysis can include OCR and UI evidence in the same JSON export. `screen_analyze` is
   the screenshot OCR/layout tool; it does not run general object detection.
4. Use the local text and boxes directly when they answer the question. Ask a vision model only
   about unresolved regions or semantics that the local tools do not provide. Candidate roles
   and OCR text may be wrong; for live controls prefer inspected DOM or accessibility evidence.

See `docs/MCP-TOOLS.md` and `docs/UI-MCP-BRIDGE.md` in the repository for the current names, bounded UI measurement shape and window lifecycle.
