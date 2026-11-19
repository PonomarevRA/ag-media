---
name: ag-media-ui-review
description: Measure live Studio UI geometry and verify compact controls, or analyze a supplied screenshot or scroll sequence with Studio's local tools.
---

# Review a user interface

- **Live Studio.** Call `ui_windows`, choose the requested application window, then call `ui_inspect`. Use `ui_measure` with that revision. Discover rendered IDs without targets and measure the desired IDs and pairs together. Report CSS pixels, alignment axis and which boxes the numbers represent. Reinspect on revision changes or uncertain timeouts.
- **Image supplied by the user.** Discover `screen_analyze` with `media_catalog(category="picture", query="screen", include_schema=true)`. Request `overview` first, then `full` for the region where detail is needed. For sequential screenshots of the same scrolling viewport, discover and use `screen_scroll` in capture order. OCR and geometry provide candidate labels and controls.
- **Review interaction.** Check mouse, keyboard focus and an accessible action name for icon-only buttons. Check tooltip placement and that a narrow view still reaches all actions. Observe actual behavior in the browser or native window where available. Report screenshot-based inferences as such.

See `docs/MCP-TOOLS.md` and `docs/UI-MCP-BRIDGE.md` in the repository for the current names, bounded UI measurement shape and window lifecycle.
