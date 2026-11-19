---
name: ag-media-interval-review
description: Inspect a specific period in an AG Media Studio graph with timed dialogue, shots, objects, UI, a frame grid and uncertain evidence for editing or reporting.
---

# Review a media interval

1. Use `media_catalog(category="graph", query="interval", include_schema=true)` and resolve the exact source with `media_graph_list` and `media_graph_overview`. Pick one `snapshot_id` for related queries. If building is requested, pass the local file and use progressive snapshots for preliminary review.
2. Call `media_graph_interval` with source milliseconds and a half-open interval of at most 60 seconds. Begin with `frames=false` when the question is about dialogue or timestamps. Request the frame grid or one `media_graph_frame` when visual evidence matters. Ask for `wire="expanded"` if the client needs plain JSON.
3. Read coverage, omitted counts, and uncertainties before drawing conclusions. Quote ASR/OCR as observations. Check the actual frame time and regard draft speaker and object labels as hypotheses. Missing layers in an intermediate snapshot may be published later.
4. Retrieve the exact node or next timeline page for a consequential detail. Return concise findings with source timecodes, the snapshot ID and any unresolved or unobserved interval.

See `MEDIA-GRAPH.md` in the repository for interval wire format, coverage and frame timing.
