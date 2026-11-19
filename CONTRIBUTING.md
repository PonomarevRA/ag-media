# Contributing

Use an isolated virtual environment and the source setup in `README.md`.
Keep user media, local configuration, model files and generated artifacts outside
the source tree. Add synthetic fixtures instead of real recordings or account
information. Preserve upstream license and attribution files when changing any
third-party component.

Before proposing a change, run its focused Python or JavaScript tests and the
source export checks in `PUBLISHING.md`. Include the actual checks and known
limitations in the change description. Publication and license selection remain
separate owner decisions.

## Compare MCP performance

Create a graph in an isolated workspace from synthetic media. Save its snapshot
ID in a corpus JSON array of objects, each with a `snapshotId` field. Run the
same corpus and repeat count before and after a change:

```sh
.venv/bin/python scripts/check_media_mcp.py \
  --studio-data /absolute/test-workspace \
  --graph-root /absolute/test-workspace/media-graphs \
  --corpus /absolute/corpus.json --repeats 10 --output /absolute/result.json
```

Compare findings, speech fingerprints and response sizes alongside latency.
The report separates MCP startup from request timings. Its sampled RSS sums
process memory across the client and children. This check reads existing graphs
and original media fragments. It measures graph and MCP operations.
