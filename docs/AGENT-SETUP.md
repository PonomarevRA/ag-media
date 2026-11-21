# Connect a local agent to AG Media Studio

[Run Studio](GETTING-STARTED.md) · [Scenario skills](../skills/README.md) · [Tool catalog](MCP-TOOLS.md)

## Source checkout

Create the Python environment and start Studio as described in [First launch](GETTING-STARTED.md). Use absolute local paths in the MCP client. Point the client at the port printed by Studio and at the same workspace supplied to Studio with `--data`.

Example for Codex `config.toml` on macOS or Linux:

```toml
[mcp_servers.ag_media_studio]
command = "/absolute/path/to/ag-media-studio/.venv/bin/python"
args = [
  "/absolute/path/to/ag-media-studio/ag_media_studio_mcp.py",
  "--port", "8766",
  "--studio-data", "/absolute/path/to/workspace",
  "--output", "/absolute/path/to/workspace/mcp-output",
]
```

On Windows the command is `.venv/Scripts/python.exe` inside the checkout. In TOML strings use forward slashes in Windows paths or single-quoted literal strings with backslashes. For a client that accepts JSON, use its local stdio server configuration with the same `command` and `args` fields. Preserve other configured servers. `--graph-root` can select an existing graph directory if it lives outside that workspace. If the MCP process runs model workers, supply their runtime and model environment variables in the client configuration.

The MCP executable communicates on stdin/stdout. Start it through the client. Graph reads and tools that own independent local workers can run without the HTTP server. Studio-backed tools, including the live UI bridge and queued speech generation, require a reachable Studio instance on the selected loopback port.

## Installed desktop app

Open the app for the intended workspace. On macOS and Linux, pass `--print-config` to the native executable. For a custom workspace include `--data /absolute/path/to/workspace`. On Windows, use the bundled Python command. Use the returned config for that instance and repeat discovery after a port change. [Native setup](NATIVE-DISTRIBUTION.md) · [Windows MCP command](WINDOWS-DISTRIBUTION.md#mcp)

## Verify the connection

1. Reconnect the client and ask it to call `media_catalog(include_graph=true)`. Inspect the returned categories, including `ui`, `picture`, `graph`, `sound` and `transcript`. Discover the current schema with `include_schema=true` and a category or query.
2. Open the Studio page, enable its MCP Bridge, and run `ui_windows()`. Pick the window by its origin, route, workspace and published state. If no window appears, check the port and the checkbox in that page.
3. Run `ui_inspect(window_id=...)`. Use its revision for `ui_measure` or `ui_execute`. Inspect again after a stale revision or a timeout whose outcome is unknown.
4. For stored graphs, call `media_graph_list()`. Select the snapshot by source and revision. Empty results are valid for a fresh workspace.

Use [the skills installer](../skills/README.md#install-into-your-project) to place `agent-memory` and the Studio scenario skills in your own project’s discovery directory. Open the target project in a fresh client session and verify that the skills appear and a local Studio tool responds. For repeat analysis, reuse only evidence that still matches the source and current revision.

Studio is a local single-user service. Client and application must share the local filesystem for tools accepting file paths. Keep it on loopback. A screenshot, transcript, or OCR result may contain private material. Send those results only to destinations authorized for that material.
