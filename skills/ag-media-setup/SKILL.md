---
name: ag-media-setup
description: Install or connect local AG Media Studio from source or a desktop app, find its active port and workspace, and verify the MCP tools in a fresh client.
---

# Connect AG Media Studio

1. Identify the local checkout or installed desktop instance and the requested workspace. Preserve the running app and its data if the request concerns a different workspace. Use the corresponding instructions in `docs/GETTING-STARTED.md` and `docs/AGENT-SETUP.md` in the repository.
2. For source, isolate dependencies in a Python virtual environment. Use a new `--data` directory and a free port for a demo check. For a native app, use the platform command in `docs/AGENT-SETUP.md` to obtain the open instance's configuration.
3. Point the stdio client at `ag_media_studio_mcp.py` and pass the Studio port and workspace for a source setup. Use the native config for an installed app. Preserve existing client entries. Show the exact config if client settings are unavailable.
4. Reconnect and call `media_catalog` to verify fresh discovery. With MCP Bridge enabled in the open page, call `ui_windows` and `ui_inspect` on the selected window. Report tool reachability and selected workspace separately from model readiness.
5. Install an optional engine only for the requested workflow. Inspect its local readiness, run a small authorized input, and report the actual result. The demo's tones verify the flow through the UI.
