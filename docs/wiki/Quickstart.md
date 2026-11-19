# Run and product identity

[Русский](Quickstart-ru.md) · [ag-media-studio](../../README.md)

## Local installation

Use Python 3.12 or later. FFmpeg and the selected engines are needed for media processing. Run the commands from the source root and open `http://127.0.0.1:8767/`. Demo mode uses test tones.

```text
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python ag_media_studio.py --demo --port 8767
```

## Windows PowerShell

```text
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe ag_media_studio.py --demo --port 8767
```

## Identity and compatibility

| Surface | Behaviour |
| --- | --- |
| Product / MCP server | `ag-media-studio` |
| New entry points | `ag_media_studio.py`, `ag_media_studio_mcp.py` |
| Project directory | Choose the directory name when cloning or extracting the source |
| Existing entry points | `server.py`, `voice_mcp.py` remain valid |
| Python imports | The `voice_studio` namespace remains compatible |
| Assets and settings | Existing `Voice Studio` folders, `VOICE_*` variables and browser keys remain valid |

For production work, omit `--demo` and select a separate workspace with `--data`. Read `docs/AGENT-SETUP.md` and `skills/README.md` in the repository to connect agents.

## First journey

- Open Recording and transcription and select a local source.
- Select analysis layers and inspect individual stage status.
- Check text, speaker and overlay at the same source timestamp.
- Open a profile or timeline item for details and use the existing export actions.

- [Models and storage](Models-and-storage.md)
- [Interface guide](Interface.md)
