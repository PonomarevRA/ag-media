# AG Media Studio agent skills

Connect the local MCP server using [Agent setup](../docs/AGENT-SETUP.md).
Install the skills **in the project where the agent will do the work**. That
project can be separate from Studio and does not need a copy of its source.

| Skill | Request it handles |
| --- | --- |
| [agent-memory](agent-memory/SKILL.md) | Reuse verified analysis results and keep bounded project knowledge. |
| [ag-media-setup](ag-media-setup/SKILL.md) | Install Studio, connect a fresh MCP client and verify the selected workspace. |
| [ag-media-interval-review](ag-media-interval-review/SKILL.md) | Read a timed media interval with transcript, visual evidence and uncertain findings. |
| [ag-media-ui-review](ag-media-ui-review/SKILL.md) | Start with offline OCR and UI layout for screenshots, then inspect uncertain regions; also measure live UI. |
| [ag-media-dialogue-audit](ag-media-dialogue-audit/SKILL.md) | Review speaker boundaries, overlapping voices and missing dialogue. |

## Install into your project

Run from the Studio source checkout, or use the desktop package paths below.
Replace the target with the directory that you open in your agent client:

```sh
python3 skills/install.py --project "/path/to/my/project" --client codex --check
python3 skills/install.py --project "/path/to/my/project" --client codex
```

Use `--client claude`, `--client cursor` or `--client all` for other clients. On Windows,
use `py -3` or the embedded Python indicated below. The installer creates only
the selected project skill folders, initializes empty registries when missing
and preserves existing memory, customizations and unrelated skills. An
existing `agent-memory` from another distribution or with local edits is
reported and kept. Existing scenario skills are also preserved. In a project
with a custom setup, point the agent to the reuse guide below if its installed
scenario does not link to `agent-memory` yet. Client global settings are never changed.

| Package | Python and script |
| --- | --- |
| macOS | `Contents/Resources/runtime/bin/python3` and `Contents/Resources/studio/skills/install.py` inside the app |
| Windows | `runtime/python/python.exe` and `source/skills/install.py` next to `AgMediaStudio.exe` |
| Linux DEB | `python3 /opt/ag-media-studio/studio/skills/install.py` |

Open the target project in a fresh client session after installation. Ask for
the available skills, then repeat an analysis request with the same source and
interval. The scenario skill calls for a bounded memory lookup before a new
analysis. See [reuse rules](agent-memory/references/ag-media-analysis.md) for
validity checks; the local media analysis index is in `.ag-media-memory/`,
separate from shareable project knowledge under `docs/agent-memory/`.
MCP and models are needed for new analysis, but installing and reading project
memory does not need a service or model.

For screenshots, start with the [local image recognition workflow](ag-media-ui-review/SKILL.md#local-image-recognition-first).
It gives an agent text and layout evidence before it decides whether to use a paid vision model.

Example prompts after reconnecting the client:

- `$agent-memory` Check if we already reviewed these 12–24 seconds and verify the result against the current snapshot.
- `$ag-media-setup` Verify the MCP configuration for the open instance and inspect a connected window.
- `$ag-media-interval-review` Show what was said and visible from 12 to 24 seconds. Keep source timecodes.
- `$ag-media-ui-review` Check the gap and baseline alignment between the image detail selector and run button.
- `$ag-media-dialogue-audit` Find mixed speakers or speech with missing text and separate evidence from proposed changes.

When reading the folders directly, open the guides linked by the skill in this repository. When using an installed copy, use the same repository's guide paths from its root.
