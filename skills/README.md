# AG Media Studio agent skills

Connect the local MCP server using [Agent setup](../docs/AGENT-SETUP.md). The skills are portable folders. Install them into the checkout's `.agents/skills/` for Codex or the corresponding project skills directory for your client. Copy a whole folder and preserve other client skills. Reload the client to refresh discovery.

| Skill | Request it handles |
| --- | --- |
| [ag-media-setup](ag-media-setup/SKILL.md) | Install Studio, connect a fresh MCP client and verify the selected workspace. |
| [ag-media-interval-review](ag-media-interval-review/SKILL.md) | Read a timed media interval with transcript, visual evidence and uncertain findings. |
| [ag-media-ui-review](ag-media-ui-review/SKILL.md) | Start with offline OCR and UI layout for screenshots, then inspect uncertain regions; also measure live UI. |
| [ag-media-dialogue-audit](ag-media-dialogue-audit/SKILL.md) | Review speaker boundaries, overlapping voices and missing dialogue. |

Run the following installation from the **repository root**. The command copies only missing skill folders. To update a skill, compare it with the corresponding source folder before replacing it.

**macOS and Linux**

```sh
mkdir -p .agents/skills
for skill in skills/ag-media-*/; do
  name=$(basename "$skill")
  if [ ! -e ".agents/skills/$name" ] && [ ! -L ".agents/skills/$name" ]; then
    cp -R "$skill" ".agents/skills/$name"
  fi
done
```

**Windows PowerShell**

```powershell
New-Item -ItemType Directory -Force .agents/skills | Out-Null
Get-ChildItem -Path skills -Directory -Filter 'ag-media-*' | ForEach-Object {
  $destination = Join-Path '.agents/skills' $_.Name
  if (-not (Test-Path -LiteralPath $destination)) {
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse
  }
}
```

For screenshots, start with the [local image recognition workflow](ag-media-ui-review/SKILL.md#local-image-recognition-first).
It gives an agent text and layout evidence before it decides whether to use a paid vision model.

Example prompts after reconnecting the client:

- `$ag-media-setup` Verify the MCP configuration for the open instance and inspect a connected window.
- `$ag-media-interval-review` Show what was said and visible from 12 to 24 seconds. Keep source timecodes.
- `$ag-media-ui-review` Check the gap and baseline alignment between the image detail selector and run button.
- `$ag-media-dialogue-audit` Find mixed speakers or speech with missing text and separate evidence from proposed changes.

When reading the folders directly, open the guides linked by the skill in this repository. When using an installed copy, use the same repository's guide paths from its root.
