# AG Media Studio

A local media workspace for recording, transcription, speaker profiles, video
analysis, editing and optional voice synthesis. The Python package keeps its
compatible `voice_studio` name and `VOICE_` environment variables.

## Install and run

To prepare the transcription workflow and open the Studio, run this command
from the extracted or cloned repository with Python 3.12 or later:

```sh
python -m voice_studio.app.first_run
```

On macOS the command can obtain missing tools through Homebrew. On Windows and
Linux it uses FFmpeg and whisper.cpp already available in PATH. The command
downloads the verified Whisper Base checkpoint and reuses it on later runs.
[Setup and platform prerequisites](docs/GETTING-STARTED.md)

The same guide covers a manual installation and the demonstration mode.

Optional media tools and model workers require separately provisioned engines
and model files. This source package does not include weights, FFmpeg, downloaded
engines, training datasets or recorded media. Features that depend on an absent
engine or model require that resource before processing. Review the model and
binary conditions in `THIRD-PARTY.md` before provisioning or redistribution.

## Native application

See [native distribution](docs/NATIVE-DISTRIBUTION.md) for packaging, build
prerequisites and platform status. A native package must own its local server
process and keep durable state outside its application bundle. The application
source and its platform CI templates do not establish that an installer for every
platform has been built, signed or qualified. Check the actual release artifacts
and qualification evidence for macOS, Windows and Linux separately.

## MCP and agent skills

[Agent setup](docs/AGENT-SETUP.md) covers local stdio from source, desktop MCP,
the UI bridge and smoke checks. [Agent skills](skills/README.md) contains
installation instructions and workflows for setup, interval review, UI review
and dialogue audit.

## Data and development

Keep user media, model weights and generated state outside the source repository.
`voice_paths.py` defines platform data locations and operator overrides, including
`VOICE_MODELS_DIR`, `VOICE_DEPENDENCIES_DIR` and `VOICE_TEMP_DIR`. Do not commit
local environment files, recordings, transcripts, credentials or model weights.

The [architecture guide](docs/ARCHITECTURE.md) covers package boundaries and launchers.
The [local documentation](docs/wiki/Home.md) covers the interface, models and storage. Run focused tests for the capability being changed.
For the source exporter safety contracts:

```sh
python3 -B -m unittest discover -s tests -p test_repository_export.py
```

Some integration and training checks require separately provisioned resources.
See [contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md).

## Repository preparation and license

[Publishing](PUBLISHING.md) describes preparing scanned source archives for
private and public repositories. Both profiles include installation guides,
agent skills and MIT-licensed project code. The private profile adds two internal
design guides.

Original project code is licensed under [MIT](LICENSE). Third-party code,
packages, models and operating-system frameworks retain their own terms and
notices. See [third-party notices](THIRD_PARTY_NOTICES.md), the preserved
[resource record](THIRD-PARTY.md), and
[resource inventory](packaging/resource-license-inventory.json).
