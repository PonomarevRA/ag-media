# Third-party source and resource inventory

This source export preserves `THIRD-PARTY.md` as an existing project record. Its
component and license statements were recorded on the dates written there; they
are not a newly verified legal conclusion. This file and
`packaging/resource-license-inventory.json` describe export boundaries and work
remaining before a binary release.

Both private and public source archives omit model weights, engine executables,
installed Python packages, downloaded source repositories, caches, user media and
training corpora. Imports, download scripts and integration code can still name
optional resources. Their presence does not mean those resources are bundled or
that their redistribution is authorized.

The native app includes MIT-licensed inference code from [Light-ASD at ed38c23](https://github.com/Junhua-Liao/Light-ASD/tree/ed38c232de5efe0261dbd68627c0ade7cdfe14eb) and MIT-licensed MFCC code from [python_speech_features 0.6](https://pypi.org/project/python_speech_features/0.6/). Their original notices are in [Light-ASD LICENSE](voice_studio/vendor/light_asd/LICENSE.md) and [python_speech_features LICENSE](voice_studio/vendor/speech_features/LICENSE.md). Model checkpoints remain separate downloads.

Original project code is licensed under MIT in `LICENSE`, as selected by the
project owner. This does not replace or remove any upstream license, copyright,
attribution, NOTICE or other third-party rights. Any third-party source added to the allowlist
must carry its original license and applicable notices. No AG Design checkout,
package or running local AG service is part of the source-export dependency
closure. If reusable AG components are copied later, record their exact source
revision, license and notices before including them.

The minimal declared Python dependencies are MLflow Skinny, SQLAlchemy, Alembic,
MessagePack, the MCP Python SDK, FastAPI and Uvicorn. Sherpa-ONNX is an
optional processing dependency outside the core requirements. Their installed versions,
transitive dependencies and license files depend on the actual build environment.
A source requirements range is not a reproducible binary inventory. The native
builder must record the resolved package versions and include their notices in
its build output. System frameworks remain supplied by the operating system.

The macOS Apple Silicon binary includes Sherpa-ONNX 1.13.8 and its ONNX Runtime
1.28.2 library. Sherpa-ONNX carries its Apache-2.0 license in the installed package
metadata. The bundle includes the [ONNX Runtime license](voice_studio/vendor/onnxruntime/LICENSE.md)
and [third-party notices](voice_studio/vendor/onnxruntime/ThirdPartyNotices.md)
from the [upstream v1.28.2 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.2).
The bundled Certifi 2026.10.23 source files and license are available in the
[upstream package](https://pypi.org/project/certifi/2026.10.23/).

The existing resource record flags these release questions:

- The exact FFmpeg/OpenCV binary build and codec closure need a separate audit.
  No FFmpeg binary is included by source export.
- Light-ASD weights, Sherpa-exported diarization files, HDemucs weights, and the
  external RNNoise `cb.rnnn` file have unresolved redistribution details in the
  existing record. They remain excluded.
- F5-TTS variants, XTTS-v2 and Fish Speech have model-specific restrictions or
  commercial permission questions in the existing record. Their weights remain
  excluded; a code license is not a weight license.
- Chatterbox's converted tokenizer provenance and exact artifact conditions need
  confirmation in the existing record. It remains excluded.
- Other resources also remain excluded even when the existing record reports a
  permissive license. Recheck the exact revision, artifact hash, license text and
  NOTICE before a future release includes any downloaded resource.

Use the official source links in `THIRD-PARTY.md` for that review. Preserve
uncertainty where source-code licensing and artifact redistribution differ.
