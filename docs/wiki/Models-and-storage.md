# Models, dependencies and storage

[Русский](Models-and-storage-ru.md) · [ag-media-studio](../../README.md)

## Model responsibilities

| Layer | Executor / limitation |
| --- | --- |
| Text | Whisper through a local backend; speech and UI locale are independent |
| Voice groups | Sherpa, with separate identity and short-turn review modules |
| Faces / identity | YuNet + SFace and conservative profile association |
| Visible speaking | Light-ASD; the visible person may be silent while someone else speaks |
| Objects / text | NanoDet and OCR backends; observed labels may vary |
| Body / gaze / clothing | Requested local vision/MediaPipe workers |
| Facial expressions | Visible-expression classifier, not psychological inference |
| Audio / intonation | Platform backends including Apple SoundAnalysis and Zipformer |
| Synthesis | Qwen3-TTS and Chatterbox in the primary mode; previous engines are research options |

This describes responsibilities, not guaranteed installation. Inspect current readiness, paths, backend and model terms in configuration and `THIRD-PARTY.md`. Fish may run as a separate local service. Initial setup can download weights; building the wiki installs no models.

## Data lifetimes

| Data | Location / purpose |
| --- | --- |
| Recordings and graphs | `data/studio` or explicit `--data`; durable user history |
| Exports and experiments | `data/exports`, `data/prosody` and experiment-specific MLflow storage |
| Models / dependencies | `voice_paths.py` paths; existing Voice Studio asset directories remain compatible |
| Temporary cache | System temp / `VOICE_TEMP_DIR`, isolated by workspace |
| Python environments | `.venv` and isolated `.engines/` environments |
| Wiki | Text, source paths/hashes and schematic SVG only; no WAV, MP4 or face photos |

Use Storage to review exact cleanup candidates. Age alone does not make a source, unique export or installed weight disposable. The product rename does not move or delete these directories.

- [Run and compatibility](Quickstart.md)
