# Карта модулей и исходников

[English](Modules.md) · [ag-media-studio](../../README.md)

Ниже названа ответственность пакета и перечислены все его исполняемые Python-модули и браузерные JS-модули. Python-описания взяты из module docstring; браузерные обязанности явно описаны и проверены по исходникам. Это карта назначения, а не доказательство полного runtime-контракта. `__init__.py` и сгенерированные зависимости не включены.

Для поиска реализации начинай с функционального пакета и проверяй указанные файлы через Code Atlas и исходник. Машиночитаемые пути и SHA-256 доступны в `docs/wiki/module-index.json` и по `/wiki/module-index.json`. Изменение исходников требует пересборки wiki; `--check` обнаруживает устаревший индекс.

## Приложение и HTTP

Входные маршруты, сохранение материала, история и локальная выдача файлов. Не содержит модели распознавания.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/app/desktop_config.py` | Explicit native-instance discovery and MCP launch configuration, with no development-port fallback. |
| `voice_studio/app/desktop_linux.py` | Native GTK3/WebKit2 desktop host; packaged Linux uses explicit distro GUI dependencies. |
| `voice_studio/app/desktop_runtime.py` | Owned loopback service for the native desktop hosts; never attaches to another instance. |
| `voice_studio/app/desktop_updates.py` | Read stable native release metadata on demand and report matching packages. |
| `voice_studio/app/dialogue_audit_service.py` | Validate and serialize source-bound dialogue audits for UI and local agents. |
| `voice_studio/app/first_run.py` | Prepare local transcription from a checkout, then open the Studio. |
| `voice_studio/app/http_transport.py` | HTTP transport for the studio's loopback server. |
| `voice_studio/app/image_analysis.py` | Adapt a bounded browser image upload to the same local vision worker used by screenshot MCP. |
| `voice_studio/app/live_capture.py` | Bounded local analysis for explicitly selected browser tab fragments. |
| `voice_studio/app/media_pipeline.py` | Reuse the graph's bounded source pipeline for long uploads instead of truncating audio. |
| `voice_studio/app/scroll_previews.py` | Keep the latest successful set of scroll mosaics in the Studio's private cache. |
| `voice_studio/app/server.py` | Serve the single-user ag-media-studio on loopback. |
| `voice_studio/app/static_assets.py` | Shared UI asset catalogue and response policy, with no model or database imports. |
| `voice_studio/app/studio.py` | Persist voice samples, queue model runs, and assemble approved audio. |
| `voice_studio/app/studio_storage.py` | Inspect this Studio workspace and expire its explicitly registered disposable caches. |
| `voice_studio/app/transcript_editor.py` | Store timed transcript material and revision-checked edits in the local studio. |
| `voice_studio/app/ui_bridge.py` | Bounded, in-memory rendezvous between local MCP and an explicitly connected page. |
| `voice_studio/app/ui_bridge_gateway.py` | Serve the updated UI/bridge while relaying domain requests to an untouched local Studio. |
| `voice_studio/app/wiki_assets.py` | Resolve only versioned, generated documentation assets, never workspace data. |

## Оркестрация и ресурсы

Общие стадии, бюджеты потоков, блокировки и время жизни кэша. Эти правила переиспользуют обработчики.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/core/analysis_stages.py` | Report real analysis boundaries to optionally subscribed media clients. |
| `voice_studio/core/mcp_lanes.py` | Bound blocking MCP operations without blocking the server event loop or other media lanes. |
| `voice_studio/core/media_event_bus.py` | Bounded job-local scheduling for graph publication and object enrichment. |
| `voice_studio/core/media_limits.py` | Shared source limits; individual native inference windows remain bounded separately. |
| `voice_studio/core/media_resources.py` | Plan CPU and memory for concurrent local media inference. |
| `voice_studio/core/optional_runtime.py` | Find optional inference wheels installed outside the application bundle. |
| `voice_studio/core/process_usage.py` | Collect process timings without requiring Unix-only resource counters. |
| `voice_studio/core/storage_cache.py` | Coordinate regenerable storage with its readers across Studio and MCP processes. |
| `voice_studio/core/tool_names.py` | The finite page-operation allowlist; no legacy aliases or runtime imports. |
| `voice_studio/core/voice_registry_lock.py` | Serialize writers to one source's private voice registry without locking UI parsing. |
| `voice_studio/core/worker_process.py` | Build child commands for modules owned by the installed Studio package. |
| `voice_studio/core/workspace_lock.py` | Cross-platform, process-lifetime ownership for one writable Studio workspace. |

## Речь и спикеры

Распознавание, языки ролика, голосовые группы, короткие ответы, сопоставление и редакторские исправления.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/speech/asr_context_graph.py` | Suggest Russian ASR corrections using typed local evidence links. |
| `voice_studio/speech/asr_recovery.py` | Retry structurally unreliable clip transcripts on their original PCM clock. |
| `voice_studio/speech/dialogue_audit.py` | Find reviewable dialogue defects without editing ASR or inferring identities. |
| `voice_studio/speech/dialogue_audit_runner.py` | Run a local, cancellable dialogue audit with content-keyed atomic outputs. |
| `voice_studio/speech/embedding_protocol.py` | Bounded wire values shared by the native embedding parent and child. |
| `voice_studio/speech/embedding_worker.py` | Own one lazy native speaker process, bounded serial admission and its deadline. |
| `voice_studio/speech/embedding_worker_server.py` | Native-only JSONL service; model resources are resolved exclusively by its parent. |
| `voice_studio/speech/media_transcript.py` | Extract local media audio and return segment timestamps and SRT subtitles. |
| `voice_studio/speech/native_turn_bounds.py` | Adapt bounded native EOF rounding for dialogue audits without changing raw facts. |
| `voice_studio/speech/person_mentions.py` | Find text positions with the local ag-privacy model and leave cue text intact. |
| `voice_studio/speech/recognition.py` | Transcribe a local recording with multilingual Whisper. |
| `voice_studio/speech/short_reply_review.py` | Recover a missed short speaker turn from independent face and voice evidence. |
| `voice_studio/speech/speaker_context.py` | Reapply explicit speaker anchors to exact cues while retaining competing evidence. |
| `voice_studio/speech/speaker_diarization.py` | Run local speaker segmentation and label timed Whisper words. |
| `voice_studio/speech/speaker_identity.py` | Replay confirmed same-person groups on an exact source using timed evidence. |
| `voice_studio/speech/speaker_photos.py` | Sample visible faces from local video using YuNet in the vision runtime. |
| `voice_studio/speech/speaker_pipeline.py` | Connect source-scoped voice profiles and non-destructive dialogue evidence to stages. |
| `voice_studio/speech/speaker_profiles.py` | Catalog reviewed speaker references for repeated analysis of an exact upload. |
| `voice_studio/speech/speaker_review.py` | Review provisional Sherpa clusters against local voice and active-face references. |
| `voice_studio/speech/speech_overlap.py` | Represent simultaneous audio or visual speech without assigning mixed words. |
| `voice_studio/speech/transcription_languages.py` | Keep clip recognition languages separate from studio languages and live capture. |
| `voice_studio/speech/voice_profile_evidence.py` | Validate speaking evidence separately from face presence and acoustic identity. |
| `voice_studio/speech/voice_profile_public.py` | Whitelist vector-free profile/evidence facts at every public JSON boundary. |
| `voice_studio/speech/voice_profiles.py` | Bounded, source-scoped acoustic identities with explicit speaking-video priority. |
| `voice_studio/speech/voiced_speaker_review.py` | Reconcile one clean Sherpa turn with a confirmed talker in a wider subtitle. |
| `voice_studio/speech/whisper_assets.py` | Resolve and provision the multilingual Whisper Base runtime assets. |

## Анализ и обработка звука

Шумоподавление, EQ, паузы, фоновые события, музыка и измерение интонации.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/audio/agent_audio.py` | Export aligned dialogue, music and effects for local MCP clients. |
| `voice_studio/audio/audio_processing.py` | Prepare a separate WAV preview with local noise reduction and broad EQ. |
| `voice_studio/audio/audio_scene.py` | Classify independent music and sound cues on prepared audio and attach pitch evidence. |
| `voice_studio/audio/dialogue_pauses.py` | Measure edit-review intervals on the same millisecond clock as dialogue cues. |
| `voice_studio/audio/music_sample.py` | Prepare a short original clip and the signal removed by local denoising. |
| `voice_studio/audio/pauses.py` | Measure quiet spans and suggest punctuation at recognized boundaries. |
| `voice_studio/audio/pitch_vdsp.py` | Use Accelerate to compute the same autocorrelation numerators on macOS. |
| `voice_studio/audio/speech_intonation.py` | Estimate phrase-level pitch contours from the original speech PCM. |
| `voice_studio/audio/zipformer_audio.py` | Run the local Sherpa-ONNX audio tagger on overlapping PCM windows. |

## Кадры и визуальные наблюдения

Планы, лица, активное говорение, фигуры, взгляд, одежда, предметы, OCR, интерфейсы и технический экспорт.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/vision/activity_worker_session.py` | Reuse the local Light-ASD model across bounded, synchronous requests. |
| `voice_studio/vision/adaptive_faces.py` | Reduce YuNet calls at 25 fps by refining unstable six-frame intervals. |
| `voice_studio/vision/camera_motion.py` | Estimate shot-local apparent camera motion from sparse background matches. |
| `voice_studio/vision/clothing_match.py` | Use segmented clothing as corroboration for borderline within-video face matches. |
| `voice_studio/vision/clothing_review.py` | Sample a small set of outfit references between the face and audio stages. |
| `voice_studio/vision/face_appearance.py` | Summarize sampled face appearance without using demographics to match identities. |
| `voice_studio/vision/face_reid.py` | Match separate visual tracks within one upload using bounded face signatures. |
| `voice_studio/vision/face_speakers.py` | Bind the numbered face boxes to transcript speakers using aligned evidence. |
| `voice_studio/vision/facial_expression_model.py` | Classify visible facial expression using the pinned OpenCV Zoo model locally. |
| `voice_studio/vision/facial_expressions.py` | Project measured expression samples onto source time without inferring mental states. |
| `voice_studio/vision/head_pose.py` | Estimate screen-relative head orientation from a tracked Face Landmarker crop. |
| `voice_studio/vision/image_input.py` | Shared bounds and decoder validation for browser image uploads and MCP screenshots. |
| `voice_studio/vision/image_objects.py` | Use the scene detector for one still image, without creating video tracks or speaker identities. |
| `voice_studio/vision/image_status.py` | Reduce screenshot OCR coverage without hiding failed crops behind successful control-label reads. |
| `voice_studio/vision/object_appearance.py` | Measure object appearance on the scene worker's decoded frames. |
| `voice_studio/vision/object_reconciliation.py` | Run optional local-part crop measurement in the existing vision environment. |
| `voice_studio/vision/object_reconciliation_parts.py` | Bounded optional local-part measurements on original video object crops. |
| `voice_studio/vision/person_face_pilot.py` | Sample visible people and retry YuNet on source-resolution head crops. |
| `voice_studio/vision/person_layers.py` | Select and execute bounded body, gaze or clothing-mask tasks. |
| `voice_studio/vision/person_motion.py` | Estimate shot-local person motion and relative depth from existing geometry. |
| `voice_studio/vision/scene_objects.py` | Decode COCO NanoDet-Plus predictions using the existing OpenCV DNN runtime. |
| `voice_studio/vision/scene_ocr.py` | Run local bilingual OCR and project sector coordinates back onto the video. |
| `voice_studio/vision/scene_sampling.py` | Allocate native object and OCR samples on the source playback clock. |
| `voice_studio/vision/scene_transcript.py` | Reconcile a near-miss surname with repeated title-card OCR during the same cue. |
| `voice_studio/vision/screen_analysis.py` | Select screen crops and run local OCR without increasing ordinary scene sampling. |
| `voice_studio/vision/screen_layout.py` | Recover visible layout evidence from an already decoded frame and its OCR boxes. |
| `voice_studio/vision/screen_scroll.py` | Register vertical screen motion and keep each OCR reading on the source clock. |
| `voice_studio/vision/screen_session.py` | Bound private screenshot results by both session age and generation count. |
| `voice_studio/vision/screenshot_structure.py` | Turn OCR lines into conservative list and title-bar evidence for a still image. |
| `voice_studio/vision/shot_moodbook.py` | Extract a compact shot palette and visible-person statistics from sampled video pixels. |
| `voice_studio/vision/small_object_pass.py` | Add detail from production anchor crops without replacing the baseline observations. |
| `voice_studio/vision/static_face_evidence.py` | Conservative static/frozen face-crop veto for unverified visual inferences. |
| `voice_studio/vision/tracking.py` | Record experiment metadata in a local MLflow SQLite store. |
| `voice_studio/vision/ui_activity.py` | Track sampled visual anchors without inferring speech, people, or identity. |
| `voice_studio/vision/ui_static_regions.py` | Cheap, color-independent geometry proposals from an already decoded BGR frame. |
| `voice_studio/vision/video_activity_chunks.py` | Accumulate independent Light-ASD windows on the video playback clock. |
| `voice_studio/vision/video_colors.py` | Measure representative colors and caption contrast from source video pixels. |
| `voice_studio/vision/video_export.py` | Bake timed face boxes, shot numbers and captions into a local MP4. |
| `voice_studio/vision/video_revolver.py` | Collect visual proposals without rewriting the audio transcript or speaker labels. |
| `voice_studio/vision/video_scene.py` | Plan local scene OCR and object recognition without re-running face detection. |
| `voice_studio/vision/visual_attributes.py` | Estimate head turn and a dominant clothing patch from a clear detector sample. |

## Граф и цифровые профили

Нормализация наблюдений, временные окна, идентичность предметов, вложенность, происхождение и компактное хранение.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/graph/dialogue_audit_graph.py` | Publish vector-free voice identities and review-required dialogue issues beside raw evidence. |
| `voice_studio/graph/media_analysis_revolver.py` | Persist successful media-analysis stages so a local graph build can resume. |
| `voice_studio/graph/media_graph_events.py` | Detach cumulative graph reports before asynchronous immutable publication. |
| `voice_studio/graph/media_graph_expressions.py` | Keep facial-expression classifier evidence queryable with explicit temporal provenance. |
| `voice_studio/graph/media_graph_frame.py` | Render a source frame at a media-graph time without storing screenshot blobs. |
| `voice_studio/graph/media_graph_frame_cache.py` | Bounded process-local reuse of undecorated source PNGs and color measurements. |
| `voice_studio/graph/media_graph_health.py` | Expose evidence gaps without interpreting detection counts as people or model accuracy. |
| `voice_studio/graph/media_graph_ingest.py` | Index local media on its playback clock using bounded transcription windows. |
| `voice_studio/graph/media_graph_interfaces.py` | Project sampled visual screen evidence onto the source-clock media graph. |
| `voice_studio/graph/media_graph_interval.py` | Bounded, source-clock evidence briefs from one immutable media graph. |
| `voice_studio/graph/media_graph_interval_frames.py` | Build a bounded source-frame grid with measured timestamps and separate captions. |
| `voice_studio/graph/media_graph_interval_wire.py` | Reversibly tabulate an interval packet without selecting or interpreting evidence. |
| `voice_studio/graph/media_graph_mod_extensions.py` | Validate and project bounded user-defined graph facts without changing native nodes. |
| `voice_studio/graph/media_graph_model.py` | Build a typed, source-clock graph from local media analysis observations. |
| `voice_studio/graph/media_graph_objects.py` | Track sampled object boxes within cuts and describe their relation to stable speakers. |
| `voice_studio/graph/media_graph_placement.py` | Fit a transparent replacement asset to one measured object on the source clock. |
| `voice_studio/graph/media_graph_progress.py` | Bounded progress state for immutable snapshots created by one graph build. |
| `voice_studio/graph/media_graph_store.py` | Immutable local media snapshots with indexed time ranges, text and relations. |
| `voice_studio/graph/media_object_profiles.py` | Describe shot-local objects using detector labels and text on their surface. |
| `voice_studio/graph/media_object_review.py` | Cache the optional final identity review independently from scene detection. |
| `voice_studio/graph/media_stage_cache.py` | Dependency-keyed checkpoints for the common short-media pipeline. |
| `voice_studio/graph/media_stage_selection.py` | Validate selective media runs and expose pending work without starting a worker. |
| `voice_studio/graph/object_reappearance.py` | Link non-overlapping object track groups using bounded comparisons of measured views. |
| `voice_studio/graph/object_reconciliation.py` | Optional conservative identity reconciliation from measured source-crop parts. |
| `voice_studio/graph/object_screens.py` | Infer display context from timed UI evidence without replacing detector observations. |
| `voice_studio/graph/scene_hierarchy.py` | Build a frame-local containment tree while preserving the identity and timing graphs. |
| `voice_studio/graph/ui_activity_graph.py` | Publish raw sampled UI anchors without turning their changes into speech identity. |

## Синтез и дубляж

Адаптеры TTS, корпус, попытки, метрики, сборка блоков, тайминг дубляжа и сохранение фона.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/synthesis/auto_synthesis.py` | Persist and run bounded synthesis trials with an immutable evaluation recording. |
| `voice_studio/synthesis/block_assembly.py` | Validate time ranges from approved takes and join their normalized PCM. |
| `voice_studio/synthesis/commercial_voice.py` | Pinned local MLX checkpoints for reference-based voice synthesis. |
| `voice_studio/synthesis/corpus.py` | Stable reading prompts and separate evaluation material. |
| `voice_studio/synthesis/dub_background.py` | Separate a continuous source bed and mix it with an aligned dub. |
| `voice_studio/synthesis/dub_timing.py` | Measure conservative speech edges in 16-bit mono PCM. |
| `voice_studio/synthesis/dub_voice_profiles.py` | Build bounded speaker prompts for cross-language dubbing. |
| `voice_studio/synthesis/engines.py` | Local engine adapters and audio preparation for Voice Studio. |
| `voice_studio/synthesis/fragment_similarity.py` | Compare the voice in matching, timed word windows from an original and a take. |
| `voice_studio/synthesis/prosody_alignment.py` | Acoustic phrase placement and conservative F0 transfer, independent of any TTS model. |
| `voice_studio/synthesis/prosody_evaluation.py` | Conservative experiment-only promotion rules; scores are proxies, never perceptual truth. |
| `voice_studio/synthesis/setup_commercial_voices.py` | Install exact checkpoint revisions after creating the MLX audio environment. |
| `voice_studio/synthesis/setup_dub_background.py` | Install the fixed TorchAudio accompaniment checkpoint for local dubbing. |
| `voice_studio/synthesis/speaker_metric.py` | Measure the cosine of normalized speaker embeddings using the fixed base XTTS encoder. |
| `voice_studio/synthesis/speech_metrics.py` | Measure words and speaker similarity independently for a local WAV pair. |
| `voice_studio/synthesis/video_dub.py` | Render an English dub from a reviewed transcript on the source video's clock. |
| `voice_studio/synthesis/voice_controls.py` | Explicit pronunciation conditioning supported by the pinned local MLX backends. |

## Просодия и контекст

Ударения, контекст предыдущих слов, мини-модели, граф морфем/слогов и подготовка корпуса записи.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/prosody/morpheme_graph.py` | Build a versioned word alignment and optional dictionary morphology layer. |
| `voice_studio/prosody/prosody.py` | Plan opt-in prosody and separate transcript corrections from canonical text. |
| `voice_studio/prosody/reading_story.py` | Russian voice recording script. Each part also works as a separate reference. |
| `voice_studio/prosody/revolver.py` | Prepare independent synthesis candidates and review the resulting speech. |
| `voice_studio/prosody/stress.py` | Compact character model and dictionary for Russian lexical stress. |
| `voice_studio/prosody/stress_char_runtime.py` | Dependency-free inference for the local Russian character stress model. |
| `voice_studio/prosody/stress_context.py` | Sparse local context model for Russian homographs. |

## Контракты агентов

Каталог инструментов, stdio, граф, скриншоты и запросы обработки звука; это API поверх функциональных модулей.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/mcp/agent_audio_mcp.py` | Publish bounded, pollable audio work to MCP clients. |
| `voice_studio/mcp/dialogue_audit_mcp.py` | Expose independently queued dialogue checks and vector-free voice profiles to agents. |
| `voice_studio/mcp/media_graph_mcp.py` | Register the Video as Code workflow on the existing local MCP server. |
| `voice_studio/mcp/runtime_info.py` | Describe the MCP source baseline so clients can detect a long-lived process after updates. |
| `voice_studio/mcp/screen_mcp.py` | Expose local screenshot OCR and geometry as an editable first-pass description. |
| `voice_studio/mcp/ui_bridge_mcp.py` | MCP access to shared semantic tools in a selected live Studio page. |
| `voice_studio/mcp/voice_catalog.py` | Discover registered MCP tools by media domain and task. |
| `voice_studio/mcp/voice_mcp.py` | ag-media-studio MCP over stdio, backed by the running local studio. |
| `voice_studio/mcp/web_graph.py` | Expose the named read operations of the local media graph to the Studio UI. |

## Изолированные обработчики

Запуск моделей в отдельных окружениях и процессах с ограниченным контрактом ввода/вывода.

| Источник | Назначение модуля |
| --- | --- |
| `voice_studio/workers/clothing_sample_worker.py` | Extract small clothing descriptors in the existing MediaPipe CPU environment. |
| `voice_studio/workers/commercial_voice_worker.py` | Render one speech segment with the pinned MLX runtime. |
| `voice_studio/workers/dub_background_worker.py` | Extract the accompaniment of a stereo WAV using the installed HDemucs weights. |
| `voice_studio/workers/light_asd_worker.py` | Score time-aligned local audio and tracked faces with the pinned Light-ASD model. |
| `voice_studio/workers/live_face_worker.py` | Track sampled tab faces in one bounded, warm local OpenCV process. |
| `voice_studio/workers/model_worker.py` | Run inference in the separately installed F5 or XTTS environment. |
| `voice_studio/workers/mrx_audio_worker.py` | Run the pinned MERL three-source model in the existing TorchAudio environment. |
| `voice_studio/workers/person_layer_worker.py` | Run body detection, head and gaze estimation, and clothing masks on decoded crops. |

## Браузерные модули

Формы, навигация, запись, плееры, шкала, инспекторы, разметка и WebMCP. Выводы модели не создаются интерфейсным цветом.

| Источник | Назначение модуля |
| --- | --- |
| `web/analysis-stage-selection.js` | Переключатели и статусы выборочного анализа и повторного запуска. |
| `web/app.js` | Состояние приложения, формы записи/профилей/синтеза, загрузка результатов и маршрутизация рабочих страниц. |
| `web/auto-synthesis.js` | Граф карточек попыток автосинтеза, время, оценки, прослушивание и отображение выбранного результата. |
| `web/browser-ml-kernel.js` | Проверяемое ядро нормализации векторов и косинусного сходства с WebNN и эталонным JS-путём. |
| `web/browser-ml-worker.js` | Изолированный worker для ограниченных проверок backend и вычислений косинусного сходства. |
| `web/browser-ml.js` | Клиент браузерного ML worker: один активный запрос, обработка ошибок и управление жизненным циклом. |
| `web/capture-audio.js` | Захват аудио из MediaStream и сборка PCM/WAV фрагментов в браузере. |
| `web/capture-worklet.js` | AudioWorklet для передачи аудиосэмплов из realtime audio graph в поток захвата. |
| `web/deck.js` | Многодорожечный TrackDeck: общая позиция прослушивания, управление дорожками и индикация деки. |
| `web/dialogue-audit.js` | Независимая проверка границ диалога, переход к кандидатам по времени исходника и черновики ASR без замены исходного текста. |
| `web/editor-monitor-actions.js` | Компактные действия монитора: перенос существующих контролов экспорта/разметки, раскрытие масок и восстановление при dispose. |
| `web/editor-panel-resizer.js` | Переиспользуемый AG-adapter изменения размеров панели мышью и клавиатурой. |
| `web/editor-workspace.js` | Компоновка монитора/инспектора/шкалы, ограничения размеров и сохранение предпочтений рабочей области. |
| `web/expression-frame-preview.js` | Открытие исходного кадра выражения лица и отрисовка рамки головы с синхронизацией времени video. |
| `web/facial-expression-view.js` | Краткое представление измеренной статистики выражений в профиле лица/спикера. |
| `web/facial-expressions.js` | Нормализация интервалов/меток выражений, группы дорожек и ограниченные выборки статистики для UI/MCP. |
| `web/i18n.js` | Выбор RU/EN интерфейса, переводы сообщений и сохранение языковой настройки браузера. |
| `web/image-analysis-model.js` | Проверка импорта изображений и нормализация групп результатов. |
| `web/image-analysis.js` | Импорт изображения, фоновый анализ, разметка предпросмотра и просмотр результата. |
| `web/mcp-bridge.js` | Ревизии контекста и канал страницы для локальных MCP-клиентов. |
| `web/media-transcript.js` | UI-контроллер загрузки/анализа медиа, сохранённых транскриптов и общей установки слоёв отчёта. |
| `web/messages-en.js` | Английский каталог строк интерфейса по исходным русским ключам. |
| `web/morpheme-graph.js` | Просмотр графа морфем и слогов, блоков синтеза и связанных фрагментов. |
| `web/music-sample.js` | UI выделения и прослушивания фрагмента фоновой музыки для дальнейшей проверки. |
| `web/person-layer.js` | Запрос масок одежды и упаковка исходника/настроек в существующий экспортный контракт. |
| `web/player.js` | Доступный плеер WAV: транспорт, позиция, громкость и жизненный цикл обработчиков. |
| `web/processing.js` | Управление дорожками обработки, шумоподавлением/EQ, запросом обработанного WAV и скачиванием результата. |
| `web/profile-moment-timeline.js` | Общая компактная шкала моментов профиля с маркерами, таймкодами и выборкой по времени. |
| `web/speaker-gaze.js` | Интервалы направления взгляда, их метки/иконки и статистика по спикеру. |
| `web/speaker-label.js` | Единая эффективная метка спикера с учётом редакторских исправлений и смешанных реплик. |
| `web/speaker-profile-dialog.js` | Диалог цифрового профиля спикера: свойства, JSON, взгляд, выражения и переходы ко времени. |
| `web/speaker-profiles.js` | Сборка браузерных профилей и привязок face/speaker для списка, диалога и редакторских действий. |
| `web/spectrum.js` | Расчёт/отрисовка спектральных полос и частотной характеристики эквалайзера. |
| `web/stress-comparison.js` | A/B UI проверок ударения и пилота просодии, подписи и группировка дублей. |
| `web/studio-design.js` | Установка дизайн-профиля и темы, палитра сигналов и фабрики поверхностей плееров. |
| `web/studio-icons.js` | SVG-иконки существующего семейства и подсказки для мыши/клавиатуры. |
| `web/studio-mod-fields.js` | Поля настроек модели по схеме и проверка типов значений. |
| `web/studio-model-install.js` | Установка моделей по функциям и вывод прогресса в настройках. |
| `web/studio-mods.js` | Список провайдеров, параметры, доступность и управление. |
| `web/studio-navigation.js` | Одна навигация для desktop-sidebar и native мобильного диалога без дублирования ID. |
| `web/studio-pipeline-model.js` | Проверка зависимостей и раскладка графа обработки. |
| `web/studio-pipeline.js` | Редактируемый граф обработки, управление зависимостями и статусы провайдеров. |
| `web/studio-settings.js` | Страница настроек и действия над сохранёнными материалами через текущие API. |
| `web/studio-shell.js` | Сворачивание sidebar в icon rail, хранение предпочтения и подсказки скрытых подписей. |
| `web/studio-storage.js` | Представление инвентаря хранения, кандидатов очистки и явных операций пользователя. |
| `web/tool-names.js` | Общие актуальные имена инструментов WebMCP и моста открытой страницы. |
| `web/transcript-chat.js` | Компактные сообщения расшифровки, сохраняющие исходные действия и подробности анализа. |
| `web/transcript-editor.js` | Редактирование сохранённого транскрипта и отдельных реплик с контролем текущей ревизии. |
| `web/transcript-inspector.js` | Сворачивание и восстановление правой панели расшифровки без удаления исходного video. |
| `web/ui-measurement-geometry.js` | Рамки экрана, отступы текста и попарное выравнивание в CSS-пикселях. |
| `web/ui-measurements.js` | Измерение DOM и поиск отображаемых элементов для агентов студии. |
| `web/video-analysis.js` | Чтение потока прогресса, слияние частичных результатов и восстановление состояния анализа. |
| `web/video-color-palette.js` | Палитра измеренных цветов кадра/предмета и доступные текстовые значения. |
| `web/video-dub.js` | UI перевода и сборки английского дубляжа, состояния задания и выдачи результата. |
| `web/video-evidence.js` | Кандидаты фото лица для реплики, основания Light-ASD и действия прикрепления/удаления подтверждённого фото спикера. |
| `web/video-frame-details.js` | Компактные сведения текущего кадра: планы, речь, лица, фигуры и предметы. |
| `web/video-object-appearance-timeline.js` | Мини-шкала повторных появлений предмета с позициями, интервалами и подсказками времени. |
| `web/video-object-profile.js` | Представление цифрового профиля предмета, исходных меток/уверенности, кадров и склейки прокрутки. |
| `web/video-overlay.js` | Единый план и canvas-отрисовка технического слоя: спикеры, рамки, предметы, реплики и экспорт. |
| `web/video-scene.js` | Просмотр визуального отчёта сцены, OCR/предметов, профилей и переходов к соответствующим кадрам. |
| `web/video-shortcuts.js` | Транспорт video и горячие клавиши с границами времени, исключением полей ввода и модальных окон. |
| `web/video-statistics-chart.js` | Адаптивный SVG-график статистики: дискретные интервалы, точные подсказки и выбор с мыши, клавиатуры или касания; освобождение наблюдателей. |
| `web/video-statistics-view.js` | Нативный диалог статистики с SVG-графиками, точными таблицами и переходом в исходное видео. |
| `web/video-statistics.js` | Агрегация готового отчёта: уникальные профили, слова и измеренные временные ряды без запуска моделей. |
| `web/video-timeline.js` | Построение и управление многослойной монтажной шкалой: дорожки, seek, масштаб, fit и плотность. |
| `web/web-mcp.js` | Регистрация браузерных MCP-инструментов для текущей страницы и UI управления подключением. |
