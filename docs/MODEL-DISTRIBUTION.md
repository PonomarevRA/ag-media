# Поставка весов моделей

При установке Whisper Base, YuNet и Zipformer Small INT8 через «Моды и модели»
студия сначала проверяет локальные веса. Если веса нужны, установщик скачивает
модель из [релиза models-v1](https://github.com/PonomarevRA/ag-media/releases/tag/models-v1).
При ошибке скачивания или проверки SHA-256 он автоматически обращается к
оригинальному источнику. Первая загрузка зеркала занимает 149 МиБ даже при
установке одной модели. Проверенный архив доступен другим установкам из временной
папки, пока он хранится в кэше. Установленные веса сохраняются в Models вне приложения.

| Артефакт | Статус поставки |
| --- | --- |
| `whisper/ggml-base.bin` | MIT. Право явно указано на веса Whisper и на зафиксированную конверсию GGML. [Модель Whisper](https://github.com/openai/whisper#license) · [GGML](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1) |
| `vision/yunet.onnx` | MIT в папке конкретной модели. [YuNet](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) |
| Zipformer Small INT8 | Apache-2.0 у издателя этой модели. Пакет содержит INT8-файл и таблицу классов. [Модель](https://huggingface.co/k2-fsa/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15) |
| Whisper Small и Large v3 Turbo в GGML | MIT распространяется на веса и конверсию. Эти файлы можно выпустить отдельно, зафиксировав SHA-256. [Whisper](https://github.com/openai/whisper#license) · [GGML](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1) |
| SFace INT8 2021dec, NanoDet 2022nov, MobileFaceNet 2022july | Лицензии OpenCV Zoo Apache-2.0 распространяются на файлы соответствующих моделей. При выпуске сохраняем уведомления и сверяем точный ONNX. [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license) · [NanoDet](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet#license) · [MobileFaceNet](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition#license) |
| Qwen3-TTS MLX | Карточка MLX-весов указывает Apache-2.0. Для выпуска нужен аудит всего фиксированного снимка и вложенных компонентов. [Карточка модели](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit) |
| Chatterbox MLX | Карточка весов указывает MIT. Полный набор также требует отдельный S3TokenizerV2, его условия пока не подтверждены для перепоставки. [Карточка модели](https://huggingface.co/mlx-community/chatterbox-multilingual-v3) |
| ERes2Net для Sherpa | Сборка с SHA-256 `1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b` — Apache-2.0 с лицензией и NOTICE. Для текущего файла из установщика с другим SHA-256 нужно отдельное подтверждение. [Сборка и уведомления](https://huggingface.co/onemira/eres2net-base-3dspeaker-onnx/tree/main) |
| Light-ASD, Pyannote ONNX из Sherpa и другие модели | Для каждого файла проверяем его собственную лицензию, происхождение и требования к уведомлениям. Право на зеркалирование точных весов Light-ASD и экспорта Pyannote отдельно не подтверждено. [Источники и лицензии](../THIRD-PARTY.md) |

Автономный архив содержит папку `Models`, описание происхождения в
`MANIFEST.json` и уведомления о лицензиях. Проверьте SHA-256 архива и файлов по
манифесту. На macOS по умолчанию перенесите файлы в
`~/Library/Application Support/Voice Studio/Models` с теми же относительными
путями. На любой платформе путь показывает `python voice_paths.py models`.
При необходимости настройте `VOICE_MODELS_DIR` до запуска. В пределах каждого
набора его движки и библиотеки устанавливаются независимо от весов.

Перед публикацией нового пакета сверяем SHA-256 с официальным артефактом,
сохраняем условия и авторские уведомления. Для остальных уже поддержанных
компонентов используем список точных источников. [Инвентарь поставки](../packaging/resource-license-inventory.json)
