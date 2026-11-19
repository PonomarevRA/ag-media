# Локальный аудит диалога

Аудит отдельно находит интервалы, требующие проверки: несколько акустических меток внутри одной фразы, смену говорящего, одновременные голоса, редкий текст над длинной речью, смещение границ и речевые интервалы без текста. Он сохраняет инвентарь кандидатов с причинами и исходными таймкодами. Акустический кластер не считается человеком. Кандидат не означает подтверждённую ошибку.

Исходный ASR не меняется. Необязательный повтор локального Whisper работает только на коротких неперекрывающихся голосовых фрагментах. Его нативный JSON хранится в `repairs.windows[].rawASR`; отдельные `suggestions` требуют проверки. Слова из повторного распознавания не распределяются автоматически между людьми.

## Запуск

Из корня Voice Studio, через существующее окружение с установленными зависимостями:

```sh
.venv/bin/python scripts/check_dialogue_audit.py \
  --report /absolute/analysis/transcript.json \
  --turns /absolute/analysis/speech-turns.json \
  --source /absolute/media/source-mono-16k.wav \
  --output /absolute/analysis/dialogue-audit
```

`--rerun --max-windows 6 --max-audio-ms 30000 --timeout 120` добавляет ограниченный повтор ASR. `--model /absolute/existing/model.bin` выбирает уже установленную модель. Скачивания и внешних запросов нет. CLI печатает события в stderr, итоговый JSON с путём результата — в stdout; код возврата 0 означает complete, 2 — ошибку/отмену/таймаут. SIGINT/SIGTERM останавливает принадлежащий запуску процесс Whisper.

Источник CLI — полный mono 16 kHz, 16-bit PCM WAV на исходной шкале времени. Исторический `durationMs` может округляться до 100 мс; каждый фактический crop дополнительно проверяется по границе WAV. Если ASR-отчёт содержит хеш исходного MOV, интеграция должна использовать API с `source_metadata`, сохраняя различие MOV и декодированного PCM. Удалять хеш из отчёта для обхода проверки нельзя.

## Importable API

```python
from voice_studio.speech.dialogue_audit import AuditOptions, audit_dialogue
from voice_studio.speech.dialogue_audit_runner import run_audit

inventory = audit_dialogue(
    report, turns,
    source={"sha256": original_sha256, "durationMs": report["durationMs"]},
    visual_facts=optional_av_facts,
    options=AuditOptions(),
)

result = run_audit(
    report_path, turns_path, pcm_wav_path, output_dir,
    visual_path=None, options=AuditOptions(),
    rerun=False, max_windows=6, max_audio_ms=30000, timeout_seconds=120,
    cancelled=lambda: cancellation_event.is_set(), emit=on_progress,
    model_path=None,
    source_metadata={"path": original_mov_path, "sha256": original_sha256,
                     "durationMs": report_duration_ms, "clockOffsetMs": 0},
)
```

`audit_dialogue` — чистая детерминированная функция без I/O. `turns` принимает список или `{"turns": [...]}`. Каждый факт имеет целочисленные `startMs`, `endMs` и необязательный `speaker` (integer/string/null). Интервалы полуоткрытые `[startMs, endMs)`. Все находятся внутри `durationMs`. `sourceSha256`, `sourceDigest` и `source.sha256` отчёта, если присутствуют, должны совпадать с `source.sha256`.

`run_audit` проверяет байты PCM. Передавая `source_metadata`, вызывающий код отвечает за проверку исходного MOV; результат явно указывает `reportBinding=caller-verified-original-source` и сохраняет отдельные `pcmSha256`, `pcmPath`, `pcmSizeBytes`. Без метаданных WAV считается исходником. Если исторический отчёт не содержит хеша, результат честно помечен `caller-supplied-source-duration-checked`.

## JSON результата

| Поле | Содержимое |
|---|---|
| `schemaVersion`, `algorithmVersion` | Версии формата и детектора |
| `source`, `inputHashes`, `options`, `cacheKey` | Источник, хеши входных JSON и параметры детектора |
| `candidates[]` | `id`, `startMs`, `endMs`, `priority`, `reasons`, `evidence`, `reviewWindows` |
| `evidence` | Индексы исходных фраз, акустические интервалы, длительность речи, плотность слов, смены меток, фактическое пересечение, AV provenance |
| `reviewWindows[]` | Чистые интервалы с исходной акустической меткой и `identity=unresolved` |
| `repairs` | Состояние, бюджет, нативный ASR по окнам и отдельные suggestions |
| `runOptions`, `runKey` | Версия runner, бюджет, хеши модели и executable, параметры повторного ASR |
| `status`, `artifactPath`, `cacheHit`, `elapsedSeconds` | Итог запуска; на cache hit время относится к текущему чтению, `cachedResultElapsedSeconds` — к исходному результату |

`reasons` содержит устойчивые коды: `multiple-acoustic-labels`, `speaker-transition-inside-cue`, `simultaneous-voices`, `sparse-text-over-voiced-audio`, `cue-boundary-drift`, `unknown-acoustic-label`, `voiced-audio-without-text`.

Последовательная смена голосов не означает одновременную речь. Дубликаты интервалов удаляются перед подсчётом длительностей. Короткая посторонняя метка ниже порога не создаёт mixed-кандидат. Неизвестный speaker сохраняется неизвестным. Из ASR snippets вычитаются интервалы других меток. Кроп с PCM RMS ниже 80 не передаётся Whisper. Для sparse-кандидатов первым выбирается самый длинный чистый фрагмент: короткий распознанный вопрос не должен постоянно вытеснять потерянный ответ.

`timingBasis=asr-segment-offsets`, `forceAligned=false` обязательны для suggestions. Нативные token offsets и `t_dtw=-1` сохраняются без reinterpretation. Точного forced alignment этот пайп не выполняет.

## AV и безопасность результата

Опциональные AV-факты добавляют evidence. Явные визуальные факты получают `authority=explicit-video-first`, слабые/отсутствующие — `audio-fallback`. Аудит не меняет speaker/person по этим фактам. Сырые зелёные рамки `visual-activity-candidate` не объявляются явным доказательством личности.

По умолчанию: не более 50 000 входных строк (cues + turns + AV samples), 2 000 кандидатов, 256 turns на фразу. Превышение отклоняется. Размеры ASR snippets — 650–6500 мс. Жёсткие пределы runner: 100 окон, 600 000 мс аудио и 3600 секунд общего deadline. Пользовательские меньшие бюджеты сохраняются. Deadline/отмена проверяются при хешировании, ожидании lock и работе каждого child. Детерминированный этап ограничен лимитами строк и проверяется между этапами; произвольный callback прогресса должен возвращаться быстро.

Runner учитывает `core.media_resources.stage_threads(4)`: при `VOICE_RESOURCE_PROFILE=throughput` и `VOICE_CPU_BUDGET=1` Whisper получает `-t 1`. Выбранное значение фиксируется один раз, сохраняется в `runOptions.effectiveThreads` и `decodeOptions`, затем участвует в ключе кэша.

Ключ успешного кэша включает источник (MOV и PCM при интеграции), report/turn/AV JSON, версии, options, модель и executable. На повторе эти входы снова хешируются. Незавершённый результат не попадает в успешный кэш. Все JSON пишутся атомарно. Успешный результат — `audit.json`; неуспешный ASR-повтор сохраняет `attempt-<id>.json` и оставляет предыдущий `audit.json`. Ошибка валидации или отмена до построения инвентаря возвращает исключение без изменения последнего результата.

## Проверка контрактов

```sh
.venv/bin/python -m unittest discover -s tests -p test_dialogue_audit.py -v
```

Покрыты последовательная/одновременная речь, sparse, неизвестный speaker, дубликаты, тишина, подавление короткой ложной метки, выход за границы, source/hash mismatch, MOV/PCM identity, кэш и смена входов/options, бюджеты, отдельный raw ASR, отказ/таймаут/отмена с сохранением последнего результата, реальное убийство subprocess и освобождение lock. Этот набор не доказывает recall/precision кандидатов на других звонках и не заменяет оценку ASR.
