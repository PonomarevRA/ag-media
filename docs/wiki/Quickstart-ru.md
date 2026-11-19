# Запуск и переименование

[English](Quickstart.md) · [ag-media-studio](../../README.md)

## Локальная установка

Требуется Python 3.12+. Для реального анализа нужны FFmpeg и выбранные модели. Выполни команды из корня исходников и открой `http://127.0.0.1:8767/`. Демо использует тестовые тоны.

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

## Что изменилось в имени

| Поверхность | Текущее поведение |
| --- | --- |
| Продукт и сервер MCP | `ag-media-studio` |
| Новые launcher-файлы | `ag_media_studio.py`, `ag_media_studio_mcp.py` |
| Каталог проекта | Имя каталога можно выбрать при клонировании или распаковке |
| Старые launcher-файлы | `server.py` и `voice_mcp.py` продолжают работать |
| Внутренний Python-пакет | `voice_studio` сохранён, чтобы не сломать импорты и worker-команды |
| Веса, зависимости и настройки | Исторические папки `Voice Studio`, переменные `VOICE_*` и ключи браузера сохранены |

Для рабочей студии убери `--demo` и выбери внешний каталог через `--data`. Навыки агентов и настройка MCP описаны в `docs/AGENT-SETUP.md` и `skills/README.md` в репозитории.

## Первый сценарий

- Открой «Запись и транскрипция», выбери видео или аудио.
- Выбери нужные слои анализа. Следи за статусами этапов, а не только общим временем.
- Проверь реплики, спикеров и рамки на одном и том же таймкоде.
- Открой профиль или элемент дорожки для деталей. Скачивай результат через существующие действия экспорта.

- [Данные и модели](Models-and-storage-ru.md)
- [Интерфейс](Interface-ru.md)
