"""Studio-authored reading prompts with instructions separate from spoken text."""

from dataclasses import asdict, dataclass
import re

VERSION = "voice-reference-v1"


@dataclass(frozen=True)
class Prompt:
    """One prompted take and the intended delivery."""

    id: str
    label: str
    hint: str
    text: str
    role: str = "style"


GUIDES = {
    "ru": (
        Prompt("neutral-a", "Нейтрально", "Обычный темп, мягкое понижение в конце.",
               "Утром над рекой рассеялся туман. На причале уже ждали рыбаки, а вдоль берега тихо шуршал камыш.", "neutral"),
        Prompt("neutral-b", "Разные звуки", "Говори разборчиво, сохраняя естественный темп.",
               "Фёдор взял ключи и вышел во двор. Щенок догнал его у большого ящика с грушами и смешно чихнул.", "neutral"),
        Prompt("question", "Вопрос", "Выдели «сегодня» и подними тон на этом слове.",
               "Мы отправляемся сегодня, а не завтра?"),
        Prompt("contrast", "Уточнение", "Поставь логическое ударение на «справа». Закончи спокойно.",
               "Вход справа, рядом с высокой синей дверью."),
        Prompt("list", "Перечисление", "Сохрани ожидание после первых пунктов, заверши на последнем.",
               "В сумке лежат блокнот, фонарик, тёплый шарф и две бутылки воды."),
        Prompt("pause", "Пауза", "Сделай паузу около полусекунды между предложениями.",
               "Сначала я посмотрю на карту. Потом покажу тебе короткую дорогу."),
        Prompt("exclamation", "Радостно", "Начни чуть энергичнее обычного. Сохрани комфортную громкость.",
               "Отлично, мы успели! До отправления ещё три минуты."),
        Prompt("holdout", "Контрольная фраза", "Прочитай естественно. Этот дубль используется для сравнения синтеза.",
               "В пятницу после обеда поезд прибудет на вторую платформу. Ты будешь ждать меня у выхода?", "evaluation"),
    ),
    "en": (
        Prompt("neutral-a", "Нейтрально", "Обычный темп, мягкое понижение в конце.",
               "By morning the mist had lifted from the river. Two boats were already waiting beside the wooden pier.", "neutral"),
        Prompt("neutral-b", "Разные звуки", "Говори разборчиво, сохраняя естественный темп.",
               "Grace pushed the gate open. A small dog dashed across the yard and stopped beside a box of ripe peaches.", "neutral"),
        Prompt("question", "Вопрос", "Выдели «today» и подними тон на этом слове.",
               "Are we leaving today, or should we wait until tomorrow?"),
        Prompt("contrast", "Уточнение", "Поставь логическое ударение на «right». Закончи спокойно.",
               "The entrance is on the right, next to the tall blue door."),
        Prompt("list", "Перечисление", "Сохрани ожидание после первых пунктов, заверши на последнем.",
               "I packed a notebook, a small torch, a warm scarf, and two bottles of water."),
        Prompt("pause", "Пауза", "Сделай паузу около полусекунды между предложениями.",
               "First I will check the map. Then I will show you a quicker route."),
        Prompt("exclamation", "Радостно", "Начни чуть энергичнее обычного. Сохрани комфортную громкость.",
               "Great, we made it! There are still three minutes before departure."),
        Prompt("holdout", "Контрольная фраза", "Прочитай естественно. Этот дубль используется для сравнения синтеза.",
               "The train arrives at platform two on Friday afternoon. Will you be waiting by the exit?", "evaluation"),
    ),
}


def reference_script(language):
    """Return the shareable transcript and delivery guide for one Studio language."""
    return {"version": VERSION, "language": language, "prompts": [asdict(row) for row in GUIDES[language]]}


def spoken_words(text):
    """Ignore presentation punctuation when checking the prompted words."""
    return re.findall(r"[^\W_]+", text.casefold().replace("ё", "е"))


def recorded_prompt(language, prompt_id, text):
    """Verify the declared prompt against the approved transcript."""
    prompt = next((row for row in GUIDES[language] if row.id == prompt_id), None)
    if prompt is None or spoken_words(prompt.text) != spoken_words(text):
        raise ValueError("Расшифровка должна совпадать со словами выбранной фразы эталона.")
    return prompt
