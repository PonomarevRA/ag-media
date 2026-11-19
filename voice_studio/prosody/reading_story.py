"""Russian voice recording script. Each part also works as a separate reference."""

STORY_TITLE = "Поезд на рассвете"
STORY_PARTS = (
    ("Спокойный рассказ", "В четверг чуть свет Фёдор вышел из тёплой избы. У крыльца его ждал рыжий щенок."),
    ("Вопрос и пауза", "«Это свист? Где наш поезд?» У платформы тихо пели птицы, потом донёсся гудок. «Я слышу!»"),
    ("Перечисление", "Фёдор проверил рюкзак: хлеб, мёд, щётка, ключ и подъёмный крюк."),
    ("Просьба и восклицание", "«Возьми фонарь, пожалуйста. Стой, не беги!» Щенок юркнул в чащу и выскочил к рельсам, спугнув ежа."),
    ("Контраст и спокойный финал", "Там был не поезд, а цирковой фургон. Мягкий свет лёг на траву, а щенок свернулся клубком."),
)
STORY_TEXT = " ".join(text for _, text in STORY_PARTS)


def recording_story(language):
    """Expose the original text and its exact cut points for the Russian recorder."""
    if language != "ru":
        return None
    return {"title": STORY_TITLE, "text": STORY_TEXT, "parts": [
        {"label": label, "text": text} for label, text in STORY_PARTS
    ]}
