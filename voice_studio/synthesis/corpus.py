"""Stable reading prompts and separate evaluation material."""

EXTRA = {
    "ru": [
        ("short", "Ну что, поехали?"),
        ("short", "Постой! Я уже здесь."),
        ("long", "Представь, что у каждого игрока есть свой план, но итог зависит от обоих. Если изменить только один ход, станет ли кому-нибудь лучше?"),
        ("short", "Встречаемся в полдень."),
        ("medium", "Через пять минут поезд отправится с третьей платформы."),
        ("long", "Мы договорились делить задачи поровну. Сегодня у Маши срочный отчёт, поэтому я возьму две заявки, а завтра мы поменяемся."),
        ("short", "Скажи это ещё раз."),
        ("medium", "Мягкий свет ложится на крышу, а дождь шумит всё тише."),
        ("long", "В одном кафе подают горячую пиццу, в другом готовят лапшу. Если нам важнее увидеться, придётся сначала договориться о месте."),
        ("short", "Две чашки, пожалуйста."),
        ("medium", "Почему рыжий щенок спрятал мяч под широким креслом?"),
        ("long", "Вчера задача казалась простой. Мы записали все варианты, проверили цифры и обнаружили, что самый быстрый путь ведёт в тупик."),
        ("short", "Нет, подожди немного."),
        ("medium", "Саша купил три груши и поделился одной с другом."),
        ("long", "На первом круге никто не уступил. На втором мы попробовали чередоваться, и очередь стала двигаться быстрее. Как бы ты поступил?"),
        ("short", "Вот это поворот!"),
    ],
    "en": [
        ("short", "Ready when you are."),
        ("short", "Wait! I can see you."),
        ("long", "Imagine two players making their own plans. The result depends on both choices. What happens if only one player changes their mind?"),
        ("short", "Meet me at noon."),
        ("medium", "In five minutes, the train leaves from platform three."),
        ("long", "We agreed to share the work equally. Maya has an urgent report today, so I will take two tickets, and tomorrow we can switch."),
        ("short", "Say that once more."),
        ("medium", "Soft light falls across the roof as the rain grows quieter."),
        ("long", "One café serves hot pizza and the other makes noodles. If seeing each other matters most, we should agree on a place first."),
        ("short", "Two cups, please."),
        ("medium", "Why did the small brown puppy hide its ball under the chair?"),
        ("long", "Yesterday the task looked simple. We wrote down the options, checked the numbers, and found that the fastest route led nowhere."),
        ("short", "No, wait a little."),
        ("medium", "Sam bought three pears and shared one with a friend."),
        ("long", "Nobody took turns in the first round. The second time we alternated, and the line moved faster. What would you have done?"),
        ("short", "What a surprise!"),
    ],
}
BANDS = {"short": "2–4 с", "medium": "4–8 с", "long": "8–14 с"}
HELD_OUT = {
    "ru": ["Ты меня слышишь?", "На пятой минуте мы поменяли правила и снова запустили таймер.",
           "Пока остальные спорили о том, кто должен начать, Лена предложила провести два коротких раунда. Первый ход в каждом раунде сделали разные игроки."],
    "en": ["Can you hear me?", "At the five-minute mark, we changed the rules and restarted the timer.",
           "While everyone argued about who should start, Lena suggested two short rounds. A different player made the first move in each round."],
}


def reading_plan(language, original, recordings):
    """Find the next unread prompt, keeping legacy exact-text takes recognized."""
    items = [(text, "medium") for text in original]
    items.extend((text, band) for band, text in EXTRA[language])
    completed = set()
    for row in recordings:
        if row["prompt_number"] is not None:
            completed.add(row["prompt_number"])
        else:
            completed.update(i for i, (text, _) in enumerate(items, 1) if text == row["text"])
    next_prompt = next(((i, text, BANDS[band]) for i, (text, band) in enumerate(items, 1)
                        if i not in completed), (None, "", ""))
    number, text, band = next_prompt
    return {"prompt": text, "promptNumber": number, "promptBand": band, "promptCount": len(items),
            "promptDone": len(completed), "heldOut": HELD_OUT[language]}
