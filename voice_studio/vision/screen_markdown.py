"""Serialize recorded interface evidence as a readable Markdown layout and coordinate index."""

from collections import defaultdict
from math import isfinite
import re

from voice_studio.vision.screen_markdown_layout import layout_table


LABELS = {
    "ru": ("Распознанный интерфейс", "Склейка прокрутки", "Расположение на экране", "Координаты для агента",
           "Подпись", "Тип", "Ячейка", "Рамка", "Кадр",
           "Нет распознанных элементов", "OCR и роли элементов требуют проверки"),
    "en": ("Recognized interface", "Stitched scroll", "Screen layout", "Agent coordinates",
           "Label", "Role", "Cell", "Box", "Frame",
           "No recognized elements", "OCR and control roles are provisional"),
}
ROLES = {
    "ru": {"text_line": "Текст", "button_candidate": "Кнопка?", "field_candidate": "Поле ввода?",
           "icon_candidate": "Иконка?", "block_candidate": "Блок?", "window": "Окно",
           "list": "Список", "list_row": "Строка списка", "timeline_clip": "Клип", "object": "Предмет"},
    "en": {"text_line": "Text", "button_candidate": "Button?", "field_candidate": "Field?",
           "icon_candidate": "Icon?", "block_candidate": "Block?", "window": "Window",
           "list": "List", "list_row": "List row", "timeline_clip": "Clip", "object": "Object"},
}


def literal(value):
    """Keep OCR labels as text, with no tables, markup, links or line breaks from the source."""
    compact = " ".join(str(value or "").split())[:300]
    return re.sub(r"[&<>|\[\]!*_`~:#@().\\]", lambda match: f"&#{ord(match[0])};", compact)


def table(headers, rows):
    """Write a GitHub-flavored Markdown table with already escaped cells."""
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(row) + " |" for row in rows)]


def timecode(ms):
    """Use millisecond positions on the original recording or capture sequence."""
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def entries(lines, controls):
    """Deduplicate identical labels and near-identical role boxes while preserving source coordinates."""
    seen = set()
    selected = []
    for item in (*lines, *controls):
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if any(type(value) not in (int, float) or not isfinite(value) for value in box):
            continue
        label = literal(item.get("text") or item.get("title"))
        role = item.get("role", "text_line")
        key = role, label, tuple(round(float(value), 3) for value in box)
        if key in seen or (role == "text_line" and not label):
            continue
        seen.add(key)
        selected.append({**item, "role": role, "label": label})
    return sorted(selected, key=lambda row: (row["box"][1] + row["box"][3] / 2, row["box"][0]))


def coordinate_rows(items, width, height):
    """Index exact pixel boxes and optional original times under separate role headings."""
    by_role = defaultdict(list)
    for item in items:
        x, y, w, h = item["box"]
        left, top = round(x * width), round(y * height)
        right, bottom = min(width, round((x + w) * width)), min(height, round((y + h) * height))
        pixel = item.get("pixelBox") or [left, top, right - left, bottom - top]
        column, row = item.get("column"), item.get("row")
        cell = f"C{column} R{row}" if column and row else "—"
        timestamp = item.get("timeMs")
        source = item.get("sourceBox")
        source_box = "`" + ", ".join(f"{value:.4f}" for value in source) + "`" if source else "—"
        by_role[item["role"]].append([item["label"] or "—", cell, f"`{', '.join(map(str, pixel))}`",
                                     timecode(timestamp) if type(timestamp) is int and timestamp >= 0 else "—", source_box])
    return by_role


def screen_section(items, width, height, language, grid=None):
    """Pair a compact visual table with an index grouped by recognizable roles."""
    _, _, layout, details, label, role_title, cell, box, frame, empty, _ = LABELS[language]
    if not items:
        return [empty, ""]
    roles = ROLES[language]
    units = "пиксели, отсчёт сверху слева" if language == "ru" else "pixels, top-left origin"
    result = [f"**{width} × {height} px**", "", f"### {layout}", "",
              *layout_table(items, roles, width, height), "",
              f"### {details}", "", f"`x, y, width, height` · {units}", ""]
    if grid and grid.get("rows") and grid.get("columns"):
        result.extend((f"C1–C{grid['columns']} · R1–R{grid['rows']}", ""))
    for role, rows in coordinate_rows(items, width, height).items():
        source_box = "Рамка кадра [0–1]" if language == "ru" else "Frame box [0–1]"
        headers = [label, cell, box, frame, source_box]
        columns = [index for index in range(len(headers)) if index in (0, 2) or any(row[index] != "—" for row in rows)]
        role_id = re.sub(r"[^a-zA-Z0-9_-]", "", role)
        result.extend((f"#### {roles.get(role, role_title)} · `{role_id}`", "",
                       *table([headers[i] for i in columns], ([row[i] for i in columns] for row in rows)), ""))
    return result


def screenshot_markdown(result, language="ru"):
    """Build the same document for uploaded images and the local screenshot MCP tool."""
    title, *_, note = LABELS[language]
    width, height = result.get("imageSize") or [1, 1]
    grid = result.get("controlGrid") or {}
    layout = (result.get("interface") or {}).get("elements", ())
    controls = [row for row in layout if row.get("role") != "text_line"]
    for observation in grid.get("controls", ()):
        # Detailed bands supply exact grid positions for controls already in the overview.
        for index, row in enumerate(controls):
            if (observation["role"] == row.get("role")
                    and all(abs(a - b) < .02 for a, b in zip(observation["box"], row["box"], strict=True))):
                controls[index] = {**row, **observation}
                break
        else:
            controls.append(observation)
    structure = result.get("structure") or {}
    for field, role in (("windows", "window"), ("lists", "list"), ("timelineClips", "timeline_clip")):
        controls.extend({**row, "role": role} for row in structure.get(field, ()))
    controls.extend({**row, "role": "list_row"} for group in structure.get("lists", ()) for row in group.get("rows", ()))
    controls.extend({**row, "role": "object", "text": row.get("label")} for row in result.get("objects", ()))
    items = entries(result.get("ocr", ()), controls)
    return "\n".join([f"# {title}", "", note, "", *screen_section(items, width, height, language, grid.get("grid"))])


def scroll_markdown(pages, language="ru"):
    """Keep each registered page separate with its own image coordinates and time span."""
    _, title, *_, empty, note = LABELS[language]
    sections = [f"# {title}", "", note, ""]
    for page in pages:
        start, end = timecode(page["startMs"]), timecode(page["endMs"])
        items = entries(page.get("mosaicText", ()), page.get("controls", ()))
        sections.extend((f"## {start}–{end}", "",
                         *screen_section(items, page["width"], page["height"], language, page.get("grid"))))
    if not pages:
        sections.extend((empty, ""))
    return "\n".join(sections)
