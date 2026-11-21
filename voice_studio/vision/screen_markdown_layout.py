"""Project screen boxes onto an HTML table embedded in a Markdown document."""

from math import ceil
from statistics import mean

COLUMNS = 16
CONTROLS = {"button_candidate", "field_candidate", "icon_candidate", "timeline_clip"}


def interval(item):
    """
    Map an observed box to an exclusive span of grid columns.

    item is a screen observation with a normalized box. Keep at least one
    column for small icons and clip rounding at the page boundary.
    """
    x, _, w, _ = item["box"]
    first = max(0, min(COLUMNS - 1, round(x * COLUMNS)))
    return first, max(first + 1, min(COLUMNS, round((x + w) * COLUMNS)))


def covers_text(control, line):
    """
    Test whether a control's visible caption duplicates an OCR label.

    Both observations have escaped labels and normalized boxes. Require
    the same label and a centered containment match.
    """
    if not control["label"] or control["label"] != line["label"]:
        return False
    x, y, w, h = control["box"]
    a, b, p, q = line["box"]
    return x <= a + p / 2 <= x + w and y <= b + q / 2 <= y + h and p <= 1.3 * w and q <= 1.3 * h


def visual_items(items):
    """
    Keep the observed captions once in the visual layout.

    items are the indexed observations. Duplicate OCR labels inside their
    labeled button cells remain present in the separate coordinate index.
    """
    controls = [row for row in items if row["role"] in CONTROLS]
    return sorted((row for row in items
                   if row["role"] != "text_line" or not any(covers_text(control, row) for control in controls)),
                  key=lambda item: (item["box"][1], item["box"][0]))


def table_bands(items, row_height):
    """
    Build screen bands from overlapping boxes without merging distinct rows.

    items is a y-ordered list of observations. row_height is a normalized
    height used to keep large inferred regions from swallowing inner rows.
    """
    bands = []
    for row in items:
        _, y, _, height = row["box"]
        bottom = y + min(height, row_height)
        if bands and y < bands[-1][1] and min(bottom, bands[-1][1]) - y >= .35 * min(height, row_height):
            bands[-1][2].append(row)
        else:
            bands.append((y, bottom, [row]))
    return bands


def grid_cell(first, end, markup):
    """
    Render one observed or empty cell across the specified grid columns.

    first/end are the exclusive bounds. markup contains only generated
    markup and pre-escaped OCR labels.
    """
    span = end - first
    return f'<td colspan="{span}" width="{span / COLUMNS * 100:g}%" valign="top">{markup}</td>'


def cell_label(item, roles):
    """
    Render a caption and distinguish visible candidate controls from prose.

    item contains a roles key and a pre-escaped label. roles supplies safe
    role descriptors for readings with no text, such as icon candidates.
    """
    role, label = item["role"], item["label"] or roles.get(item["role"], "•")
    if role in CONTROLS:
        return f"<kbd>{label}</kbd>"
    if role in {"window", "list", "block_candidate"}:
        return f"<strong>{label}</strong>"
    return label


def render_band(band, roles):
    """
    Place same-row observations into slots with explicit gaps and widths.

    band contains observations measured at the same vertical position.
    Close labels sharing grid columns are shown in one cell, in screen order.
    """
    cells = []
    for row in sorted(band, key=lambda item: item["box"][0]):
        first, end = interval(row)
        label = cell_label(row, roles)
        if cells and first < cells[-1][1]:
            a, b, labels = cells[-1]
            cells[-1] = (a, max(b, end), [*labels, label])
        else:
            cells.append((first, end, [label]))
    markup, position = [], 0
    for first, end, labels in cells:
        if first > position:
            markup.append(grid_cell(position, first, "&nbsp;"))
        markup.append(grid_cell(first, end, "<br>".join(labels)))
        position = end
    if position < COLUMNS:
        markup.append(grid_cell(position, COLUMNS, "&nbsp;"))
    return "<tr>" + "".join(markup) + "</tr>"


def html_layout(items, roles, width, height):
    """
    Emit a headerless table whose rows and cell spans follow measured screen boxes.

    items include only trusted geometry and pre-escaped captions. width/height
    are original image dimensions; large vertical gaps are capped to keep
    tall reports compact. Return HTML lines for a Markdown HTML block.
    """
    typical = min(1, max(16, width / 24) / height)
    rows = ['<table width="100%">', f'<colgroup><col span="{COLUMNS}" width="{100 / COLUMNS:g}%"></colgroup>', "<tbody>"]
    position = 0
    for y, bottom, band in table_bands(visual_items(items), typical):
        gap = y - position
        if gap > typical * 1.6:
            blanks = min(3, ceil(gap / typical) - 1)
            rows.extend(f"<tr>{grid_cell(0, COLUMNS, '&nbsp;')}</tr>" for _ in range(blanks))
        rows.append(render_band(band, roles))
        position = bottom
    rows.extend(["</tbody>", "</table>"])
    return rows


def center_x(item):
    """Return the normalized horizontal center of an observed block."""
    x, _, width, _ = item["box"]
    return x + width / 2


def stack_text(items, height_px):
    """
    Keep nearby consecutive OCR lines as one multiline caption block.

    items is a list of visual observations without duplicate control
    captions. height_px bounds inter-line distance on tall mosaics. Stacking
    retains reading order and the union of measured boxes.
    """
    output = []
    maximum_gap = min(.045, 23 / height_px)
    for item in items:
        x, y, width, height = item["box"]
        if item["role"] == "text_line":
            adjacent = []
            for index, old in enumerate(output):
                a, b, w, h = old["box"]
                gap = y - b - h
                if (old["role"] == "text_line" and -.012 <= gap <= maximum_gap
                        and abs(center_x(item) - center_x(old)) <= .085
                        and min(w, width) >= .25 * max(w, width)):
                    adjacent.append((abs(gap), index))
            if adjacent:
                index = min(adjacent)[1]
                old = output[index]
                a, b, w, _ = old["box"]
                left = min(a, x)
                output[index] = {**old, "box": [left, b, max(a + w, x + width) - left, y + height - b],
                                 "label": old["label"] + "<br>" + item["label"]}
                continue
        output.append(item)
    return output


def page_sections(items):
    """
    Yield local groups of rows rather than imposing one grid on the page.

    Wide fields, toolbar-like elements or major vertical gaps divide the
    layout. items must be in top-to-bottom order.
    """
    current, bottom = [], 0
    for item in items:
        _, top, width, height = item["box"]
        wide = width >= .60 and item["role"] != "text_line"
        if current and (wide or top - bottom > .15):
            yield current
            current = []
        if wide:
            yield [item]
        else:
            current.append(item)
        bottom = max(bottom, top + height)
    if current:
        yield current


def markdown_matrix(items, roles):
    """
    Recover local native Markdown rows and columns from measured alignment.

    items are observed blocks within one section. roles labels an unlabeled
    candidate. Return None for crowded or competing cells that require spans.
    """
    column_groups = []
    for center in sorted(center_x(item) for item in items):
        if column_groups and center - mean(column_groups[-1]) < .085:
            column_groups[-1].append(center)
        else:
            column_groups.append([center])
    if len(column_groups) > 8:
        return None
    anchors = [mean(group) for group in column_groups]
    bands = table_bands(items, 1)
    rows = []
    for _, _, band in bands:
        cells = [""] * len(anchors)
        for item in band:
            column = min(range(len(anchors)), key=lambda index: abs(center_x(item) - anchors[index]))
            if cells[column]:
                return None
            cells[column] = item["label"] or roles.get(item["role"], "•")
        rows.append(cells)
    def row(values):
        """Write one native pipe row with already escaped cell text."""
        return "| " + " | ".join(values) + " |"
    return [row([""] * len(anchors)), row(["---"] * len(anchors)), *(row(cells) for cells in rows)]


def layout_table(items, roles, width, height):
    """
    Reconstruct the visible page as semantic Markdown tables.

    items include trusted normalized geometry and escaped labels; roles
    supplies candidate names. width/height are used only for complex sections
    where an HTML table inside the Markdown preserves cell spans.
    """
    visual = stack_text(visual_items(items), height)
    rows = []
    for section in page_sections(visual):
        if rows:
            rows.append("")
        rows.extend(markdown_matrix(section, roles) or html_layout(section, roles, width, height))
    return rows
