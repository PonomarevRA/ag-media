"""Build the offline wiki and Markdown mirror from one structured, versioned source."""

import argparse
import ast
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "docs" / "wiki"
SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


def inline(text):
    """Escape prose; the only inline formatting supported is literal backtick code."""
    parts = str(text).split("`")
    return "".join(f"<code>{escape(part)}</code>" if index % 2 else escape(part)
                   for index, part in enumerate(parts))


def module_index(web_summaries):
    """Capture Python module docs and explicit JS responsibilities without executing code."""
    rows = []
    for path in sorted((ROOT / "voice_studio").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        summary = (ast.get_docstring(tree) or "No module docstring; inspect the source contract.").split("\n")[0]
        rows.append({"path": path.relative_to(ROOT).as_posix(), "group": path.parent.name,
                     "summary": summary, "summaryRu": summary, "summaryOrigin": "source-module-docstring",
                     "sha256": sha256(path.read_bytes()).hexdigest()})
    actual_names = {path.name for path in (ROOT / "web").glob("*.js")}
    if actual_names != set(web_summaries):
        raise ValueError("Update the explicit webModules catalogue for added/removed browser modules")
    for path in sorted((ROOT / "web").glob("*.js")):
        summary = web_summaries[path.name]
        rows.append({"path": path.relative_to(ROOT).as_posix(), "group": "web",
                     "summary": summary["en"], "summaryRu": summary["ru"],
                     "summaryOrigin": "reviewed-wiki-module-responsibility",
                     "sha256": sha256(path.read_bytes()).hexdigest()})
    return rows


def safe_svg(path):
    """Keep schematic assets vector-only: no script, external references, embedded media or HTML."""
    tree = ElementTree.fromstring(path.read_bytes())
    allowed = {"svg", "title", "desc", "g", "rect", "circle", "ellipse", "line", "path", "text", "tspan", "polygon", "polyline"}
    for node in tree.iter():
        if node.tag.split("}")[-1] not in allowed:
            raise ValueError(f"Unsupported schematic element in {path.name}")
        for key, value in node.attrib.items():
            if key.lower().startswith("on") or key.endswith("href") or "url(" in value.lower():
                raise ValueError(f"Active or externally referenced schematic in {path.name}")


def block_markup(block, page_ids, image_names):
    """Render one typed block to HTML and Markdown; raw HTML is never interpreted."""
    kind = block["type"]
    if kind == "p":
        return f"<p>{inline(block['text'])}</p>", block["text"] + "\n"
    if kind == "h2":
        return f"<h2>{inline(block['text'])}</h2>", "## " + block["text"] + "\n"
    if kind == "list":
        return "<ul>" + "".join(f"<li>{inline(item)}</li>" for item in block["items"]) + "</ul>", "\n".join("- " + item for item in block["items"]) + "\n"
    if kind == "code":
        return f"<pre><code>{escape(block['text'])}</code></pre>", "```text\n" + block["text"] + "\n```\n"
    if kind == "table":
        headers = block["headers"]
        body = ["<div class=table-scroll data-ag-width='fill' data-ag-width-min-em='8'><table><thead><tr>" + "".join(f"<th scope=col>{inline(cell)}</th>" for cell in headers) + "</tr></thead><tbody>"]
        md = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in block["rows"]:
            if len(row) != len(headers):
                raise ValueError("Wiki table has inconsistent columns")
            body.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
            md.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
        return "".join(body) + "</tbody></table></div>", "\n".join(md) + "\n"
    if kind == "image":
        name = block["asset"]
        if name not in image_names:
            raise ValueError(f"Missing schematic: {name}")
        alt = escape(block["alt"], quote=True)
        caption = inline(block["caption"])
        return f'<figure><a href="assets/{name}"><img src="assets/{name}" alt="{alt}" width="1200" loading="lazy"></a><figcaption>{caption}</figcaption></figure>', f"![{block['alt']}](assets/{name})\n\n{block['caption']}\n"
    if kind == "links":
        html, md = [], []
        for link in block["items"]:
            target = link["target"]
            if target in page_ids:
                href, md_href = target + ".html", target + ".md"
            elif urlsplit(target).scheme == "https":
                href = md_href = target
            else:
                raise ValueError(f"Unsafe or unknown wiki link: {target}")
            html.append(f'<li><a href="{escape(href, quote=True)}">{inline(link["label"])}</a></li>')
            md.append(f'- [{link["label"]}]({md_href})')
        return "<ul>" + "".join(html) + "</ul>", "\n".join(md) + "\n"
    raise ValueError(f"Unknown documentation block: {kind}")


def render_all():
    """Produce deterministic files and validate every page, module group, asset and link."""
    content = json.loads((WIKI / "content.json").read_text(encoding="utf-8"))
    pages = content["pages"]
    ids = {page["id"] for page in pages}
    if len(ids) != len(pages) or any(not SAFE_NAME.fullmatch(key) for key in ids):
        raise ValueError("Duplicate or unsafe wiki page ID")
    images = {path.name: path for path in (WIKI / "assets").glob("*.svg")}
    for path in images.values():
        safe_svg(path)
    modules = module_index(content["webModules"])
    files = {}
    for page in pages:
        ru = page["locale"] == "ru"
        blocks = list(page["blocks"])
        if page.get("moduleIndex"):
            for group in content["groups"]:
                rows = [[f"`{row['path']}`", row["summaryRu"] if ru else row["summary"]]
                        for row in modules if row["group"] == group["id"]]
                blocks += [{"type": "h2", "text": group[page["locale"]]}, {"type": "p", "text": group["description"][page["locale"]]},
                           {"type": "table", "headers": ["Источник" if ru else "Source", "Назначение модуля" if ru else "Module responsibility"], "rows": rows}]
        nav = []
        for item in pages:
            if item["locale"] == page["locale"]:
                current = ' aria-current="page"' if item["id"] == page["id"] else ""
                nav.append(f'<a href="{item["id"]}.html"{current}>{escape(item["nav"])}</a>')
        rendered = [block_markup(block, ids, images) for block in blocks]
        switch = page["translation"]
        if switch not in ids:
            raise ValueError("Missing translated wiki page")
        lead = "Документация · локальная версия" if ru else "Documentation · offline edition"
        back = "Вернуться в студию" if ru else "Back to studio"
        skip = "К содержанию" if ru else "Skip to content"
        foot = "Схемы условные. Реальные видео, голоса и лица не включены." if ru else "Schematic examples only. No real videos, voices or faces are included."
        html = f'''<!doctype html>
<html lang="{page['locale']}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(page['title'])} · ag-media-studio</title><link rel="stylesheet" href="wiki.css"></head>
<body><a class="skip" href="#content">{skip}</a><header class="wiki-header" data-ag-width="fill" data-ag-width-min-em="8"><a class="brand" data-ag-width="content" data-ag-width-min-em="1" href="Home{'-ru' if ru else ''}.html">ag-media-studio</a><span>{lead}</span><a href="{switch}.html" lang="{'en' if ru else 'ru'}">{'English' if ru else 'Русский'}</a><a href="/?ui={page['locale']}">{back}</a></header>
<div class="wiki-layout"><nav class="wiki-nav" data-ag-width="fill" data-ag-width-min-em="8" aria-label="{'Разделы документации' if ru else 'Documentation sections'}">{''.join(nav)}</nav>
<main id="content" tabindex="-1" data-ag-width="fill" data-ag-width-min-em="8"><p class="eyebrow">{escape(page['nav'])}</p><h1>{escape(page['title'])}</h1>{''.join(pair[0] for pair in rendered)}<footer>{foot}</footer></main></div></body></html>
'''
        md = f"# {page['title']}\n\n[{'English' if ru else 'Русский'}]({switch}.md) · [ag-media-studio](../../README.md)\n\n" + "\n".join(pair[1] for pair in rendered)
        files[f"site/{page['id']}.html"] = html.encode()
        files[f"{page['id']}.md"] = md.encode()
    files["site/wiki.css"] = (WIKI / "wiki.css").read_bytes()
    for name, path in images.items():
        files[f"site/assets/{name}"] = path.read_bytes()
    index = json.dumps({"schemaVersion": "ag-media-studio-modules/1", "modules": modules}, ensure_ascii=False, indent=2).encode() + b"\n"
    files["module-index.json"] = files["site/module-index.json"] = index
    manifest = {}
    for name in files:
        if not name.startswith("site/"):
            continue
        suffix = Path(name).suffix
        manifest[name[5:]] = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json; charset=utf-8"}[suffix]
    files["site/manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return files


def main():
    """Write generated docs, or fail --check if sources and committed outputs differ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = render_all()
    stale = []
    for name, data in files.items():
        path = WIKI / name
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if stale:
        print("Stale wiki outputs: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(f"Wiki {'verified' if args.check else 'built'}: {len(files)} files; no model execution or network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
