"""Discover registered MCP tools by media domain and task."""

import json
import re
from typing import Literal
from voice_studio.mcp.runtime_info import runtime_info

Category = Literal["sound", "face", "picture", "graph", "ui", "models", "runtime", "transcript"]
DOMAINS = {
    "models": "Model/provider mods, settings, contracts and processing pipelines; not audio-only.",
    "runtime": "Tool discovery and browser execution capabilities; no inferred acceleration.",
    "transcript": "Saved timed transcripts and source-bound editorial changes.",
    "sound": "Speech, speaker turns, dialogue stems, music, effects and cleanup.",
    "face": "Visible speakers, face tracks, people counting, gaze and clothing masks.",
    "picture": "Video frames, shots, objects, text and composition.",
    "graph": "Build and navigate timed media evidence.",
    "ui": "Live Studio window control, recorded interfaces, controls, diagrams, screen OCR and scrolling captures.",
}
PROFILES = {
    "ui_windows": (("ui",), "live application window browser native MCP Bridge интерфейс окно управление"),
    "ui_inspect": (("ui",), "live interface page controls state inspect состояние интерфейс окно"),
    "ui_execute": (("ui",), "live interface navigation seek command bridge управление таймкод"),
    "dialogue_audit": (("sound", "graph"), "dialogue boundaries speaker turns mixed phrases missing speech audit границы реплики проверка диалог"),
    "dialogue_audit_job": (("sound", "graph"), "audit status очередь проверка диалог"),
    "speaker_profiles": (("sound", "face"), "digital voice profile audio video anchors reference голос профиль якоря говорящий"),
    "browser_ml_capabilities": (("sound", "face", "picture", "ui"), "WebNN browser acceleration cosine embedding векторы ускорение браузер"),
    "screen_scroll": (("picture", "ui"), "screenshot scrolling stitch mosaic OCR скриншот прокрутка мозаика"),
    "screen_analyze": (("picture", "ui"), "screenshot UI interface button form diagram OCR скриншот интерфейс кнопка поле схема черновик"),
    "audio_separate": (("sound",), "stems dialogue music effects разделение дорожки реплики музыка звуки шумодав"),
    "audio_denoise": (("sound",), "cleanup noise rnnoise шумодав шумоподавление чистка"),
    "audio_job": (("sound",), "status stems denoise очередь дорожки шумоподавление"),
    "media_transcribe": (("sound", "face", "picture"),
                         "speech speakers gaze objects ocr транскрипция спикеры лица предметы подсчёт людей"),
    "media_transcribe_stream": (("sound",), "speech hls rtsp поток транскрипция"),
    "video_clothing_masks": (("face", "picture"), "segment garment тело лицо одежда маска"),
    "media_graph_build": (("graph", "sound", "face", "picture", "ui"), "analyze scene layers объект лицо звук граф подсчёт людей"),
    "media_graph_frame": (("graph", "face", "picture", "ui"), "frame screenshot overlay кадр скриншот рамка взгляд"),
    "media_graph_interval": (("graph", "sound", "face", "picture", "ui"),
                             "interval lossless compact expanded tables facts doubts grid разобрать интервал сводка сомнения сетка кадры"),
    "media_graph_audio": (("graph", "sound"), "clip audio speech вырезать звук реплика"),
    "media_graph_object_placement": (("graph", "picture"), "placement tracking objects предмет масштаб"),
    "media_graph_overview": (("graph", "ui"), "counts coverage interface интерфейс покрытие"),
    "media_graph_node": (("graph", "ui"), "node relations interface controls узел связи интерфейс"),
    "media_graph_timeline": (("graph", "sound", "face", "picture", "ui"),
                             "track timecoded layer camera motion depth таймлайн дорожки камера движение глубина"),
    "media_graph_search": (("graph", "sound", "face", "picture", "ui"), "find objects speakers cues поиск"),
}


PROFILES.update({
    "media_catalog": (("runtime",), "catalogue categories tools graph discover каталог категории инструменты"),
    "ui_measure": (("ui",), "live measure geometry padding gap alignment text inset batch измерение отступы текст кнопка массив"),
    "speech_sources": (("sound",), "speech synthesis engines reference recordings голос синтез источники"),
    "speech_generate": (("sound",), "speech synthesis generation voice clone синтез озвучка голос"),
    "speech_job": (("sound",), "speech synthesis generation status WAV озвучка готовность"),
    "media_graph_job": (("graph",), "graph analysis progress status граф статус анализ"),
    "media_graph_list": (("graph",), "saved immutable media graphs граф список"),
    "transcript_materials": (("transcript", "sound"), "saved editable transcripts список транскрипции"),
    "transcript_get": (("transcript", "sound"), "timed transcript cues revision текст реплики таймкоды"),
    "transcript_edit": (("transcript", "sound"), "edit transcript speaker revision исправить текст спикер"),
})
for _action in ("pipeline", "contracts", "list", "rescan", "enable", "select", "configure", "template", "test"):
    PROFILES["media_mods_" + _action] = (("models",), "model mod provider contract settings pipeline модель мод настройки провайдер контракт пайплайн")
# Browser math/probing is a runtime capability; speaker evidence has separate domain tools.
PROFILES["browser_ml_capabilities"] = (("runtime",), "browser WebNN execution probe cosine браузер выполнение WebNN векторы")


def tool_scope(name):
    """Separate live UI, recorded media and model-management operations even when categories overlap."""
    scopes = {"ui_": "live-window", "screen_": "local-screenshot", "media_graph_": "timed-media-graph",
              "media_mods_": "model-management", "transcript_": "saved-transcript", "audio_": "source-audio",
              "speech_": "speech-synthesis", "dialogue_": "dialogue-audit", "video_": "source-video"}
    for prefix, scope in scopes.items():
        if name.startswith(prefix):
            return scope
    if name in {"media_catalog", "browser_ml_capabilities"}:
        return "runtime"
    if name.startswith(("media_transcribe", "speaker_")):
        return "source-media"
    return "unknown"


def tool_entry(tool, include_schema):
    """Decorate the live tool schema with discoverability metadata."""
    canonical = tool.name
    categories, keywords = PROFILES.get(canonical, ((), ""))
    row = {"name": tool.name, "description": tool.description or "", "categories": list(categories)}
    row["scope"] = tool_scope(canonical)
    if include_schema:
        row["inputSchema"] = tool.inputSchema
    searchable = re.sub(r"[_-]", " ", " ".join((tool.name, canonical, row["description"], *categories, keywords))).casefold()
    return row, searchable


async def read_catalog(mcp, category=None, query="", include_schema=False, include_graph=False):
    """Read actual registrations and filter across categories and keywords."""
    tokens = re.findall(r"[^\W_]+", query.casefold())
    entries = []
    for tool in sorted(await mcp.list_tools(), key=lambda item: item.name):
        row, searchable = tool_entry(tool, include_schema)
        if (category is None or category in row["categories"]) and all(word in searchable for word in tokens):
            entries.append(row)
    result = {"categories": DOMAINS, "count": len(entries), "tools": entries, "runtime": runtime_info(),
              "namingContract": "ag-media-mcp/3",
              "uncategorized": [row["name"] for row in entries if not row["categories"]]}
    if include_graph:
        result["graph"] = {"schemaVersion": "ag-media/tool-category-graph/1",
                           "tools": [{"id": row["name"], "scope": row["scope"]} for row in entries],
                           "categories": [{"id": name, "description": description} for name, description in DOMAINS.items()],
                           "links": [{"toolId": row["name"], "categoryId": group} for row in entries for group in row["categories"]]}
    return result


def register_catalog(mcp):
    """Publish a searchable MCP tool and a discoverable JSON resource."""
    @mcp.tool()
    async def media_catalog(category: Category | None = None, query: str = "",
                            include_schema: bool = False, include_graph: bool = False) -> dict:
        """Discover actual tools by sound,face,picture,graph,ui,models,runtime or transcript and keywords. Optionally return the canonical tool/category graph; uncategorized registrations are explicit."""
        return await read_catalog(mcp, category, query, include_schema, include_graph)

    @mcp.resource("ag-media-studio://catalog", mime_type="application/json",
                  description="Current ag-media-studio tool catalogue.")
    async def media_catalog_resource() -> str:
        """Read the same tool inventory from clients that support MCP resources."""
        return json.dumps(await read_catalog(mcp), ensure_ascii=False)
