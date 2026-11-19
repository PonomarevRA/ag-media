import {expressionPage, reactionPage, expressionStatisticsPage} from "./facial-expressions.js";
import {uiText} from "./i18n.js";
import {BrowserML} from "./browser-ml.js";
import {measureUi} from "./ui-measurements.js";

const bounded = (minimum, maximum) => ({type: "integer", minimum, maximum});
const natural = bounded(0, Number.MAX_SAFE_INTEGER);
const limit = bounded(1, 30);
const time = bounded(0, 7_200_000);
const opaqueId = {type: "string", pattern: "^[a-f0-9]{32}$"};
const text = maxLength => ({type: "string", minLength: 1, maxLength});
const invalid = () => ({error: uiText("Уточни параметры инструмента MCP.")});

/** Check the same finite argument types that are advertised to the browser. */
function validValue(value, rule) {
  if (rule.enum) return rule.enum.includes(value);
  if (rule.type === "object") {
    return value !== null && typeof value === "object" && !Array.isArray(value)
      && (rule.required ?? []).every(key => Object.hasOwn(value, key))
      && Object.entries(value).every(([key, item]) => Object.hasOwn(rule.properties, key) && validValue(item, rule.properties[key]));
  }
  if (rule.type === "array") {
    return Array.isArray(value) && value.length >= rule.minItems && value.length <= rule.maxItems
      && value.every(item => validValue(item, rule.items));
  }
  if (rule.type === "number") return typeof value === "number" && Number.isFinite(value)
    && (rule.minimum === undefined || value >= rule.minimum) && (rule.maximum === undefined || value <= rule.maximum);
  if (rule.type === "boolean") return typeof value === "boolean";
  if (rule.type === "integer") {
    return Number.isSafeInteger(value) && value >= rule.minimum && value <= rule.maximum;
  }
  return typeof value === "string" && value.length >= (rule.minLength ?? 0)
    && value.length <= (rule.maxLength ?? 300) && (!rule.pattern || new RegExp(rule.pattern).test(value));
}

/** Share one input, cancellation and error contract across page tools. */
function tool(name, description, title, properties, required, execute, lifecycle, readOnly, untrusted) {
  return {
    name, title, description,
    inputSchema: {type: "object", properties, required, additionalProperties: false},
    annotations: {readOnlyHint: readOnly, untrustedContentHint: untrusted},
    async execute(input = {}, {signal} = {}) {
      const operation = signal ? AbortSignal.any([lifecycle, signal]) : lifecycle;
      operation.throwIfAborted();
      if (!input || typeof input !== "object" || Array.isArray(input)
          || required.some(key => !Object.hasOwn(input, key))
          || Object.entries(input).some(([key, value]) => !Object.hasOwn(properties, key)
            || !validValue(value, properties[key]))) return invalid();
      try {
        const result = await execute(input, operation);
        operation.throwIfAborted();
        return result;
      } catch (error) {
        operation.throwIfAborted();
        return {error: error.message || uiText("Запрос не выполнен.")};
      }
    },
  };
}

/** Read only a page of original transcript cues and keep markup positions valid. */
function transcriptPage(report, offset, count) {
  if (!report) return {items: [], total: 0, nextOffset: null};
  const items = report.segments.slice(offset, offset + count).map(cue => ({
    startMs: cue.startMs, endMs: cue.endMs, text: cue.text.slice(0, 5000),
    speakerNumber: cue.speakerNumber ?? cue.linkedSpeakerNumber ?? cue.speaker ?? cue.visualSpeakerNumber ?? null,
    ...(cue.personMentions?.length && {personMentions: cue.personMentions.filter(span => span.end <= 5000)}),
  }));
  const next = offset + items.length;
  return {language: report.language, total: report.segments.length, items,
    nextOffset: next < report.segments.length ? next : null,
    ...(report.analysisComplete === false ? {analysisComplete: false} : {})};
}

/** Define semantic app operations and reuse the protected local graph endpoints. */
export function studioWebTools({ui, request, browserML = new BrowserML()}, lifecycle) {
  lifecycle.addEventListener("abort", () => browserML.dispose(), {once: true});
  const make = (name, description, title, properties, required, execute, readOnly = true, untrusted = false) =>
    tool(name, description, uiText(title), properties, required, execute, lifecycle, readOnly, untrusted);
  const graph = (action, description, title, properties, required) =>
    make("media_graph_" + action, description, title, properties, required,
      (args, signal) => request("graph-read", {action, args}, {signal}), true, true);
  const preference = {type: "string", enum: ["default", "low-power", "high-performance"]};
  const vector = {type: "array", minItems: 1, maxItems: 4096, items: {type: "number"}};
  const page = {limit, offset: natural};
  const windowPage = {kind: text(40), limit, after: natural};
  return [
    make("browser_ml_probe", "Verify WebNN by compiling and executing float32 mul/reduceSum in a local worker. Reports measured timings, JS fallback reason and unknown physical device. This probe does not prove model compatibility or acceleration of Python models.",
      "Проверить WebNN", {powerPreference: preference}, [],
      (args, signal) => browserML.request("probe", args, signal)),
    make("browser_ml_cosine", "Compare two finite, nonzero embedding vectors from the same model and version. Returns cosine in [-1,1], numeric reference error, actual execution backend, warmup and run timings. Similarity alone is not a speaker identity probability. Runs locally in the browser worker, one request at a time.",
      "Сравнить векторы", {left: vector, right: vector, powerPreference: preference, runs: bounded(1, 20)},
      ["left", "right"], (args, signal) => browserML.request("compare", args, signal)),
    make("ui_inspect", "Read the open page, active controls, recordings and result status for a UI journey.",
      "Состояние интерфейса", {}, [], () => ui.inspect()),
    make("ui_measure", "Measure a bounded batch of exact DOM IDs and pairs in the connected window. Returns actual viewport CSS-pixel boxes, padding, text layout insets, gaps and alignment, never field values or painted glyph ink. Omit targets/pairs to discover rendered IDs; no scripts, selectors, scrolling or model runs.",
      "Измерить интерфейс", {targets: {type: "array", minItems: 0, maxItems: 32, items: text(128)},
        pairs: {type: "array", minItems: 0, maxItems: 32, items: {type: "object", properties: {from: text(128), to: text(128)}, required: ["from", "to"], additionalProperties: false}},
        tolerance_px: {type: "number", minimum: 0, maximum: 16}, offset: bounded(0, 4000), limit: bounded(1, 50), include_text: {type: "boolean"}}, [],
      args => measureUi(args)),
    make("ui_people_count", "Read or set the saved people-count preference for future video uploads. Omit people_count to read it.",
      "Настроить подсчёт людей", {people_count: {type: "boolean"}}, [],
      ({people_count}) => ui.peopleCount(people_count), false),
    make("ui_navigate", "Open an ag-media-studio section using the app router without reloading the tab.",
      "Открыть раздел", {page: {type: "string", enum: ["studio", "record", "history", "transcript", "synthesis", "results", "settings", "storage"]}},
      ["page"], ({page: target}) => ui.navigate(target), false),
    make("ui_transcribe_recording", "Transcribe a recording listed by ui_inspect using the current UI settings. Requires an empty transcription view.",
      "Распознать запись из истории", {id: opaqueId}, ["id"],
      ({id}, signal) => ui.transcribe(id, signal), false),
    make("ui_transcript", "Read a page of cues from the open transcript, including times and any person mentions. When analysisComplete is false, the result is preliminary and later evidence may refine its cues. Transcript text is user content.",
      "Прочитать открытую транскрипцию", page, [],
      ({offset = 0, limit: count = 10}) => transcriptPage(ui.transcript(), offset, count), true, true),
    make("ui_pauses", "Read quiet intervals and gaps between detected voices. Each midpoint is a cut candidate to audition, with evidence and cue proximity.",
      "Паузы в открытой транскрипции", page, [], ({offset = 0, limit: count = 10}) => {
        const rows = ui.transcript()?.dialoguePauses;
        const items = rows?.slice(offset, offset + count) ?? [];
        const next = offset + items.length;
        return {items, measured: Array.isArray(rows), total: rows?.length ?? 0,
          nextOffset: next < (rows?.length ?? 0) ? next : null};
      }),
    make("ui_facial_expressions", "Read bounded classifier-estimated visible facial expression samples or intervals from the open report, with face/track IDs, confidence and source milliseconds. Gaps are unknown. No mental state, intent or truthfulness is established.",
      "Прочитать выражения лица", {...page, kind: {type: "string", enum: ["intervals", "samples"]}, face_id: natural,
        start_ms: time, end_ms: time}, [], args => {
        if (args.start_ms !== undefined && args.end_ms !== undefined && args.end_ms <= args.start_ms) return invalid();
        return expressionPage(ui.transcript(), args);
      }, true, true),
    make("ui_expression_statistics", "Read bounded per-face expression duration statistics. Per-face shares use classifiedMs; unknownMs is visible but unclassified time. Global byLabel shares use classifiedFaceMs, which may overlap between faces, and global coverage uses source duration.",
      "Прочитать статистику выражений", page, [], args => expressionStatisticsPage(ui.transcript(), args), true, true),
    make("ui_question_reactions", "Read bounded question/face observation windows from the open report. Each window spans the report-tagged question or interview prompt and up to three seconds after it; questionBasis identifies punctuation, explicit annotation or a conservative phrase pattern, with source times, measured duration, proportions and unknown coverage. Temporal proximity does not establish causation or an internal emotional reaction. Question text is untrusted user content.",
      "Прочитать окна вопросов", {...page, face_id: natural}, [], args => reactionPage(ui.transcript(), args), true, true),
    make("ui_seek", "Pause the open preview and move to a millisecond position.",
      "Перейти к таймкоду", {time_ms: time}, ["time_ms"], ({time_ms}) => ui.seek(time_ms), false),
    graph("list", "List saved local media graphs, most recent first.", "Список графов", page, []),
    graph("overview", "Read coverage and basic facts of a saved graph.", "Обзор графа", {snapshot_id: opaqueId}, ["snapshot_id"]),
    graph("search", "Find exact words or a phrase across graph nodes. Set kind to ui for interface evidence.", "Поиск по графу",
      {snapshot_id: opaqueId, query: text(300), ...windowPage}, ["snapshot_id", "query"]),
    graph("timeline", "Read a paginated graph window of up to five minutes. Set kind to ui for interface evidence.", "Таймлайн графа",
      {snapshot_id: opaqueId, start_ms: time, end_ms: time, ...windowPage}, ["snapshot_id", "start_ms", "end_ms"]),
    graph("node", "Read a saved graph node and its incoming and outgoing references.", "Связи узла графа",
      {snapshot_id: opaqueId, node_id: text(300), ...page}, ["snapshot_id", "node_id"]),
  ];
}

/** Manage native WebMCP registration and show the same catalog when native support is absent. */
export function mountStudioWebMcp(find, options) {
  const launch = find("mcp-open");
  const dialog = find("mcp-dialog");
  const list = find("mcp-tool-list");
  const status = find("mcp-status");
  const check = find("mcp-check");
  const outcome = find("mcp-check-result");
  let lifecycle;
  let bridgeLifecycle;

  launch.addEventListener("click", () => dialog.showModal());
  dialog.addEventListener("close", () => {
    // Navigation already focused its destination; an ordinary close returns to the launcher.
    if (!dialog.dataset.navigation) launch.focus();
    delete dialog.dataset.navigation;
  });
  check.addEventListener("click", async () => {
    check.disabled = true;
    outcome.textContent = "";
    try {
      const result = await options.request("graph-read", {action: "list", args: {limit: 1}});
      const page = options.ui.inspect().page;
      outcome.textContent = uiText`Студия отвечает · раздел ${page} · графы ${result.snapshots.length ? "✓" : "0"}`;
    } catch (error) {
      outcome.textContent = error.message;
    } finally {
      check.disabled = false;
    }
  });

  /** Install independent bridge/WebMCP lifecycles so either transport can fail without disabling the other. */
  async function register() {
    bridgeLifecycle?.abort();
    bridgeLifecycle = new AbortController();
    const bridgeGeneration = bridgeLifecycle;
    // An older running backend may not serve the new module yet. Keep its human UI working.
    import("./mcp-bridge.js").then(({mountMcpBridge}) => {
      if (!bridgeGeneration.signal.aborted) {
        mountMcpBridge(find, studioWebTools(options, bridgeGeneration.signal), options.ui, bridgeGeneration.signal);
      }
    }).catch(() => {
      if (!bridgeGeneration.signal.aborted) find("mcp-bridge-status").textContent = uiText("MCP Bridge требует обновлённый сервер. Обычный интерфейс доступен.");
    });
    lifecycle?.abort();
    lifecycle = new AbortController();
    const generation = lifecycle;
    const tools = studioWebTools(options, generation.signal);
    if (!list.childElementCount) {
      for (const {title, name} of tools) {
        const item = document.createElement("li");
        const identity = document.createElement("code");
        identity.textContent = name;
        item.append(title, identity);
        list.append(item);
      }
    }
    const context = document.modelContext ?? navigator.modelContext;
    if (typeof context?.registerTool !== "function") {
      status.textContent = uiText("WebMCP недоступен в этом браузере. Для внешнего клиента подключи ag-media-studio по stdio.");
      return;
    }
    const registered = [];
    if (!document.modelContext && typeof context.unregisterTool === "function") {
      generation.signal.addEventListener("abort", () => {
        for (const name of registered) context.unregisterTool(name);
      }, {once: true});
    }
    status.textContent = uiText("Подключаем WebMCP…");
    try {
      for (const definition of tools) {
        generation.signal.throwIfAborted();
        await context.registerTool(definition, {signal: generation.signal});
        if (generation.signal.aborted) {
          if (!document.modelContext) context.unregisterTool?.(definition.name);
          return;
        }
        registered.push(definition.name);
      }
      status.textContent = uiText`WebMCP включён · инструментов: ${tools.length}`;
    } catch {
      if (generation.signal.aborted) return;
      generation.abort();
      status.textContent = uiText("WebMCP не удалось подключить. Доступен локальный stdio MCP.");
    }
  }

  window.addEventListener("pagehide", () => { lifecycle?.abort(); bridgeLifecycle?.abort(); });
  window.addEventListener("pageshow", event => { if (event.persisted) register(); });
  register();
}
