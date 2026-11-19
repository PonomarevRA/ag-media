import {uiText} from "./i18n.js";
import {UI_TOOL_NAMES} from "./tool-names.js";

export const UI_BRIDGE_TOOLS = UI_TOOL_NAMES;

/** Fence commands against a document, route, speech workspace and report replacement. */
export function createUiBridge(tools, ui, readRoute = () => location.pathname + location.search + location.hash) {
  const selected = new Map(tools.filter(tool => UI_BRIDGE_TOOLS.has(tool.name)).map(tool => [tool.name, tool]));
  let revision = crypto.randomUUID().replaceAll("-", "");
  let priorSegments, priorScope;
  let running = false;

  /** Fence source replacement, cue growth and explicit revisions without hashing transcript contents. */
  function inspect() {
    const state = ui.inspect();
    const report = ui.transcript();
    // Live capture returns a fresh report wrapper around the same cue collection on each read.
    // The host's UI epoch fences in-place edits without changing the material's persisted CAS revision.
    const segments = report?.segments;
    const scope = JSON.stringify([readRoute(), state.speechLanguage, report?.id,
      report?.sourceDigest, report?.revision, segments?.length, ui.contextRevision?.()]);
    if (segments !== priorSegments || scope !== priorScope) {
      revision = crypto.randomUUID().replaceAll("-", "");
      priorSegments = segments;
      priorScope = scope;
    }
    return {revision, ui: state};
  }

  /** Reuse WebMCP input validation and handlers; never interpret supplied source code or selectors. */
  async function execute(command, signal) {
    const state = inspect();
    // A geometric read must not append unrelated transcript/recording/form state to its result.
    const context = current => command?.name === "ui_measure" ? {revision: current.revision} : current;
    if (signal.aborted) return {status: "error", error: "ui_bridge_disconnected"};
    if (!command || !selected.has(command.name)) return {status: "error", error: "ui_tool_not_allowed"};
    if (!Number.isFinite(command.expiresAtMs) || Date.now() >= command.expiresAtMs) return {status: "error", error: "ui_command_expired"};
    if (command.name !== "ui_inspect" && command.revision !== state.revision) {
      return {status: "error", error: "ui_context_changed", ...context(state)};
    }
    if (running) return {status: "error", error: "ui_window_busy"};
    running = true;
    const timeout = new AbortController();
    const cancel = () => timeout.abort();
    signal.addEventListener("abort", cancel, {once: true});
    const timer = setTimeout(cancel, Math.max(0, command.expiresAtMs - Date.now()));
    try {
      const result = await selected.get(command.name).execute(command.arguments ?? {}, {signal: timeout.signal});
      timeout.signal.throwIfAborted();
      return {status: result?.error ? "error" : "ok", result, ...context(inspect())};
    } catch {
      return {status: "error", error: "ui_action_interrupted_outcome_unknown", ...context(inspect())};
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", cancel);
      running = false;
    }
  }

  const catalogue = [...selected.values()].map(({name, description, inputSchema}) => ({name, description, inputSchema}));
  return {inspect, execute, catalogue};
}

/** Connect this page over an idle long poll; disabling or leaving revokes its channel. */
export function mountMcpBridge(find, tools, ui, lifecycle) {
  const enabled = find("mcp-bridge-enabled");
  const status = find("mcp-bridge-status");
  const bridge = createUiBridge(tools, ui);
  let active;
  try { if (sessionStorage.getItem("ag-media-ui-bridge") === "off") enabled.checked = false; }
  catch { /* A private/embedded browser may disable storage; the checkbox still works. */ }
  window.agMediaStudioMcpBridge = Object.freeze({version: 1, inspect: bridge.inspect});

  /** Ignore obsolete lifecycle callbacks and avoid repeating identical live announcements. */
  function display(message) {
    if (!lifecycle.aborted && status.textContent !== message) status.textContent = message;
  }

  /** Use the existing same-origin boundary without access to another page's private token. */
  async function request(operation, data, signal, keepalive = false) {
    const response = await fetch("/api/ui-bridge/" + operation, {method: "POST", signal, keepalive,
      headers: {"Content-Type": "application/json", "X-Voice-Studio": "1"}, body: JSON.stringify(data)});
    if (!response.ok) throw Error("ui_bridge_http_" + response.status);
    return response.json();
  }

  /** Stop pending requests promptly without leaving a retry timer alive. */
  function pause(ms, signal) {
    return new Promise(resolve => {
      const complete = () => { clearTimeout(timer); signal.removeEventListener("abort", complete); resolve(); };
      const timer = setTimeout(complete, ms);
      signal.addEventListener("abort", complete, {once: true});
      if (signal.aborted) complete();
    });
  }

  /** Correlate one result, avoid replay, and re-register only after a transport/session loss. */
  async function connect(controller) {
    const {signal} = controller;
    let channel, completed, failures = 0;
    try {
      while (!signal.aborted) {
        try {
          if (!channel) {
            channel = await request("register", {tools: bridge.catalogue, state: bridge.inspect()}, signal);
            if (signal.aborted) break;
          }
          display(uiText`MCP Bridge подключён · окно ${channel.windowId.slice(0, 8)}`);
          const reply = await request("poll", {...channel, state: bridge.inspect(), completed}, signal);
          completed = undefined;
          failures = 0;
          if (reply.command) {
            display(uiText`MCP Bridge · ${reply.command.name}`);
            let result = await bridge.execute(reply.command, signal);
            if (new TextEncoder().encode(JSON.stringify(result)).length > 48_000) {
              result = {status: "error", error: "ui_result_too_large"};
            }
            completed = {id: reply.command.id, result};
          }
        } catch (error) {
          if (signal.aborted) break;
          // An uncertain acknowledgement is never replayed against a new window identity.
          if (error.message === "ui_bridge_http_404") { channel = undefined; completed = undefined; }
          display(uiText("MCP Bridge: ожидаем подключение к студии…"));
          await pause(Math.min(15_000, 1000 * 2 ** Math.min(failures++, 4)), signal);
        }
      }
    } finally {
      if (channel) await request("disconnect", channel, undefined, true).catch(() => {});
      if (active === controller) display(uiText("MCP Bridge отключён для этого окна."));
    }
  }

  /** Treat the visible checkbox as immediate permission to connect or revoke this page. */
  function change() {
    active?.abort();
    try { sessionStorage.setItem("ag-media-ui-bridge", enabled.checked ? "on" : "off"); }
    catch { /* Connection permission remains scoped to this page when storage is unavailable. */ }
    if (!enabled.checked || lifecycle.aborted) {
      display(uiText("MCP Bridge отключён для этого окна."));
      return;
    }
    active = new AbortController();
    void connect(active);
  }
  enabled.addEventListener("change", change);
  lifecycle.addEventListener("abort", () => {
    enabled.removeEventListener("change", change);
    active?.abort();
  }, {once: true});
  change();
}
