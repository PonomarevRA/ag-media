import {apiText, uiText} from "./i18n.js";
import {mountEditorWindow} from "./editor-object-window.js";
import {ANALYSIS_STAGE_CHOICES} from "./analysis-stage-selection.js";

const retryNames = new Map(ANALYSIS_STAGE_CHOICES);

/** Show a measured step in the same working-window and recovery system as object details. */
export function mountAnalysisStepWindow(find, retry) {
  const panel = find("media-step-window");
  const heading = find("media-step-title");
  const action = find("media-step-retry");
  const window = mountEditorWindow(find, {prefix: "media-step", panel, heading, onClose: () => close(true)});
  let selected;
  let trigger;
  let allowed = false;

  function close(restoreFocus = false) {
    const returnStep = panel.contains(document.activeElement) ? selected?.id : undefined;
    window.clear();
    if (restoreFocus && trigger?.isConnected) trigger.focus({preventScroll: true});
    selected = trigger = undefined;
    return returnStep;
  }

  /** Update only the open or minimized step; unrelated progress preserves window focus. */
  function update(row) {
    if (selected?.id !== row.id) return;
    selected = row;
    const {name, state, status, elapsed, retryGroup, reason, error, lastFailure, progress, windowStartMs} = row;
    heading.textContent = name;
    window.setLabel(name);
    find("media-step-state").textContent = [status, elapsed].filter(Boolean).join(" · ");
    const context = find("media-step-context");
    const windowTime = Number.isFinite(windowStartMs) ? uiText`Окно от ${(windowStartMs / 1000).toFixed(1)} с` : "";
    context.textContent = [progress, windowTime].filter(Boolean).join(" · ");
    context.hidden = !context.textContent;
    const details = find("media-step-diagnostic");
    const message = find("media-step-message");
    const kind = find("media-step-error-type");
    const diagnostic = error || lastFailure;
    const diagnosticTitle = diagnostic && !error ? uiText("Последняя ошибка до восстановления")
      : state === "failed" ? uiText("Ошибка шага") : uiText("Подробности шага");
    find("media-step-diagnostic-title").textContent = diagnosticTitle;
    message.textContent = apiText(diagnostic?.message || reason || "");
    kind.textContent = diagnostic?.type || "";
    kind.hidden = !kind.textContent;
    details.hidden = !message.textContent;
    const group = retryNames.get(retryGroup);
    const groupLabel = find("media-step-retry-group");
    groupLabel.textContent = group ? uiText`Группа запуска · ${uiText(group)}` : "";
    groupLabel.hidden = !group;
    action.hidden = !group;
    action.disabled = !allowed;
    action.textContent = uiText(state === "pending" ? "Запустить этап" : "Пересчитать этап");
  }

  action.onclick = () => {
    if (!allowed || !selected?.retryGroup) return;
    retry([selected.retryGroup], selected.state !== "pending");
  };

  return {
    update,
    close,
    setAvailable(value) {
      allowed = value;
      action.disabled = !allowed;
    },
    open(row, opener) {
      selected = row;
      trigger = opener;
      update(row);
      window.show();
    },
  };
}
