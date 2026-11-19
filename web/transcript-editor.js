import {uiText} from "./i18n.js";
/** Open and save the shared local materials used by the browser and MCP. */
export function mountTranscriptEditor(find, request, view) {
  const root = find("transcript-materials");
  const list = find("transcript-material-list");
  const openButton = find("transcript-material-open");
  const refreshButton = find("transcript-material-refresh");
  const save = find("transcript-material-save");
  const current = find("transcript-current");
  const status = find("transcript-material-status");
  let sourceBusy = false;
  let listLanguage;
  let pending = false;
  let sourceGeneration = 0;

  /** Accept a completion only while the host retains its exact source and operation generation. */
  function ownsSource(source, generation) {
    return generation === sourceGeneration && source === view.current();
  }

  function update() {
    const data = view.current();
    const blocked = sourceBusy || pending;
    current.hidden = !data?.id;
    current.textContent = data?.id ? uiText`${data.title} · версия ${data.revision}` : "";
    current.title = current.textContent;
    save.hidden = !data?.segments.length || Boolean(data.id);
    save.disabled = blocked;
    list.disabled = refreshButton.disabled = blocked;
    openButton.disabled = blocked || !list.value;
  }

  /** Refresh saved choices without letting an obsolete source load clear the active status. */
  async function refresh() {
    if (sourceBusy || pending) return;
    const source = view.current(), generation = sourceGeneration;
    try {
      const previous = list.value;
      const language = view.current()?.language;
      const {items} = await request("transcript-materials", undefined, {language});
      if (!ownsSource(source, generation)) return;
      listLanguage = language;
      list.replaceChildren(new Option(items.length ? uiText("Выбери материал") : uiText("Материалов пока нет"), ""));
      for (const item of items) list.add(new Option(uiText`${item.title} · версия ${item.revision}`, item.id));
      if (items.some(item => item.id === previous)) list.value = previous;
      status.textContent = "";
    } catch (error) {
      if (ownsSource(source, generation)) status.textContent = error.message;
    }
    update();
  }

  /** Open one saved material only if no newer host source supersedes this read. */
  async function open(id, language = listLanguage) {
    if (!id || sourceBusy || pending) return;
    const source = view.current(), generation = sourceGeneration;
    pending = true;
    update();
    status.textContent = "";
    try {
      const data = await request("transcript-material", undefined, {id, language});
      if (!ownsSource(source, generation)) return;
      view.apply(data, source?.id === id);
      status.textContent = uiText`Открыта версия ${data.revision}`;
      return data;
    } catch (error) {
      if (ownsSource(source, generation)) {
        status.textContent = error.message;
        root.open = true;
      }
    } finally {
      pending = false;
      update();
    }
  }

  save.addEventListener("click", async () => {
    const draft = view.current();
    if (sourceBusy || pending || !draft?.segments.length || draft.id) return;
    const generation = sourceGeneration;
    let refreshList = false;
    pending = true;
    update();
    status.textContent = "";
    try {
      const data = await request("transcript-material", {
        title: view.title(), segments: draft.segments, sourceDigest: draft.sourceDigest,
        dialoguePauses: draft.dialoguePauses,
      }, {language: draft.language});
      // Persistence completes even when a new source owns the visible workbench.
      if (ownsSource(draft, generation)) {
        view.apply(data, true);
        root.open = true;
        refreshList = true;
      }
      return data;
    } catch (error) {
      if (ownsSource(draft, generation)) {
        status.textContent = error.message;
        root.open = true;
      }
    } finally {
      pending = false;
      if (refreshList) await refresh();
      else update();
    }
  });
  openButton.addEventListener("click", () => open(list.value));
  refreshButton.addEventListener("click", refresh);
  list.addEventListener("change", update);
  root.addEventListener("toggle", () => { if (root.open) refresh(); });

  return {
    open,
    update,
    setBusy(value) {
      if (value && !sourceBusy) sourceGeneration++;
      sourceBusy = Boolean(value);
      update();
    },
    /** Persist a revision-checked cue edit, applying its result only to the source that submitted it. */
    async patch(startMs, changes) {
      if (sourceBusy || pending) throw new Error(uiText("Дождись завершения текущей операции."));
      const source = view.current(), generation = sourceGeneration;
      const {id, revision, language} = source;
      pending = true;
      update();
      try {
        const data = await request("transcript-cue", {id, revision, startMs, ...changes}, {language});
        if (!ownsSource(source, generation)) return data;
        view.apply(data, true);
        const summary = document.getElementById("cue-edit-" + startMs);
        const messageDetails = summary?.closest(".transcript-message-details");
        if (messageDetails) messageDetails.open = true;
        summary?.focus();
        return data;
      } finally {
        pending = false;
        update();
      }
    },
  };
}

/** Render a safely editable cue with an optional preceding quote as evidence for the speaker number. */
export function mountCueEditor(cue, cues, patch) {
  const root = document.createElement("details");
  root.className = "transcript-cue-editor";
  const summary = document.createElement("summary");
  summary.id = "cue-edit-" + cue.startMs;
  summary.textContent = uiText("Править реплику");
  root.append(summary);
  const build = () => {
    if (!root.open) return;
    root.removeEventListener("toggle", build);
    root.append(cueForm(cue, cues, patch));
  };
  root.addEventListener("toggle", build);
  return root;
}

/** Build the form only on demand to keep large transcripts responsive. */
function cueForm(cue, cues, patch) {
  const form = document.createElement("form");
  const number = document.createElement("input");
  number.type = "number";
  number.min = "1";
  number.max = "1000";
  number.step = "1";
  number.placeholder = String(cue.speaker ?? cue.visualSpeakerNumber ?? "");
  number.value = cue.speakerNumber ?? "";
  const words = document.createElement("textarea");
  words.dir = "auto";
  words.rows = 3;
  words.maxLength = 5000;
  words.required = true;
  words.value = cue.text;
  const evidence = document.createElement("select");
  const previous = cue.speakerNumberEvidence?.contextStartMs;
  evidence.add(new Option(uiText("Без ссылки на другую реплику"), ""));
  const recent = cues.filter(other => other.startMs < cue.startMs).slice(-3);
  if (previous !== undefined && !recent.some(other => other.startMs === previous)) {
    const selected = cues.find(other => other.startMs === previous);
    if (selected) recent.unshift(selected);
  }
  for (const other of recent) {
    const excerpt = other.text.length > 105 ? other.text.slice(0, 104) + "…" : other.text;
    evidence.add(new Option(uiText`${(other.startMs / 1000).toFixed(2)} с · ${excerpt}`, String(other.startMs)));
  }
  evidence.value = previous === undefined ? "" : String(previous);
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.textContent = uiText("Сохранить правку");
  const status = document.createElement("p");
  status.className = "muted";
  status.setAttribute("role", "status");
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const changes = {};
    const chosen = number.value ? number.valueAsNumber : null;
    const selectedEvidence = evidence.value ? Number(evidence.value) : null;
    if (chosen !== (cue.speakerNumber ?? null) || (chosen && selectedEvidence !== (previous ?? null))) {
      changes.speakerNumber = chosen;
      changes.contextStartMs = selectedEvidence;
    }
    if (words.value.trim() !== cue.text) changes.text = words.value;
    if (!Object.keys(changes).length) {
      status.textContent = uiText("Изменений нет.");
      return;
    }
    submit.disabled = true;
    status.textContent = "";
    try {
      await patch(cue.startMs, changes);
    } catch (error) {
      status.textContent = error.message;
    } finally {
      submit.disabled = false;
    }
  });
  for (const [title, control] of [[uiText("Номер спикера"), number], [uiText("Реплика-основание"), evidence], [uiText("Текст"), words]]) {
    const label = document.createElement("label");
    label.className = "select-field";
    label.append(title, control);
    form.append(label);
  }
  form.append(submit, status);
  return form;
}
