import {uiText} from "./i18n.js";
const LABELS = {
  "lapshi-f5": uiText("F5 · обычный"),
  "lapshi-accent": "F5 · accent-tune",
  "lapshi-coqui": "Coqui XTTS",
};

/** Return the model label for a controlled pronunciation trial. */
export function stressTakeLabel(variant) {
  return LABELS[variant];
}

/** Show one original and matching model variants per seed, with a browser-local listening choice. */
export function mountStressComparison({request, refresh, act, compare}) {
  const panel = document.querySelector("#stress-panel");
  const rounds = document.querySelector("#stress-rounds");
  const status = document.querySelector("#stress-status");
  const trigger = document.querySelector("#stress-run");
  let signature;

  /** Make a radio choice and an explicit control for opening the shared A/B deck. */
  function addTake(list, row, original, name, preference) {
    const group = document.createElement("div");
    group.className = "stress-take";
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = preference;
    input.value = row.id;
    input.checked = localStorage.getItem(preference) === row.id;
    input.disabled = row.status !== "done";
    input.onchange = () => localStorage.setItem(preference, row.id);
    label.append(input, document.createTextNode(name));
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = row.status === "done" ? uiText("A/B с исходником")
      : {queued: uiText("В очереди"), running: uiText("Синтезируется"), failed: uiText("Ошибка")}[row.status];
    button.disabled = row.status !== "done" || !original;
    button.setAttribute("aria-label", `${button.textContent} · ${name}`);
    button.onclick = () => compare(original.id, uiText("Твоя запись"), row.id, name);
    group.append(label, button);
    list.append(group);
  }

  /** Refresh groups only when visible results or available sources changed. */
  return function render(data, language) {
    const all = data.jobs.filter(job => stressTakeLabel(job.variant));
    const latest = all[0];
    // Existing studio results predate the batch column and belong to one initial run.
    const candidates = latest ? all.filter(job => job.comparison_id === latest.comparison_id &&
      (job.batch_id || null) === (latest.batch_id || null)) : [];
    const original = data.recordings.find(row => /(?:^|[^\p{L}\p{N}_])лапши(?=$|[^\p{L}\p{N}_])/iu.test(row.text));
    const reference = data.recordings.find(row => row.id !== original?.id && !row.evaluation_only && row.seconds >= 3);
    panel.hidden = language !== "ru" || (!original && !candidates.length);
    const active = candidates.some(job => job.status === "queued" || job.status === "running");
    const complete = candidates.length === 9 && candidates.every(job => job.status === "done") &&
      latest.comparison_id === original?.id;
    trigger.hidden = !data.stressComparisonReady || !original || !reference || active || complete;
    trigger.onclick = () => act(async () => {
      await request("compare-lapshi", {comparison: original.id, refs: [reference.id]});
      await refresh();
    });
    const nextSignature = JSON.stringify([candidates.map(row => [row.id, row.status]), data.recordings.map(row => row.id)]);
    if (signature === nextSignature) return;
    signature = nextSignature;
    const groups = new Map();
    for (const job of candidates) {
      const key = `${job.comparison_id}-${job.seed}`;
      if (!groups.has(key)) groups.set(key, new Map());
      const group = groups.get(key);
      if (!group.has(job.variant)) group.set(job.variant, job);
    }
    rounds.replaceChildren();
    const sorted = [...groups.values()].sort((left, right) => left.values().next().value.seed - right.values().next().value.seed);
    for (const group of sorted) {
      const job = group.values().next().value;
      const source = data.recordings.find(row => row.id === job.comparison_id);
      const list = document.createElement("fieldset");
      const title = document.createElement("legend");
      title.textContent = `Seed ${job.seed}`;
      list.append(title);
      const batch = job.batch_id ? `:${job.batch_id}` : "";
      const preference = `voice-studio:lapshi:${job.comparison_id}:${job.seed}${batch}`;
      for (const [variant, name] of Object.entries(LABELS)) {
        const take = group.get(variant);
        if (take) addTake(list, take, source, name, preference);
      }
      rounds.append(list);
    }
    const completed = candidates.filter(job => job.status === "done").length;
    status.textContent = candidates.length ? uiText`Готово дорожек: ${completed} из ${candidates.length}`
      : data.stressComparisonReady ? "" : uiText("Весы F5 accent-tune не установлены.");
  };
}

/** Render ready same-seed pairs with stable blind labels and a persisted listening verdict. */
export function mountProsodyPilot(request, compare) {
  const panel = document.querySelector("#prosody-pilot");
  const container = document.querySelector("#prosody-pilot-pairs");
  let signature;

  /** Build the controls for one pair, using the plain job ID as its persisted identity. */
  function pairCard(plain, marked) {
    const field = document.createElement("fieldset");
    const title = document.createElement("legend");
    title.textContent = `Seed ${plain.seed}`;
    const text = document.createElement("p");
    text.textContent = plain.text;
    const actions = document.createElement("div");
    actions.className = "actions";
    const play = document.createElement("button");
    play.type = "button";
    play.textContent = uiText("Открыть A/B");
    const plusFirst = parseInt(plain.id.slice(-1), 16) % 2 === 0;
    const a = plusFirst ? marked : plain;
    const b = plusFirst ? plain : marked;
    play.onclick = () => compare(a.id, uiText("Дорожка A"), b.id, uiText("Дорожка B"));
    const label = document.createElement("label");
    label.className = "select-field";
    label.textContent = uiText("Верное ударение");
    const select = document.createElement("select");
    const choices = [
      ["", uiText("Выбери после прослушивания")],
      [plusFirst ? "plus" : "plain", uiText("Только в A")],
      [plusFirst ? "plain" : "plus", uiText("Только в B")],
      ["both", uiText("В обеих")], ["neither", uiText("Ни в одной")], ["unclear", uiText("Не могу определить")],
    ];
    select.replaceChildren(...choices.map(([value, caption]) => new Option(caption, value)));
    select.value = plain.listener_choice || "";
    label.append(select);
    const feedback = document.createElement("p");
    feedback.setAttribute("role", "status");
    if (plain.listener_choice) feedback.textContent = uiText("Оценка сохранена в MLflow.");
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = uiText("Сохранить оценку");
    save.disabled = !select.value;
    select.onchange = () => { save.disabled = !select.value; };
    save.onclick = async () => {
      save.disabled = select.disabled = true;
      try {
        await request("rate-prosody-pair", {id: plain.id, choice: select.value});
        feedback.textContent = uiText("Сохранено в MLflow.");
      } catch (error) {
        feedback.textContent = error.message;
      } finally {
        select.disabled = false;
        save.disabled = !select.value;
      }
    };
    actions.append(play);
    field.append(title, text, actions, label, save, feedback);
    return field;
  }

  return jobs => {
    const marked = new Map(jobs.filter(job => job.variant === "pilot-plus" && job.status === "done")
      .map(job => [job.baseline_id, job]));
    const pairs = jobs.filter(job => job.variant === "pilot-plain" && job.status === "done" && marked.has(job.id))
      .sort((left, right) => left.seed - right.seed);
    panel.hidden = !pairs.length;
    const next = JSON.stringify(pairs.map(job => [job.id, marked.get(job.id).id, job.listener_choice]));
    if (signature === next) return;
    signature = next;
    container.replaceChildren(...pairs.map(job => pairCard(job, marked.get(job.id))));
  };
}
