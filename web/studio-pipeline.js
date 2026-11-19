import {createCardConnections} from './vendor/ag-card-connections.js';
import {uiText} from './i18n.js';
import {layoutPipeline, changePipelineDependency} from './studio-pipeline-model.js';

const NATIVE_LABELS = {
  'builtin:speech': 'Речь и говорящие', 'builtin:audio': 'Музыка, фон и интонация',
  'builtin:visual': 'Лица и активный говорящий', 'builtin:people': 'Фигуры, взгляд и маски',
  'builtin:scene': 'Предметы и интерфейсы', 'builtin:voice': 'Проверка голосовых профилей',
  'builtin:object_review': 'Сопоставление предметов', 'builtin:report': 'Результат выбранных этапов',
};

/** Translate host-owned native slots while retaining user-supplied mod names verbatim. */
function stageLabel(node) {
  return node.kind === 'builtin' ? uiText(NATIVE_LABELS[node.id] ?? node.label) : node.label;
}

/** Mount the Settings DAG editor. The backend owns compatibility, persistence and scheduling. */
export function mountStudioPipeline(find, request) {
  const canvas = find('pipeline-canvas');
  const inspector = find('pipeline-inspector');
  const status = find('pipeline-status');
  const save = find('pipeline-save');
  const undo = find('pipeline-undo');
  const redo = find('pipeline-redo');
  let plan, draft = {}, selected, connections, loading = false, sequence = 0;
  let past = [], future = [];

  /** Keep one explicit draft history; no layout coordinates are persisted as execution order. */
  function commit(next) {
    past.push(draft);
    if (past.length > 50) past.shift();
    draft = next;
    future = [];
    render();
    status.textContent = uiText('План изменён. Сохрани его для следующего запуска.');
  }

  /** Disable persistence while a request is pending; keep draft data on any server error. */
  function controls() {
    undo.disabled = loading || !past.length;
    redo.disabled = loading || !future.length;
    save.disabled = loading || !plan || JSON.stringify(draft) === JSON.stringify(plan.dependencies ?? {});
    find('pipeline-refresh').disabled = loading;
    canvas.setAttribute("aria-busy", String(loading));
    for (const input of inspector.querySelectorAll('input')) input.disabled = loading || input.dataset.blocked === 'true';
  }

  /** Render semantic cards and real directed AG connections, then a matching dependency inspector. */
  function render() {
    connections?.dispose();
    canvas.replaceChildren();
    const graph = layoutPipeline(plan, draft);
    const cards = new Map();
    for (const node of graph.nodes) {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'pipeline-node';
      card.dataset.nodeId = node.id;
      card.dataset.kind = node.kind;
      card.style.gridColumn = String(node.column + 1);
      card.style.gridRow = String(node.row + 1);
      card.setAttribute('aria-pressed', String(selected === node.id));
      const label = document.createElement('strong');
      label.textContent = stageLabel(node);
      const subtitle = document.createElement('small');
      subtitle.textContent = uiText(node.editable ? node.status === 'missing' ? 'Мод недоступен' : node.enabled === false ? 'Мод выключен' : 'Дополнительный мод' : 'Встроенный этап');
      card.append(label, subtitle);
      card.addEventListener('click', () => {
        selected = node.id;
        for (const [id, element] of cards) element.setAttribute('aria-pressed', String(selected === id));
        renderInspector();
      });
      cards.set(node.id, card);
      canvas.append(card);
    }
    const edges = graph.edges.map(edge => ({...edge, label: `${cards.get(edge.from).textContent} → ${cards.get(edge.to).textContent}`}));
    connections = createCardConnections({container: canvas, cards, edges, onSelect: edge => {
      selected = edge.to;
      cards.get(selected).click();
      cards.get(selected).focus();
    }});
    const edgeList = canvas.querySelector('.ag-card-connections__list');
    edgeList.setAttribute('aria-label', uiText('Связи этапов'));
    find('pipeline-edges').replaceChildren(edgeList);
    renderInspector();
    controls();
  }

  /** Edit extra-stage parents, retaining mandatory native edges and an accessible list of their names. */
  function renderInspector() {
    inspector.replaceChildren();
    const node = plan.nodes.find(item => item.id === selected);
    const heading = document.createElement('h3');
    heading.textContent = node ? stageLabel(node) : uiText('Выбери этап на графе');
    inspector.append(heading);
    if (!node) return;
    const identity = document.createElement('code');
    identity.textContent = node.id;
    inspector.append(identity);
    const description = document.createElement('p');
    description.textContent = uiText(node.editable ? 'Мод получит результаты выбранных предшественников. Обязательный результат анализа сохраняется.' : 'Обязательные связи встроенного этапа защищены. Здесь показан порядок зависимостей, а не журнал фактических запусков.');
    inspector.append(description);
    if (node.editable) {
      const group = document.createElement('fieldset');
      const legend = document.createElement('legend');
      legend.textContent = uiText('Запускать после');
      group.append(legend);
      const candidates = plan.nodes.filter(item => item.editable && item.id !== node.id);
      if (!candidates.length) {
        const empty = document.createElement('p');
        empty.textContent = uiText('Включи ещё один дополнительный мод, чтобы связать этапы.');
        group.append(empty);
      }
      for (const candidate of candidates) {
        const label = document.createElement('label');
        label.className = 'inline-check';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = (draft[node.id] ?? []).includes(candidate.id);
        input.dataset.blocked = String(candidate.enabled === false && !input.checked);
        input.disabled = loading || input.dataset.blocked === 'true';
        input.dataset.parentId = candidate.id;
        input.addEventListener('change', () => {
          try {
            const next = changePipelineDependency(plan, draft, node.id, candidate.id, input.checked);
            commit(next);
            [...inspector.querySelectorAll('input')].find(item => item.dataset.parentId === candidate.id)?.focus();
          } catch (error) {
            input.checked = !input.checked;
            status.textContent = uiText('Связь не добавлена: проверь цикл или доступность мода.');
          }
        });
        label.append(input, document.createTextNode(candidate.label));
        group.append(label);
      }
      inspector.append(group);
    }
    const list = document.createElement('ul');
    const current = layoutPipeline(plan, draft).nodes.find(item => item.id === selected);
    for (const parent of current.dependsOn) {
      const item = plan.nodes.find(entry => entry.id === parent);
      const row = document.createElement('li');
      row.textContent = stageLabel(item);
      list.append(row);
    }
    inspector.append(list);
    controls();
  }

  /** Reload only on explicit user action; a failed save never discards the current draft. */
  async function open(force = false) {
    if (loading) return;
    if (!force && plan && JSON.stringify(draft) !== JSON.stringify(plan.dependencies ?? {})) {
      status.textContent = uiText("Черновик плана сохранён на экране. Примени изменения или обнови план.");
      return;
    }
    const version = ++sequence;
    loading = true;
    controls();
    status.textContent = uiText('Читаем план обработки…');
    try {
      const result = await request('mods/pipeline');
      if (version !== sequence) return;
      layoutPipeline(result);
      plan = result;
      draft = structuredClone(result.dependencies ?? {});
      past = []; future = [];
      selected = plan.nodes.some(node => node.id === selected) ? selected : undefined;
      render();
      status.textContent = uiText('Выбери этап, чтобы настроить его зависимости.');
    } catch (error) {
      status.textContent = error.message;
    } finally {
      loading = false;
      controls();
    }
  }

  save.addEventListener('click', async () => {
    if (loading || !plan) return;
    loading = true; controls();
    status.textContent = uiText('Сохраняем план…');
    try {
      const result = await request('mods/pipeline', {dependencies: draft, expectedRevision: plan.revision});
      layoutPipeline(result);
      plan = result; draft = structuredClone(result.dependencies ?? {});
      past = []; future = [];
      render();
      status.textContent = uiText('План сохранён. Изменения применятся к следующему запуску.');
    } catch (error) {
      status.textContent = error.message;
    } finally {
      loading = false; controls();
      if (!save.disabled) save.focus();
      else find("pipeline-refresh").focus();
    }
  });
  undo.addEventListener('click', () => {if (!past.length || loading) return; future.push(draft); draft = past.pop(); render(); (undo.disabled ? redo : undo).focus();});
  redo.addEventListener('click', () => {if (!future.length || loading) return; past.push(draft); draft = future.pop(); render(); (redo.disabled ? undo : redo).focus();});
  find('pipeline-refresh').addEventListener('click', () => {
    const dirty = plan && JSON.stringify(draft) !== JSON.stringify(plan.dependencies ?? {});
    if (!dirty || window.confirm(uiText('Обновить план и отменить несохранённые связи?'))) open(true);
  });
  return {open, catalogChanged() {
    if (!plan || JSON.stringify(draft) === JSON.stringify(plan.dependencies ?? {})) return open();
    status.textContent = uiText("Каталог модов изменился. Сохрани черновик или обнови план для новых этапов.");
  }};
}
