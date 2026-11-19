/** Resolve editable mod edges while preserving the host's mandatory native dependencies. */
export function pipelineNodes(plan, dependencies = plan.dependencies ?? {}) {
  const nodes = new Map(plan.nodes.map(node => [node.id, node]));
  if (nodes.size !== plan.nodes.length || nodes.size > 128) throw new Error('Invalid pipeline nodes');
  return plan.nodes.map(node => {
    const fixed = (node.dependsOn ?? []).filter(id => !node.editable || !nodes.get(id)?.editable);
    const chosen = node.editable ? dependencies[node.id] ?? [] : [];
    return {...node, dependsOn: [...new Set([...fixed, ...chosen])]};
  });
}

/** Compute stable topological columns; reject cycles and dangling links before drawing or saving. */
export function layoutPipeline(plan, dependencies) {
  const nodes = pipelineNodes(plan, dependencies);
  const known = new Map(nodes.map(node => [node.id, node]));
  const remaining = new Map(nodes.map(node => [node.id, new Set(node.dependsOn)]));
  const levels = new Map();
  const edges = [];
  for (const node of nodes) for (const parent of node.dependsOn) {
    if (!known.has(parent) || parent === node.id) throw new Error('Invalid pipeline dependency');
    edges.push({from: parent, to: node.id});
  }
  if (edges.length > 200) throw new Error('Pipeline supports at most 200 edges');
  while (remaining.size) {
    let advanced = false;
    for (const [id, parents] of remaining) {
      if ([...parents].some(parent => !levels.has(parent))) continue;
      let level = 0;
      for (const parent of parents) level = Math.max(level, levels.get(parent) + 1);
      levels.set(id, level);
      remaining.delete(id);
      advanced = true;
    }
    if (!advanced) throw new Error('Pipeline contains a cycle');
  }
  const rows = new Map();
  const positions = nodes.map(node => {
    const column = levels.get(node.id);
    const row = rows.get(column) ?? 0;
    rows.set(column, row + 1);
    return {...node, column, row};
  });
  return {nodes: positions, edges};
}

/** Change one dependency immutably and reject an invalid or cyclic draft. */
export function changePipelineDependency(plan, dependencies, target, parent, enabled) {
  const node = plan.nodes.find(item => item.id === target);
  const source = plan.nodes.find(item => item.id === parent);
  if (!node?.editable || !source?.editable || target === parent || enabled && source.enabled === false) {
    throw new Error('Only available mod stages can be linked');
  }
  const next = structuredClone(dependencies);
  const parents = new Set(next[target] ?? []);
  if (enabled) parents.add(parent); else parents.delete(parent);
  if (parents.size) next[target] = [...parents].sort();
  else delete next[target];
  layoutPipeline(plan, next);
  return next;
}
