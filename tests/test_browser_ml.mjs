import test from "node:test";
import assert from "node:assert/strict";
import {BrowserMLKernel} from "../web/browser-ml-kernel.js";
import {BrowserML} from "../web/browser-ml.js";
import {studioWebTools} from "../web/web-mcp.js";

/** Emulate only the graph boundary to exercise numeric regressions and partial allocations. */
function environment({wrongResult = false, failTensor = 0} = {}) {
  const state = {created: 0, destroyed: 0, tensors: 0};
  const item = value => ({value, destroy() { state.destroyed++; }});
  const context = {
    destroy() { state.destroyed++; },
    async createTensor() {
      state.tensors++;
      if (state.tensors === failTensor) throw new Error("Allocation failed");
      return item();
    },
    writeTensor(tensor, values) { tensor.value = values; },
    dispatch(graph, {left, right}, {dot}) {
      let result = 0;
      for (let i = 0; i < left.value.length; i++) result += left.value[i] * right.value[i];
      dot.value = wrongResult ? .5 : result;
    },
    async readTensor(tensor) { return new Float32Array([tensor.value]).buffer; },
  };
  class Builder {
    input() { return {}; }
    mul() { return {}; }
    reduceSum() { return {}; }
    async build() { return item(); }
  }
  return {state, performance, MLGraphBuilder: Builder,
    navigator: {ml: {async createContext() { state.created++; return context; }}}};
}

test("agent comparisons validate vectors and preserve negative and extreme-magnitude cosine", async () => {
  const kernel = new BrowserMLKernel({performance});
  for (const [left, right, expected] of [
    [[1, 0], [0, 1], 0], [[1, 2], [-1, -2], -1],
    [[1e308, 0], [1e-308, 0], 1],
  ]) {
    const result = await kernel.compare(left, right);
    assert.equal(result.backend, "javascript");
    assert(Math.abs(result.similarity - expected) < 1e-6);
    assert.match(result.fallbackReason, /unavailable/);
  }
  for (const pair of [[[0], [1]], [[1], [1, 2]], [[NaN], [1]], [[Infinity], [1]]]) {
    await assert.rejects(kernel.compare(...pair));
  }
  await assert.rejects(kernel.compare(Array(4097).fill(1), Array(4097).fill(1)));
});

test("successful inference reuses a compiled shape, checks numerics and releases all five allocations", async () => {
  const env = environment();
  const kernel = new BrowserMLKernel(env);
  const probe = await kernel.probe();
  assert.equal(probe.verifiedWebNN, true);
  assert.equal(probe.physicalDevice, null);
  assert.equal(probe.modelCompatibility, "unmeasured");
  const compared = await kernel.compare([1, 0, 0, 0], [0, 1, 0, 0], "default", 2);
  assert.equal(compared.backend, "webnn");
  assert.equal(compared.similarity, 0);
  assert.equal(compared.runMs.length, 2);
  assert.equal(compared.reused, true);
  assert.equal(compared.compileMs, 0);
  assert.equal(env.state.created, 1);
  await kernel.compare([1], [1]);
  assert.equal(env.state.created, 2);
  assert.equal(env.state.destroyed, 5);
  kernel.dispose();
  assert.equal(env.state.destroyed, 10);
});

test("numerical regression and partial allocation fall back with device resources released", async () => {
  for (const options of [{wrongResult: true}, {failTensor: 2}]) {
    const env = environment(options);
    const kernel = new BrowserMLKernel(env);
    const result = await kernel.compare([1], [1]);
    assert.equal(result.backend, "javascript");
    assert.equal(result.similarity, 1);
    assert(result.fallbackReason);
    assert.equal(env.state.destroyed, options.wrongResult ? 5 : 3);
    await kernel.compare([1], [1]);
    assert.equal(env.state.created, 1, "a failed shape should not repeatedly rebuild");
  }
});

test("worker requests have bounded concurrency, abort cleanly and reject invalid input before allocation", async () => {
  let terminated = 0;
  class Worker extends EventTarget {
    postMessage() {}
    terminate() { terminated++; }
  }
  const runtime = new BrowserML({Worker});
  await assert.rejects(runtime.request("compare", {left: [0], right: [1]}));
  assert.equal(runtime.worker, null);
  const controller = new AbortController();
  const pending = runtime.request("probe", {}, controller.signal);
  await assert.rejects(runtime.request("probe"), /busy/);
  controller.abort();
  await assert.rejects(pending, {name: "AbortError"});
  assert.equal(terminated, 1);
  assert.equal(runtime.pending, false);
  assert.equal(runtime.worker, null);
});

test("WebMCP validates bounded vector schemas and exposes executable fallback through normal tools", async () => {
  const runtime = new BrowserML({});
  const tools = studioWebTools({ui: {}, browserML: runtime}, new AbortController().signal);
  const compare = tools.find(tool => tool.name === "browser_ml_cosine");
  assert.equal((await compare.execute({left: [1, 0], right: [1, 0]})).similarity, 1);
  for (const input of [{left: [0], right: [1]}, {left: ["1"], right: [1]},
    {left: [NaN], right: [1]}, {left: [1], right: [1], runs: 21}, {left: [1], right: [1], extra: true}]) {
    assert((await compare.execute(input)).error);
  }
  const probe = tools.find(tool => tool.name === "browser_ml_probe");
  assert.equal((await probe.execute()).verifiedWebNN, false);
});
