/** Normalize bounded embeddings before float32 inference, preserving very small magnitudes. */
export function normalizedVectors(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || !left.length || left.length > 4096 || left.length !== right.length) {
    throw new Error("Supply two nonzero numeric vectors of the same length, up to 4096 values.");
  }
  const result = [];
  for (const values of [left, right]) {
    let scale = 0;
    for (const value of values) {
      if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Vectors must contain finite numbers.");
      scale = Math.max(scale, Math.abs(value));
    }
    if (!scale) throw new Error("Cosine similarity requires nonzero vectors.");
    let squares = 0;
    for (const value of values) squares += (value / scale) ** 2;
    const norm = Math.sqrt(squares);
    result.push(Float32Array.from(values, value => (value / scale) / norm));
  }
  return result;
}

/** Compute the float32 input reference in JS. The result is a cosine, never an identity probability. */
export function referenceCosine(left, right) {
  let dot = 0;
  for (let index = 0; index < left.length; index++) dot += left[index] * right[index];
  return Math.max(-1, Math.min(1, dot));
}

/** Retain one shape-specific graph and its buffers for a worker's bounded lifetime. */
export class BrowserMLKernel {
  constructor(environment = globalThis) {
    this.environment = environment;
    this.session = null;
    this.failed = null;
  }

  /** Release graphs, tensors and context on shape changes, failure or worker termination. */
  dispose() {
    const session = this.session;
    this.session = null;
    if (!session) return;
    for (const item of [session.left, session.right, session.output, session.graph, session.context]) {
      try { item?.destroy(); } catch { /* Continue releasing the remaining device allocations. */ }
    }
  }

  /** Compile mul/reduceSum once and reuse tensor storage for repeated inference. */
  async prepare(length, powerPreference) {
    const key = length + ":" + powerPreference;
    if (this.session?.key === key) return {session: this.session, reused: true};
    this.dispose();
    if (this.failed?.key === key) throw new Error(this.failed.reason);
    const {navigator, MLGraphBuilder} = this.environment;
    if (typeof navigator?.ml?.createContext !== "function" || typeof MLGraphBuilder !== "function") {
      throw new Error("WebNN is unavailable in this browser worker.");
    }
    const session = {key};
    this.session = session;
    try {
      const start = this.environment.performance.now();
      // Current spec chooses an accelerator through preferences. Physical device identity is unknown.
      session.context = await navigator.ml.createContext({powerPreference, accelerated: true});
      const builder = new MLGraphBuilder(session.context);
      const descriptor = {dataType: "float32", shape: [length]};
      const a = builder.input("left", descriptor);
      const b = builder.input("right", descriptor);
      const dot = builder.reduceSum(builder.mul(a, b), {keepDimensions: true});
      session.graph = await builder.build({dot});
      // Assign sequentially so partial allocation failures can release every completed tensor.
      session.left = await session.context.createTensor({...descriptor, writable: true});
      session.right = await session.context.createTensor({...descriptor, writable: true});
      session.output = await session.context.createTensor({dataType: "float32", shape: [1], readable: true});
      session.compileMs = this.environment.performance.now() - start;
      return {session, reused: false};
    } catch (error) {
      this.dispose();
      this.failed = {key, reason: error.message};
      throw error;
    }
  }

  /** Compare normalized embeddings and validate WebNN numerics against the JS reference on every call. */
  async compare(leftValues, rightValues, powerPreference = "default", runs = 3) {
    if (!["default", "low-power", "high-performance"].includes(powerPreference) ||
        !Number.isInteger(runs) || runs < 1 || runs > 20) throw new Error("Invalid inference settings.");
    const [left, right] = normalizedVectors(leftValues, rightValues);
    const reference = referenceCosine(left, right);
    const start = this.environment.performance.now();
    try {
      const {session, reused} = await this.prepare(left.length, powerPreference);
      const {context, graph} = session;
      context.writeTensor(session.left, left);
      context.writeTensor(session.right, right);
      const inputs = {left: session.left, right: session.right};
      const outputs = {dot: session.output};
      const warmStart = this.environment.performance.now();
      context.dispatch(graph, inputs, outputs);
      await context.readTensor(session.output);
      const warmupMs = this.environment.performance.now() - warmStart;
      const durations = [];
      let actual;
      for (let run = 0; run < runs; run++) {
        const tick = this.environment.performance.now();
        context.dispatch(graph, inputs, outputs);
        actual = new Float32Array(await context.readTensor(session.output))[0];
        durations.push(this.environment.performance.now() - tick);
      }
      const absoluteError = Math.abs(reference - actual);
      if (!Number.isFinite(actual) || absoluteError > 0.00001) {
        throw new Error("WebNN cosine failed the numeric reference check.");
      }
      return {backend: "webnn", similarity: Math.max(-1, Math.min(1, actual)), reference, absoluteError,
        dimensions: left.length, powerPreference, physicalDevice: null, reused,
        compileMs: reused ? 0 : session.compileMs, warmupMs, runMs: durations,
        totalMs: this.environment.performance.now() - start};
    } catch (error) {
      this.dispose();
      this.failed = {key: left.length + ":" + powerPreference, reason: error.message};
      return {backend: "javascript", similarity: reference, dimensions: left.length, physicalDevice: null,
        fallbackReason: String(error.message).slice(0, 200), totalMs: this.environment.performance.now() - start};
    }
  }

  /** Execute the same real workload used by comparisons instead of testing API presence alone. */
  async probe(powerPreference = "default") {
    const result = await this.compare([1, 2, 3, 4], [1, 2, 3, 4], powerPreference);
    return {scope: "browser-worker", apiPresent: typeof this.environment.navigator?.ml?.createContext === "function",
      verifiedWebNN: result.backend === "webnn", workload: "float32 mul + reduceSum", ...result,
      modelCompatibility: "unmeasured", identityDecision: "caller-must-calibrate-on-labeled-pairs"};
  }
}
