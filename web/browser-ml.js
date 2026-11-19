import {normalizedVectors, referenceCosine} from "./browser-ml-kernel.js";

/** Keep a single local inference worker with bounded work, cancellation and device teardown. */
export class BrowserML {
  constructor(environment = globalThis) {
    this.environment = environment;
    this.worker = null;
    this.pending = false;
  }

  /** Terminating the owning worker releases all of its device contexts and tensor allocations. */
  dispose() {
    this.worker?.terminate();
    this.worker = null;
  }

  /** Dispatch without an unbounded queue. Abort or timeout destroys the worker and its state. */
  async request(action, options = {}, signal) {
    signal?.throwIfAborted();
    if (this.pending) throw new Error("Browser ML is busy. Retry after the current operation.");
    if (!["probe", "compare"].includes(action)) throw new Error("Unknown browser ML operation.");
    const powerPreference = options.powerPreference ?? "default";
    const runs = options.runs ?? 3;
    if (!["default", "low-power", "high-performance"].includes(powerPreference) ||
        !Number.isInteger(runs) || runs < 1 || runs > 20) throw new Error("Invalid inference settings.");
    if (action === "compare") normalizedVectors(options.left, options.right);
    if (typeof this.environment.Worker !== "function") {
      const [left, right] = normalizedVectors(options.left ?? [1, 2], options.right ?? [1, 2]);
      return {backend: "javascript", similarity: referenceCosine(left, right), dimensions: left.length,
        verifiedWebNN: false, physicalDevice: null, fallbackReason: "Web Workers are unavailable."};
    }
    this.pending = true;
    try {
      this.worker ??= new this.environment.Worker(new URL("./browser-ml-worker.js", import.meta.url), {type: "module"});
      return await new Promise((resolve, reject) => {
        let timer;
        const finish = (error, result) => {
          clearTimeout(timer);
          signal?.removeEventListener("abort", abort);
          this.worker?.removeEventListener("message", receive);
          this.worker?.removeEventListener("error", failed);
          if (error) {
            this.dispose();
            reject(error);
          } else resolve(result);
        };
        const receive = ({data}) => finish(data.error ? new Error(data.error) : null, data.result);
        const failed = () => finish(new Error("Browser ML worker failed."));
        const abort = () => finish(signal.reason ?? new DOMException("Aborted", "AbortError"));
        this.worker.addEventListener("message", receive, {once: true});
        this.worker.addEventListener("error", failed, {once: true});
        signal?.addEventListener("abort", abort, {once: true});
        timer = setTimeout(() => finish(new Error("Browser ML timed out after 5 seconds.")), 5000);
        try {
          this.worker.postMessage({action, ...options, powerPreference, runs});
        } catch (error) {
          finish(error);
        }
      });
    } catch (error) {
      this.dispose();
      throw error;
    } finally {
      this.pending = false;
    }
  }
}
