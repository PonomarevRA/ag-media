import {BrowserMLKernel} from "./browser-ml-kernel.js";

const kernel = new BrowserMLKernel();

/** Run one bounded request off the UI thread. The client permits one in-flight request. */
self.addEventListener("message", async ({data}) => {
  try {
    const result = data.action === "probe"
      ? await kernel.probe(data.powerPreference)
      : await kernel.compare(data.left, data.right, data.powerPreference, data.runs);
    self.postMessage({result});
  } catch (error) {
    self.postMessage({error: error.message});
  }
});
