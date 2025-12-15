/// <reference lib="webworker" />

import init, { WasmSearch } from "./pkg/sr_wasm.js";

type InitMsg = {
  type: "init";
  csvText: string;
  options: unknown;
  operators: string[];
};

type RunMsg = {
  type: "run";
  stepCycles: number;
  snapshotEverySteps: number;
};

type StopMsg = { type: "stop" };

type Msg = InitMsg | RunMsg | StopMsg;

let search: WasmSearch | null = null;
let running = false;

async function sleep0(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

self.onmessage = async (e: MessageEvent<Msg>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init();
      search = new WasmSearch(msg.csvText, msg.options as any, msg.operators as any);
      self.postMessage({ type: "ready" });
      return;
    }

    if (msg.type === "stop") {
      running = false;
      self.postMessage({ type: "stopped" });
      return;
    }

    if (msg.type === "run") {
      if (!search) {
        self.postMessage({ type: "error", error: "search not initialized" });
        return;
      }

      running = true;
      let steps = 0;
      while (running && !search.is_finished()) {
        const snap = search.step(msg.stepCycles);
        steps += 1;
        if (steps % msg.snapshotEverySteps === 0) {
          self.postMessage({ type: "snapshot", snap });
          await sleep0();
        }
      }
      if (running) {
        self.postMessage({ type: "snapshot", snap: search.step(0) });
        self.postMessage({ type: "done" });
      }
      running = false;
      return;
    }
  } catch (err) {
    self.postMessage({ type: "error", error: String(err) });
  }
};
