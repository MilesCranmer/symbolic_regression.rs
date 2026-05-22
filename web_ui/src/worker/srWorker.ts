/// <reference lib="webworker" />

import init, { init_thread_pool, WasmSearch } from "../pkg/symbolic_regression_wasm.js";
import type { WorkerFromWorkerMsg, WorkerToWorkerMsg } from "./protocol";

let search: WasmSearch | null = null;
let running = false;

const SNAPSHOT_INTERVAL_MS = 200;
const FRONT_INTERVAL_MS = 250;
const STEP_CYCLES = 10;
const PARETO_K = 10_000;
const FRONT_K = 10_000;

function post(msg: WorkerFromWorkerMsg): void {
  self.postMessage(msg);
}

async function sleep0(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

async function runLoop(): Promise<void> {
  if (!search) {
    post({ type: "error", error: "search not initialized" });
    return;
  }

  running = true;
  let lastSnap = 0;
  let lastFront = 0;

  while (running && !search.is_finished()) {
    search.step(STEP_CYCLES);

    const now = performance.now();
    if (now - lastSnap >= SNAPSHOT_INTERVAL_MS) {
      const snap = search.step(0);
      post({ type: "snapshot", snap });
      lastSnap = now;
    }
    if (now - lastFront >= FRONT_INTERVAL_MS) {
      const front = search.best_equations(FRONT_K);
      post({ type: "front_update", front });
      lastFront = now;
    }

    await sleep0();
  }

  if (!running) {
    post({ type: "paused" });
    return;
  }

  post({ type: "snapshot", snap: search.step(0) });
  post({ type: "front_update", front: search.best_equations(FRONT_K) });
  post({ type: "done" });
  running = false;
}

self.onmessage = async (e: MessageEvent<WorkerToWorkerMsg>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      running = false;
      const wasmExports = await init();
      const sharedArrayBufferAvailable = typeof SharedArrayBuffer !== "undefined";
      const hasSharedMemory = sharedArrayBufferAvailable && wasmExports?.memory?.buffer instanceof SharedArrayBuffer;
      if (self.crossOriginIsolated && hasSharedMemory) {
        const n = Math.max(2, Math.min(Number(self.navigator?.hardwareConcurrency ?? 4), 16));
        try {
          await init_thread_pool(n);
        } catch (err) {
          // eslint-disable-next-line no-console
          console.warn("init_thread_pool failed, continuing single-threaded:", err);
        }
      } else {
        // eslint-disable-next-line no-console
        console.warn("SharedArrayBuffer unavailable; running single-threaded.");
      }
      search = new WasmSearch(msg.csvText, msg.options as any, msg.unary as any, msg.binary as any, msg.ternary as any);
      search.set_pareto_k(PARETO_K);
      const split = search.get_split_indices();
      post({ type: "ready", split });
      return;
    }

    if (msg.type === "reset") {
      running = false;
      search = null;
      post({ type: "reset_done" });
      return;
    }

    if (msg.type === "pause") {
      running = false;
      post({ type: "paused" });
      return;
    }

    if (msg.type === "step") {
      if (!search) {
        post({ type: "error", error: "search not initialized" });
        return;
      }
      search.step(Math.max(0, msg.cycles | 0));
      post({ type: "snapshot", snap: search.step(0) });
      post({ type: "front_update", front: search.best_equations(FRONT_K) });
      return;
    }

    if (msg.type === "evaluate") {
      if (!search) {
        post({ type: "error", error: "search not initialized" });
        return;
      }
      const result = search.evaluate_member(msg.memberId, msg.which);
      post({ type: "eval_result", requestId: msg.requestId, result });
      return;
    }

    if (msg.type === "start") {
      if (!search) {
        post({ type: "error", error: "search not initialized" });
        return;
      }
      if (!running) {
        void runLoop();
      }
      return;
    }
  } catch (err) {
    post({ type: "error", error: String(err) });
  }
};
