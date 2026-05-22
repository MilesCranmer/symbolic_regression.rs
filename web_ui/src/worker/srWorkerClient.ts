import type { WorkerFromWorkerMsg, WorkerToWorkerMsg } from "./protocol";
import type { WasmSearchOptions } from "../types/srTypes";

type Handlers = {
  onReady?: (split: unknown) => void;
  onSnapshot?: (snap: unknown) => void;
  onFrontUpdate?: (front: unknown) => void;
  onEvalResult?: (requestId: string, result: unknown) => void;
  onDone?: () => void;
  onPaused?: () => void;
  onResetDone?: () => void;
  onError?: (err: string) => void;
};

export class SrWorkerClient {
  private worker: Worker;
  private handlers: Handlers = {};

  constructor() {
    this.worker = new Worker(new URL("./srWorker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<WorkerFromWorkerMsg>) => {
      const msg = e.data;
      if (msg.type === "ready") this.handlers.onReady?.(msg.split);
      else if (msg.type === "snapshot") this.handlers.onSnapshot?.(msg.snap);
      else if (msg.type === "front_update") this.handlers.onFrontUpdate?.(msg.front);
      else if (msg.type === "eval_result") this.handlers.onEvalResult?.(msg.requestId, msg.result);
      else if (msg.type === "done") this.handlers.onDone?.();
      else if (msg.type === "paused") this.handlers.onPaused?.();
      else if (msg.type === "reset_done") this.handlers.onResetDone?.();
      else if (msg.type === "error") this.handlers.onError?.(msg.error);
    };
  }

  setHandlers(handlers: Handlers): void {
    this.handlers = handlers;
  }

  terminate(): void {
    this.worker.terminate();
  }

  post(msg: WorkerToWorkerMsg): void {
    this.worker.postMessage(msg);
  }

  init(args: {
    csvText: string;
    options: WasmSearchOptions;
    unary: string[];
    binary: string[];
    ternary: string[];
  }): void {
    this.post({ type: "init", ...args });
  }

  start(): void {
    this.post({ type: "start" });
  }

  pause(): void {
    this.post({ type: "pause" });
  }

  reset(): void {
    this.post({ type: "reset" });
  }

  step(cycles: number): void {
    this.post({ type: "step", cycles });
  }

  evaluate(requestId: string, memberId: string, which: "train" | "val"): void {
    this.post({ type: "evaluate", requestId, memberId, which });
  }
}
