import type { SearchSnapshot, EquationSummary, WasmEvalResult, WasmSearchOptions, WasmSplitIndices } from "../types/srTypes";

export type WorkerInitMsg = {
  type: "init";
  csvText: string;
  options: WasmSearchOptions;
  unary: string[];
  binary: string[];
  ternary: string[];
};

export type WorkerStartMsg = { type: "start" };
export type WorkerPauseMsg = { type: "pause" };
export type WorkerResetMsg = { type: "reset" };
export type WorkerStepMsg = { type: "step"; cycles: number };
export type WorkerEvaluateMsg = {
  type: "evaluate";
  memberId: string;
  which: "train" | "val";
  requestId: string;
};

export type WorkerToWorkerMsg =
  | WorkerInitMsg
  | WorkerStartMsg
  | WorkerPauseMsg
  | WorkerResetMsg
  | WorkerStepMsg
  | WorkerEvaluateMsg;

export type WorkerReadyMsg = { type: "ready"; split: WasmSplitIndices };
export type WorkerSnapshotMsg = { type: "snapshot"; snap: SearchSnapshot };
export type WorkerFrontUpdateMsg = { type: "front_update"; front: EquationSummary[] };
export type WorkerEvalResultMsg = { type: "eval_result"; requestId: string; result: WasmEvalResult };
export type WorkerDoneMsg = { type: "done" };
export type WorkerPausedMsg = { type: "paused" };
export type WorkerResetDoneMsg = { type: "reset_done" };
export type WorkerErrorMsg = { type: "error"; error: string };

export type WorkerFromWorkerMsg =
  | WorkerReadyMsg
  | WorkerSnapshotMsg
  | WorkerFrontUpdateMsg
  | WorkerEvalResultMsg
  | WorkerDoneMsg
  | WorkerPausedMsg
  | WorkerResetDoneMsg
  | WorkerErrorMsg;
