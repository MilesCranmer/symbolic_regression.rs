import { create } from "zustand";
import * as Papa from "papaparse";
import initWasm, { builtin_operator_registry, default_search_options } from "../pkg/symbolic_regression_wasm.js";
import { DEFAULT_CSV } from "../generated/defaultCsv";
import type {
  EquationPoint,
  EquationSummary,
  SearchSnapshot,
  WasmEvalResult,
  WasmOpInfo,
  WasmSearchOptions,
  WasmSplitIndices
} from "../types/srTypes";

export type TabKey = "data" | "clean" | "task" | "search";

type ParsedDataset = {
  headers: string[];
  rows: number[][];
};

type SearchRuntime = {
  status: "idle" | "initializing" | "ready" | "running" | "paused" | "done" | "error";
  error: string | null;
  split: WasmSplitIndices | null;
  snapshot: SearchSnapshot | null;
  front: EquationSummary[];
  selectedId: string | null;
  selectedComplexity: number | null;
  evalByKey: Record<string, WasmEvalResult>;
};

type SessionState = {
  tab: TabKey;

  wasmLoaded: boolean;
  operatorRegistry: WasmOpInfo[];
  defaultOptions: WasmSearchOptions | null;

  csvText: string;
  parsed: ParsedDataset | null;

  options: WasmSearchOptions | null;
  unaryOps: string[];
  binaryOps: string[];
  ternaryOps: string[];

  runtime: SearchRuntime;

  setTab: (tab: TabKey) => void;
  loadWasmMetadata: () => Promise<void>;
  setCsvText: (csvText: string) => void;
  parseCsv: () => void;
  setOptionsPatch: (patch: Partial<WasmSearchOptions>) => void;

  toggleOp: (arity: 1 | 2 | 3, name: string) => void;
  applyPreset: (preset: "basic" | "trig" | "explog" | "all") => void;

  setRuntime: (patch: Partial<SearchRuntime>) => void;
  setSnapshot: (snap: SearchSnapshot) => void;
  setFront: (front: EquationSummary[]) => void;
  setSelection: (id: string | null, complexity: number | null) => void;
  setEvalResult: (memberId: string, which: "train" | "val", result: WasmEvalResult) => void;
};

function keyForEval(memberId: string, which: "train" | "val"): string {
  return `${memberId}:${which}`;
}

function toggleInList(list: string[], value: string): string[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

export const useSessionStore = create<SessionState>((set, get) => ({
  tab: "data",

  wasmLoaded: false,
  operatorRegistry: [],
  defaultOptions: null,

  csvText: DEFAULT_CSV,
  parsed: null,

  options: null,
  unaryOps: [],
  binaryOps: [],
  ternaryOps: [],

  runtime: {
    status: "idle",
    error: null,
    split: null,
    snapshot: null,
    front: [],
    selectedId: null,
    selectedComplexity: null,
    evalByKey: {}
  },

  setTab: (tab) => set({ tab }),

  loadWasmMetadata: async () => {
    if (get().wasmLoaded) return;
    await initWasm();
    const [ops, defaults] = await Promise.all([builtin_operator_registry(), default_search_options()]);

    const operatorRegistry = ops as WasmOpInfo[];
    const defaultOptions = defaults as WasmSearchOptions;
    set({
      wasmLoaded: true,
      operatorRegistry,
      defaultOptions,
      options: defaultOptions
    });

    // Default operator selection (a safe "basic" set).
    get().applyPreset("basic");
  },

  setCsvText: (csvText) => set({ csvText }),

  parseCsv: () => {
    const { csvText, options } = get();
    if (!options) return;

    const res = Papa.parse(csvText, {
      skipEmptyLines: true,
      dynamicTyping: true
    });
    if (res.errors.length > 0) {
      set((s) => ({
        runtime: { ...s.runtime, status: "error", error: res.errors[0]?.message ?? "CSV parse error" }
      }));
      return;
    }

    const raw = res.data as unknown[];
    if (raw.length === 0) return;

    let headers: string[] = [];
    let dataRows: unknown[] = raw;
    if (options.has_headers) {
      const h0 = raw[0] as unknown;
      if (!Array.isArray(h0)) {
        set((s) => ({
          runtime: { ...s.runtime, status: "error", error: "expected header row to be an array" }
        }));
        return;
      }
      headers = h0.map((v) => String(v));
      dataRows = raw.slice(1);
    }

    const rows: number[][] = [];
    for (const r of dataRows) {
      if (!Array.isArray(r)) continue;
      const row = r.map((v) => Number(v));
      if (row.some((v) => !Number.isFinite(v))) continue;
      rows.push(row);
    }
    if (rows.length === 0) {
      set((s) => ({
        runtime: { ...s.runtime, status: "error", error: "no numeric data rows parsed" }
      }));
      return;
    }

    const nCols = rows[0].length;
    if (headers.length !== nCols) {
      headers = Array.from({ length: nCols }, (_, i) => `col_${i}`);
    }

    // Defaults: y is last col, X is all others.
    const y = options.y_column ?? (nCols - 1);
    const weights = options.weights_column ?? null;
    const x =
      options.x_columns ??
      Array.from({ length: nCols }, (_, i) => i).filter((i) => i !== y && i !== weights);

    set({
      parsed: { headers, rows },
      options: {
        ...options,
        y_column: y,
        weights_column: weights,
        x_columns: x
      }
    });
  },

  setOptionsPatch: (patch) =>
    set((s) => ({
      options: s.options ? { ...s.options, ...patch } : s.options
    })),

  toggleOp: (arity, name) =>
    set((s) => {
      if (arity === 1) return { unaryOps: toggleInList(s.unaryOps, name) };
      if (arity === 2) return { binaryOps: toggleInList(s.binaryOps, name) };
      return { ternaryOps: toggleInList(s.ternaryOps, name) };
    }),

  applyPreset: (preset) =>
    set(() => {
      // Default presets intentionally avoid unary negation ("neg") and all ternary ops.
      // Initial load uses "basic": only +, -, *, /.
      const basicUnary: string[] = [];
      const basicBinary = ["add", "sub", "mul", "div"];
      const trigUnary = ["sin", "cos"];
      const explogUnary = ["exp", "log"];

      if (preset === "basic") return { unaryOps: basicUnary, binaryOps: basicBinary, ternaryOps: [] };
      if (preset === "trig") return { unaryOps: [...basicUnary, ...trigUnary], binaryOps: basicBinary, ternaryOps: [] };
      if (preset === "explog")
        return { unaryOps: [...basicUnary, ...explogUnary], binaryOps: basicBinary, ternaryOps: [] };
      return {}; // "all" handled in UI (requires registry)
    }),

  setRuntime: (patch) =>
    set((s) => ({
      runtime: { ...s.runtime, ...patch }
    })),

  setSnapshot: (snap) =>
    set((s) => ({
      runtime: { ...s.runtime, snapshot: snap }
    })),

  setFront: (front) =>
    set((s) => ({
      runtime: { ...s.runtime, front }
    })),

  setSelection: (id, complexity) =>
    set((s) => ({
      runtime: { ...s.runtime, selectedId: id, selectedComplexity: complexity }
    })),

  setEvalResult: (memberId, which, result) =>
    set((s) => ({
      runtime: {
        ...s.runtime,
        evalByKey: { ...s.runtime.evalByKey, [keyForEval(memberId, which)]: result }
      }
    }))
}));

export function getSelectedSummary(front: EquationSummary[], best: EquationSummary | null, selectedId: string | null) {
  if (selectedId == null) return best;
  return front.find((m) => m.id === selectedId) ?? (best?.id === selectedId ? best : null);
}

export function getParetoPoints(snap: SearchSnapshot | null): EquationPoint[] {
  return snap?.pareto_points ?? [];
}
