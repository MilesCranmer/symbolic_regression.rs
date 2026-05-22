import { create } from "zustand";
import * as Papa from "papaparse";
import initWasm, {
  builtin_operator_registry,
  default_core_options,
  default_search_options,
} from "../pkg/symbolic_regression_wasm.js";
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

export type TabKey = "data" | "configure" | "run";

type ParsedDataset = {
  headers: string[];
  rows: number[][];
};

type SearchRuntime = {
  status: "idle" | "initializing" | "ready" | "running" | "paused" | "done" | "error";
  error: string | null;
  split: WasmSplitIndices | null;
  snapshot: SearchSnapshot | null;
  evalsPerSecond: number | null;
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
  ensureParsedForRuntime: () => boolean;
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

function parseCsvText(args: { csvText: string; hasHeaders: boolean }): { ok: true; parsed: ParsedDataset } | { ok: false; error: string } {
  const res = Papa.parse(args.csvText, {
    skipEmptyLines: true,
    dynamicTyping: true
  });
  if (res.errors.length > 0) return { ok: false, error: res.errors[0]?.message ?? "CSV parse error" };

  const raw = res.data as unknown[];
  if (raw.length === 0) return { ok: false, error: "empty CSV" };

  let headers: string[] = [];
  let dataRows: unknown[] = raw;
  if (args.hasHeaders) {
    const h0 = raw[0] as unknown;
    if (!Array.isArray(h0)) return { ok: false, error: "expected header row to be an array" };
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
  if (rows.length === 0) return { ok: false, error: "no numeric data rows parsed" };

  const nCols = rows[0].length;
  if (headers.length !== nCols) {
    headers = Array.from({ length: nCols }, (_, i) => `col_${i}`);
  }

  return { ok: true, parsed: { headers, rows } };
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
    evalsPerSecond: null,
    front: [],
    selectedId: null,
    selectedComplexity: null,
    evalByKey: {}
  },

  setTab: (tab) => set({ tab }),

  loadWasmMetadata: async () => {
    if (get().wasmLoaded) return;
    try {
      await initWasm();

      const [ops, defaults, coreDefaults] = await Promise.all([
        builtin_operator_registry(),
        default_search_options(),
        default_core_options()
      ]);

      const operatorRegistry = ops as WasmOpInfo[];
      const uiDefaults = defaults as any;
      const engineKnobDefaults = coreDefaults as any;
      const defaultOptions = { ...uiDefaults, ...engineKnobDefaults } as any as WasmSearchOptions;
      delete (defaultOptions as any).core;
      set({
        wasmLoaded: true,
        operatorRegistry,
        defaultOptions,
        options: defaultOptions
      });

      // Default operator selection (a safe "basic" set).
      get().applyPreset("basic");
    } catch (err) {
      set((s) => ({
        runtime: {
          ...s.runtime,
          status: "error",
          error: `WASM init failed: ${String(err)}`
        }
      }));
    }
  },

  setCsvText: (csvText) => set({ csvText }),

  parseCsv: () => {
    const { csvText, options } = get();
    if (!options) return;

    const parsedRes = parseCsvText({ csvText, hasHeaders: Boolean(options.has_headers) });
    if (!parsedRes.ok) {
      set((s) => ({ runtime: { ...s.runtime, status: "error", error: parsedRes.error } }));
      return;
    }
    const { headers, rows } = parsedRes.parsed;
    const nCols = rows[0]?.length ?? 0;

    // Defaults: y is last col, X is all others.
    const y = options.y_column ?? (nCols - 1);
    const weights = options.weights_column ?? null;
    const x =
      options.x_columns ??
      Array.from({ length: nCols }, (_, i) => i).filter((i) => i !== y && i !== weights);

    set({
      parsed: parsedRes.parsed,
      options: {
        ...options,
        y_column: y,
        weights_column: weights,
        x_columns: x
      }
    });
  },

  ensureParsedForRuntime: () => {
    const { parsed, csvText, options } = get();
    if (parsed) return true;
    if (!options) return false;
    const parsedRes = parseCsvText({ csvText, hasHeaders: Boolean(options.has_headers) });
    if (!parsedRes.ok) {
      set((s) => ({ runtime: { ...s.runtime, status: "error", error: parsedRes.error } }));
      return false;
    }
    set({ parsed: parsedRes.parsed });
    return true;
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
      // Initial load uses "basic": cos plus +, -, *, /.
      const basicUnary: string[] = ["cos"];
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
