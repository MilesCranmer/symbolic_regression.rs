import { useEffect, useMemo, useRef } from "react";
import { getParetoPoints, getSelectedSummary, useSessionStore } from "../../../state/sessionStore";
import type { EquationSummary, SearchSnapshot, WasmEvalResult, WasmSplitIndices } from "../../../types/srTypes";
import type { FitPlotMode } from "./types";

type SrWorkerClientLike = {
  setHandlers: (handlers: {
    onReady: (splitMaybe: unknown) => void;
    onSnapshot: (snapMaybe: unknown) => void;
    onFrontUpdate: (frontMaybe: unknown) => void;
    onEvalResult: (requestId: string, resultMaybe: unknown) => void;
    onDone: () => void;
    onPaused: () => void;
    onResetDone: () => void;
    onError: (err: string) => void;
  }) => void;
  terminate: () => void;
  init: (args: {
    csvText: string;
    options: any;
    unary: any;
    binary: any;
    ternary: any;
  }) => void;
  start: () => void;
  pause: () => void;
  reset: () => void;
  evaluate: (requestId: string, memberId: string, which: "train" | "val") => void;
};

function buildYArray(rows: number[][], yCol: number): number[] {
  return rows.map((r) => r[yCol]);
}

function buildXArray(rows: number[][], xCol: number): number[] {
  return rows.map((r) => r[xCol]);
}

function gatherByIndices(values: number[], idx: number[]): number[] {
  return idx.map((i) => values[i]).filter((v) => Number.isFinite(v));
}

function gatherPairsByIndices(x: number[], y: number[], idx: number[]): { x: number[]; y: number[] } {
  const xo: number[] = [];
  const yo: number[] = [];
  for (const i of idx) {
    const xv = x[i];
    const yv = y[i];
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
    xo.push(xv);
    yo.push(yv);
  }
  return { x: xo, y: yo };
}

export function useSearchController(Client: { new (): SrWorkerClientLike }) {
  const parsed = useSessionStore((s) => s.parsed);
  const options = useSessionStore((s) => s.options);
  const csvText = useSessionStore((s) => s.csvText);
  const ensureParsedForRuntime = useSessionStore((s) => s.ensureParsedForRuntime);
  const unaryOps = useSessionStore((s) => s.unaryOps);
  const binaryOps = useSessionStore((s) => s.binaryOps);
  const ternaryOps = useSessionStore((s) => s.ternaryOps);

  const runtime = useSessionStore((s) => s.runtime);
  const setRuntime = useSessionStore((s) => s.setRuntime);
  const setSnapshot = useSessionStore((s) => s.setSnapshot);
  const setFront = useSessionStore((s) => s.setFront);
  const setSelection = useSessionStore((s) => s.setSelection);
  const setEvalResult = useSessionStore((s) => s.setEvalResult);

  const clientRef = useRef<SrWorkerClientLike | null>(null);
  const lastSnapTimeRef = useRef<number | null>(null);
  const lastEvalsRef = useRef<number | null>(null);
  const cpsEmaRef = useRef<number | null>(null);

  const snap = runtime.snapshot;
  const points = getParetoPoints(snap);
  const best = snap?.best ?? null;
  const selectedSummary = getSelectedSummary(runtime.front, best, runtime.selectedId);

  const yCol = options?.y_column ?? (parsed ? parsed.headers.length - 1 : 0);
  const yAll = useMemo(() => (parsed ? buildYArray(parsed.rows, yCol) : []), [parsed, yCol]);
  const xSelectedCol = options?.x_columns && options.x_columns.length > 0 ? options.x_columns[0] : null;
  const xAll = useMemo(
    () => (parsed && xSelectedCol != null ? buildXArray(parsed.rows, xSelectedCol) : []),
    [parsed, xSelectedCol]
  );
  const split = runtime.split;
  // This is the *requested* complexity (what the user clicked). The displayed equation may
  // temporarily fall back to a lower complexity if the requested one is not currently
  // present on the front.
  const requestedComplexity = runtime.selectedComplexity;

  const evalTrain = runtime.selectedId != null ? runtime.evalByKey[`${runtime.selectedId}:train`] : undefined;
  const evalVal = runtime.selectedId != null ? runtime.evalByKey[`${runtime.selectedId}:val`] : undefined;

  useEffect(() => {
    if (parsed) return;
    if (!options) return;
    if (!csvText.trim()) return;
    if (runtime.status === "idle" || runtime.status === "initializing" || runtime.status === "error") return;
    ensureParsedForRuntime();
  }, [parsed, options, csvText, runtime.status, ensureParsedForRuntime]);

  useEffect(() => {
    const c = new Client();
    clientRef.current = c;
    c.setHandlers({
      onReady: (splitMaybe) => {
        setRuntime({ status: "ready", error: null, split: splitMaybe as WasmSplitIndices });
      },
      onSnapshot: (snapMaybe) => {
        const s = snapMaybe as SearchSnapshot;
        setSnapshot(s);

        const now = performance.now();
        const prevT = lastSnapTimeRef.current;
        const prevE = lastEvalsRef.current;

        if (prevT != null && prevE != null) {
          const dt = (now - prevT) / 1000;
          const de = s.total_evals - prevE;
          if (dt > 0 && de >= 0) {
            const eps = de / dt;
            const emaPrev = cpsEmaRef.current;
            const ema = emaPrev == null ? eps : 0.2 * eps + 0.8 * emaPrev;
            cpsEmaRef.current = ema;
            setRuntime({ evalsPerSecond: ema });
          }
        }

        lastSnapTimeRef.current = now;
        lastEvalsRef.current = s.total_evals;
      },
      onFrontUpdate: (frontMaybe) => {
        setFront(frontMaybe as EquationSummary[]);
      },
      onEvalResult: (requestId, resultMaybe) => {
        const r = resultMaybe as WasmEvalResult;
        const parts = requestId.split(":");
        if (parts.length !== 3) return;
        const memberId = parts[1];
        const which = parts[2] as "train" | "val";
        setEvalResult(memberId, which, r);
      },
      onDone: () => setRuntime({ status: "done" }),
      onPaused: () => setRuntime({ status: "paused" }),
      onResetDone: () =>
        setRuntime({
          status: "idle",
          split: null,
          snapshot: null,
          evalsPerSecond: null,
          front: [],
          selectedId: null,
          selectedComplexity: null,
          evalByKey: {},
          error: null
        }),
      onError: (err) => {
        // A common non-fatal case: we asked to evaluate an equation ID that has
        // just fallen out of the worker's internal cache between front updates.
        // Don't flip the whole UI into an "error" state for this.
        if (String(err).includes("member not found")) return;
        setRuntime({ status: "error", error: err });
      }
    });
    return () => c.terminate();
  }, [Client, setEvalResult, setFront, setRuntime, setSnapshot]);

  const canInit = Boolean(options) && unaryOps.length + binaryOps.length + ternaryOps.length > 0;

  const initSearch = () => {
    if (!clientRef.current || !options) return;
    setRuntime({
      status: "initializing",
      error: null,
      split: null,
      snapshot: null,
      evalsPerSecond: null,
      front: [],
      selectedId: null,
      selectedComplexity: null,
      evalByKey: {}
    });
    lastSnapTimeRef.current = null;
    lastEvalsRef.current = null;
    cpsEmaRef.current = null;
    clientRef.current.init({
      csvText,
      options,
      unary: unaryOps,
      binary: binaryOps,
      ternary: ternaryOps
    });
  };

  const start = () => {
    clientRef.current?.start();
    setRuntime({ status: "running" });
  };
  const pause = () => clientRef.current?.pause();
  const reset = () => clientRef.current?.reset();

  const selectEquation = (sel: { id: string; complexity: number }, opts?: { updateRequestedComplexity?: boolean }) => {
    const updateRequestedComplexity = opts?.updateRequestedComplexity ?? true;
    if (updateRequestedComplexity) setSelection(sel.id, sel.complexity);
    else setRuntime({ selectedId: sel.id });
    if (!clientRef.current) return;
    const reqTrain = `${crypto.randomUUID()}:${sel.id}:train`;
    clientRef.current.evaluate(reqTrain, sel.id, "train");
    if (split && split.val.length > 0) {
      const reqVal = `${crypto.randomUUID()}:${sel.id}:val`;
      clientRef.current.evaluate(reqVal, sel.id, "val");
    }
  };

  useEffect(() => {
    if (requestedComplexity == null) return;
    if (runtime.front.length === 0) return;

    // Prefer the current best equation at the requested complexity.
    let match = runtime.front
      .slice()
      .reverse()
      .find((m) => m.complexity === requestedComplexity);

    // If nothing exists at the requested complexity, fall back to the *closest lower*
    // complexity to preserve visual continuity, without changing the requested value.
    if (!match) {
      const lower = runtime.front.filter((m) => m.complexity <= requestedComplexity);
      if (lower.length > 0) {
        lower.sort((a, b) => a.complexity - b.complexity);
        match = lower[lower.length - 1];
      }
    }

    // As a last resort (e.g. all candidates are higher complexity), pick the smallest.
    if (!match) {
      const sorted = runtime.front.slice().sort((a, b) => a.complexity - b.complexity);
      match = sorted[0];
    }

    if (!match) return;
    if (match.id === runtime.selectedId) return;

    selectEquation({ id: match.id, complexity: match.complexity }, { updateRequestedComplexity: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runtime.front, requestedComplexity, runtime.selectedId]);

  const trainActual = split ? gatherByIndices(yAll, split.train) : yAll;
  const valActual = split ? gatherByIndices(yAll, split.val) : [];

  const trainYhat = evalTrain?.yhat ?? [];
  const valYhat = evalVal?.yhat ?? [];

  const canCurve1d = Boolean(options?.x_columns && options.x_columns.length === 1 && parsed);
  const effectiveFitMode: FitPlotMode = canCurve1d ? "curve_1d" : "parity";

  const trainXY = useMemo(() => {
    if (!split || xAll.length === 0) return { x: [], y: [] };
    return gatherPairsByIndices(xAll, yAll, split.train);
  }, [split, xAll, yAll]);
  const valXY = useMemo(() => {
    if (!split || xAll.length === 0) return { x: [], y: [] };
    return gatherPairsByIndices(xAll, yAll, split.val);
  }, [split, xAll, yAll]);

  return {
    runtime,
    snap,
    points,
    selectedSummary,
    split,
    evalTrain,
    evalVal,

    canInit,
    canCurve1d,
    effectiveFitMode,
    initSearch,
    start,
    pause,
    reset,
    selectEquation,

    trainActual,
    valActual,
    trainYhat,
    valYhat,
    trainXY,
    valXY
  };
}
