import React, { useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { SrWorkerClient } from "../../worker/srWorkerClient";
import { getParetoPoints, getSelectedSummary, useSessionStore } from "../../state/sessionStore";
import type { EquationSummary, SearchSnapshot, WasmEvalResult, WasmSplitIndices } from "../../types/srTypes";

function formatSci(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  return x.toExponential(3);
}

function copyToClipboard(text: string): void {
  void navigator.clipboard.writeText(text);
}

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

function sortXY(x: number[], y: number[]): { x: number[]; y: number[] } {
  const pairs = x.map((xi, i) => ({ x: xi, y: y[i] ?? NaN })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  pairs.sort((a, b) => a.x - b.x);
  return { x: pairs.map((p) => p.x), y: pairs.map((p) => p.y) };
}

type FitPlotMode = "auto" | "curve_1d" | "parity";

export function SearchSolutions(): React.ReactElement {
  const parsed = useSessionStore((s) => s.parsed);
  const options = useSessionStore((s) => s.options);
  const csvText = useSessionStore((s) => s.csvText);
  const unaryOps = useSessionStore((s) => s.unaryOps);
  const binaryOps = useSessionStore((s) => s.binaryOps);
  const ternaryOps = useSessionStore((s) => s.ternaryOps);

  const runtime = useSessionStore((s) => s.runtime);
  const setRuntime = useSessionStore((s) => s.setRuntime);
  const setSnapshot = useSessionStore((s) => s.setSnapshot);
  const setFront = useSessionStore((s) => s.setFront);
  const setSelectedId = useSessionStore((s) => s.setSelectedId);
  const setEvalResult = useSessionStore((s) => s.setEvalResult);

  const clientRef = useRef<SrWorkerClient | null>(null);
  const [snapshotHz, setSnapshotHz] = useState(5);
  const [paretoK, setParetoK] = useState(250);
  const [frontK, setFrontK] = useState(50);
  const [fitMode, setFitMode] = useState<FitPlotMode>("auto");

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

  const evalTrain = runtime.selectedId != null ? runtime.evalByKey[`${runtime.selectedId}:train`] : undefined;
  const evalVal = runtime.selectedId != null ? runtime.evalByKey[`${runtime.selectedId}:val`] : undefined;

  useEffect(() => {
    const c = new SrWorkerClient();
    clientRef.current = c;
    c.setHandlers({
      onReady: (splitMaybe) => {
        setRuntime({ status: "ready", error: null, split: splitMaybe as WasmSplitIndices });
      },
      onSnapshot: (snapMaybe) => {
        setSnapshot(snapMaybe as SearchSnapshot);
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
      onResetDone: () => setRuntime({ status: "idle", split: null, snapshot: null, front: [], selectedId: null, evalByKey: {}, error: null }),
      onError: (err) => setRuntime({ status: "error", error: err })
    });
    return () => c.terminate();
  }, [setEvalResult, setFront, setRuntime, setSnapshot]);

  const canInit = Boolean(options) && unaryOps.length + binaryOps.length + ternaryOps.length > 0;

  const initSearch = () => {
    if (!clientRef.current || !options) return;
    setRuntime({ status: "initializing", error: null, split: null, snapshot: null, front: [], selectedId: null, evalByKey: {} });
    clientRef.current.setSnapshotRate(Math.max(50, Math.floor(1000 / Math.max(1, snapshotHz))));
    clientRef.current.setParetoK(paretoK);
    clientRef.current.init({
      csvText,
      options,
      unary: unaryOps,
      binary: binaryOps,
      ternary: ternaryOps,
      paretoK,
      frontK
    });
  };

  const start = () => {
    clientRef.current?.start();
    setRuntime({ status: "running" });
  };
  const pause = () => clientRef.current?.pause();
  const reset = () => clientRef.current?.reset();

  const selectEquation = (id: string) => {
    setSelectedId(id);
    if (!clientRef.current) return;
    const reqTrain = `${crypto.randomUUID()}:${id}:train`;
    clientRef.current.evaluate(reqTrain, id, "train");
    if (split && split.val.length > 0) {
      const reqVal = `${crypto.randomUUID()}:${id}:val`;
      clientRef.current.evaluate(reqVal, id, "val");
    }
  };

  const trainActual = split ? gatherByIndices(yAll, split.train) : yAll;
  const valActual = split ? gatherByIndices(yAll, split.val) : [];

  const trainYhat = evalTrain?.yhat ?? [];
  const valYhat = evalVal?.yhat ?? [];

  const canCurve1d = Boolean(options?.x_columns && options.x_columns.length === 1 && parsed);
  const effectiveFitMode: FitPlotMode = fitMode === "auto" ? (canCurve1d ? "curve_1d" : "parity") : fitMode;

  const trainXY = useMemo(() => {
    if (!split || xAll.length === 0) return { x: [], y: [] };
    return gatherPairsByIndices(xAll, yAll, split.train);
  }, [split, xAll, yAll]);
  const valXY = useMemo(() => {
    if (!split || xAll.length === 0) return { x: [], y: [] };
    return gatherPairsByIndices(xAll, yAll, split.val);
  }, [split, xAll, yAll]);

  return (
    <div className="pane">
      <div className="card">
        <div className="cardTitle">Controls</div>
        <div className="row">
          <button onClick={initSearch} disabled={!canInit} data-testid="search-init">
            Initialize
          </button>
          <button
            onClick={start}
            disabled={runtime.status !== "ready" && runtime.status !== "paused"}
            data-testid="search-start"
          >
            Start / Resume
          </button>
          <button onClick={pause} disabled={runtime.status !== "running"}>
            Pause
          </button>
          <button onClick={reset}>Reset</button>

          <label className="field">
            <div className="label">snapshot Hz</div>
            <input type="number" min={1} max={30} value={snapshotHz} onChange={(e) => setSnapshotHz(Number(e.target.value))} />
          </label>
          <label className="field">
            <div className="label">pareto K</div>
            <input type="number" min={20} value={paretoK} onChange={(e) => setParetoK(Number(e.target.value))} />
          </label>
          <label className="field">
            <div className="label">solutions K</div>
            <input type="number" min={10} value={frontK} onChange={(e) => setFrontK(Number(e.target.value))} />
          </label>

          <label className="field">
            <div className="label">fit plot</div>
            <select value={fitMode} onChange={(e) => setFitMode(e.target.value as FitPlotMode)}>
              <option value="auto">Auto</option>
              <option value="parity">Parity (y vs ŷ)</option>
              <option value="curve_1d" disabled={!canCurve1d}>
                1D curve (x vs y)
              </option>
            </select>
          </label>

          <div className="statusLine">
            <span className="statusChip" data-testid="search-status">
              {runtime.status}
            </span>
            {runtime.error && <span className="errorText">{runtime.error}</span>}
            {snap && (
              <span className="muted">
                cycles {snap.cycles_completed}/{snap.total_cycles} ({snap.total_cycles > 0 ? ((100 * snap.cycles_completed) / snap.total_cycles).toFixed(1) : "0"}%), evals={snap.total_evals}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="grid4">
        <div className="card gridCell">
          <div className="cardTitle">Current solutions</div>
          <div className="tableWrap">
            <table className="table" data-testid="solutions-table">
              <thead>
                <tr>
                  <th>size</th>
                  <th>loss</th>
                  <th>equation</th>
                </tr>
              </thead>
              <tbody>
                {runtime.front
                  .slice()
                  .reverse()
                  .map((m) => (
                    <tr
                      key={m.id}
                      className={m.id === runtime.selectedId ? "selectedRow" : ""}
                      onClick={() => selectEquation(m.id)}
                      data-testid={`solution-row-${m.id}`}
                    >
                      <td className="mono">{m.complexity}</td>
                      <td className="mono">{formatSci(m.loss)}</td>
                      <td className="mono">{m.equation}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card gridCell">
          <div className="cardTitle">Selected solution fit</div>
          {!selectedSummary ? (
            <div className="muted">Select a solution.</div>
          ) : (
            <FitPlot
              mode={effectiveFitMode}
              hasVal={Boolean(split && split.val.length > 0)}
              trainActual={trainActual}
              valActual={valActual}
              trainYhat={trainYhat}
              valYhat={valYhat}
              trainXY={trainXY}
              valXY={valXY}
            />
          )}
        </div>

        <div className="card gridCell">
          <div className="cardTitle">Quick stats</div>
          {!selectedSummary ? (
            <div className="muted">Select a solution to compute metrics.</div>
          ) : (
            <>
              <div className="mono bigEq" data-testid="selected-equation">
                {selectedSummary.equation}
              </div>
              <div className="row">
                <button onClick={() => copyToClipboard(selectedSummary.equation)}>Copy equation</button>
              </div>
              <div className="subTitle">Train</div>
              {evalTrain ? (
                <MetricsTable m={evalTrain.metrics} />
              ) : (
                <div className="muted" data-testid="no-metrics">
                  No metrics yet (click solution to evaluate).
                </div>
              )}
              {split && split.val.length > 0 && (
                <>
                  <div className="subTitle">Validation</div>
                  {evalVal ? <MetricsTable m={evalVal.metrics} /> : <div className="muted">No metrics yet.</div>}
                </>
              )}
            </>
          )}
        </div>

        <div className="card gridCell">
          <div className="cardTitle">Live Pareto front</div>
          {points.length === 0 ? (
            <div className="muted">No points yet.</div>
          ) : (
            <Plot
              data={[
                {
                  x: points.map((p) => p.complexity),
                  y: points.map((p) => p.loss),
                  text: points.map((p) => String(p.id)),
                  type: "scatter",
                  mode: "markers",
                  marker: {
                    size: points.map((p) => (p.id === runtime.selectedId ? 10 : 6)),
                    color: points.map((p) => (p.id === runtime.selectedId ? "#ffd166" : "#2ec4b6"))
                  }
                } as any
              ]}
              layout={{
                autosize: true,
                margin: { l: 50, r: 20, t: 20, b: 50 },
                xaxis: { title: "complexity" },
                yaxis: { title: "loss" }
              }}
              style={{ width: "100%", height: "100%" }}
              config={{ displayModeBar: false, responsive: true }}
              onClick={(ev) => {
                const idx = ev.points?.[0]?.pointIndex;
                if (idx == null) return;
                const p = points[idx as number];
                if (p) selectEquation(p.id);
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function FitPlot(props: {
  mode: FitPlotMode;
  hasVal: boolean;
  trainActual: number[];
  valActual: number[];
  trainYhat: number[];
  valYhat: number[];
  trainXY: { x: number[]; y: number[] };
  valXY: { x: number[]; y: number[] };
}): React.ReactElement {
  if (props.mode === "curve_1d") {
    const trainCurve = sortXY(props.trainXY.x, props.trainYhat);
    const valCurve = sortXY(props.valXY.x, props.valYhat);
    return (
      <Plot
        data={[
          {
            x: props.trainXY.x,
            y: props.trainXY.y,
            type: "scatter",
            mode: "markers",
            name: "train data",
            marker: { size: 6, color: "#4f7cff", opacity: 0.7 }
          },
          {
            x: trainCurve.x,
            y: trainCurve.y,
            type: "scatter",
            mode: "lines",
            name: "train model",
            line: { color: "#4f7cff", width: 2 }
          },
          ...(props.hasVal
            ? ([
                {
                  x: props.valXY.x,
                  y: props.valXY.y,
                  type: "scatter",
                  mode: "markers",
                  name: "val data",
                  marker: { size: 6, color: "#ff7c7c", opacity: 0.7 }
                },
                {
                  x: valCurve.x,
                  y: valCurve.y,
                  type: "scatter",
                  mode: "lines",
                  name: "val model",
                  line: { color: "#ff7c7c", width: 2 }
                }
              ] as any[])
            : [])
        ]}
        layout={{
          autosize: true,
          margin: { l: 50, r: 20, t: 20, b: 50 },
          xaxis: { title: "x" },
          yaxis: { title: "y" }
        }}
        style={{ width: "100%", height: "100%" }}
        config={{ displayModeBar: false, responsive: true }}
      />
    );
  }

  return (
    <Plot
      data={[
        {
          x: props.trainActual,
          y: props.trainYhat,
          type: "scatter",
          mode: "markers",
          name: "train",
          marker: { size: 6, color: "#4f7cff" }
        },
        ...(props.hasVal
          ? ([
              {
                x: props.valActual,
                y: props.valYhat,
                type: "scatter",
                mode: "markers",
                name: "val",
                marker: { size: 6, color: "#ff7c7c" }
              } as any
            ] as any[])
          : [])
      ]}
      layout={{
        autosize: true,
        margin: { l: 50, r: 20, t: 20, b: 50 },
        xaxis: { title: "y (actual)" },
        yaxis: { title: "ŷ (predicted)" }
      }}
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}

function MetricsTable({ m }: { m: WasmEvalResult["metrics"] }): React.ReactElement {
  return (
    <table className="table tight">
      <tbody>
        <tr>
          <td>n</td>
          <td className="mono">{m.n}</td>
        </tr>
        <tr>
          <td>mse</td>
          <td className="mono">{formatSci(m.mse)}</td>
        </tr>
        <tr>
          <td>mae</td>
          <td className="mono">{formatSci(m.mae)}</td>
        </tr>
        <tr>
          <td>rmse</td>
          <td className="mono">{formatSci(m.rmse)}</td>
        </tr>
        <tr>
          <td>r2</td>
          <td className="mono">{formatSci(m.r2)}</td>
        </tr>
        <tr>
          <td>corr</td>
          <td className="mono">{formatSci(m.corr)}</td>
        </tr>
        <tr>
          <td>min |err|</td>
          <td className="mono">{formatSci(m.min_abs_err)}</td>
        </tr>
        <tr>
          <td>max |err|</td>
          <td className="mono">{formatSci(m.max_abs_err)}</td>
        </tr>
      </tbody>
    </table>
  );
}
