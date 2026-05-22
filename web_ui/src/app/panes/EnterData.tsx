import React, { useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { useSessionStore } from "../../state/sessionStore";
import { formatSci, plotLayoutBase, usePrefersDark } from "./searchSolutions/plotUtils";

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v | 0));
}

export function EnterData(): React.ReactElement {
  const prefersDark = usePrefersDark();
  const csvText = useSessionStore((s) => s.csvText);
  const setCsvText = useSessionStore((s) => s.setCsvText);
  const parsed = useSessionStore((s) => s.parsed);
  const options = useSessionStore((s) => s.options);
  const setOptionsPatch = useSessionStore((s) => s.setOptionsPatch);
  const parseCsv = useSessionStore((s) => s.parseCsv);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const headers = parsed?.headers ?? [];
  const rows = parsed?.rows ?? [];

  const nCols = parsed?.headers.length ?? 0;
  const colOptions = useMemo(() => {
    const h = parsed?.headers ?? [];
    return h.map((name, idx) => ({ idx, name }));
  }, [parsed]);

  const yCol = options?.y_column ?? null;
  const wCol = options?.weights_column ?? null;
  const xCols = options?.x_columns ?? null;

  const yColIdx = options?.y_column ?? (headers.length > 0 ? headers.length - 1 : 0);
  const xColsForPlot = options?.x_columns ?? [];
  const xColsForPlotKey = xColsForPlot.join(",");
  const [inspectX, setInspectX] = useState<number>(() => (xColsForPlot.length > 0 ? xColsForPlot[0] : 0));

  useEffect(() => {
    if (xColsForPlot.length === 0) return;
    if (!xColsForPlot.includes(inspectX)) setInspectX(xColsForPlot[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xColsForPlotKey, inspectX]);

  const scatter = useMemo(() => {
    if (!parsed || xColsForPlot.length === 0) return { x: [], y: [] };
    const x: number[] = [];
    const y: number[] = [];
    for (const r of rows) {
      const xv = r[inspectX];
      const yv = r[yColIdx];
      if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
      x.push(xv);
      y.push(yv);
    }
    return { x, y };
  }, [parsed, rows, inspectX, yColIdx, xColsForPlotKey]);

  const columnSummary = useMemo(() => {
    if (!parsed) return [] as { name: string; mean: number; std: number; min: number; max: number }[];
    const nCols2 = headers.length;
    const sum = Array.from({ length: nCols2 }, () => 0);
    const sumsq = Array.from({ length: nCols2 }, () => 0);
    const min = Array.from({ length: nCols2 }, () => Infinity);
    const max = Array.from({ length: nCols2 }, () => -Infinity);
    const count = Array.from({ length: nCols2 }, () => 0);

    for (const r of rows) {
      for (let j = 0; j < nCols2; j++) {
        const v = r[j];
        if (!Number.isFinite(v)) continue;
        sum[j] += v;
        sumsq[j] += v * v;
        count[j] += 1;
        if (v < min[j]) min[j] = v;
        if (v > max[j]) max[j] = v;
      }
    }

    return headers.map((name, j) => {
      const n = count[j] ?? 0;
      const mu = n > 0 ? sum[j]! / n : NaN;
      const var0 = n > 0 ? sumsq[j]! / n - mu * mu : NaN;
      const std = Number.isFinite(var0) && var0 >= 0 ? Math.sqrt(var0) : NaN;
      return { name: name ?? `col_${j}`, mean: mu, std, min: min[j]!, max: max[j]! };
    });
  }, [parsed, headers, rows]);

  const onFile = async (f: File | null) => {
    if (!f) return;
    const text = await f.text();
    setCsvText(text);
    // Convenience: auto-parse after upload to show preview immediately.
    parseCsv();
  };

  return (
    <div className="pane">
      <div className="card">
        <div className="cardTitle">CSV</div>
        <div className="row">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={options?.has_headers ?? true}
              onChange={(e) => setOptionsPatch({ has_headers: e.target.checked })}
            />
            has header row
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="hiddenFileInput"
            onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
          />
          <button onClick={() => fileInputRef.current?.click()}>Choose file…</button>
          <button onClick={parseCsv} disabled={!options} data-testid="parse-preview">
            Parse / Preview
          </button>
        </div>

        <div
          className={dragActive ? "dropzone active" : "dropzone"}
          data-testid="csv-dropzone"
          onDragEnter={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(false);
            const f = e.dataTransfer.files?.[0] ?? null;
            void onFile(f);
          }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
        >
          <div className="dropzoneTitle">Drag & drop a CSV here</div>
          <div className="muted">or click to choose a file</div>
        </div>

        <textarea className="textarea" value={csvText} onChange={(e) => setCsvText(e.target.value)} rows={10} />
      </div>

      <div className="card">
        <div className="cardTitle">Columns</div>
        {!parsed ? (
          <div className="muted">Click “Parse / Preview” to configure columns.</div>
        ) : (
          <>
            <div className="row">
              <label className="field">
                <div className="label">y column</div>
                <select
                  value={String(yCol ?? nCols - 1)}
                  onChange={(e) => {
                    const y = clampInt(Number(e.target.value), 0, nCols - 1);
                    const weights = wCol === y ? null : wCol;
                    const x = (xCols ?? []).filter((c) => c !== y && c !== weights);
                    setOptionsPatch({ y_column: y, weights_column: weights, x_columns: x });
                  }}
                >
                  {colOptions.map((c) => (
                    <option key={c.idx} value={c.idx}>
                      {c.idx}: {c.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <div className="label">weights column (optional)</div>
                <select
                  value={wCol == null ? "" : String(wCol)}
                  onChange={(e) => {
                    const v = e.target.value;
                    const weights = v === "" ? null : clampInt(Number(v), 0, nCols - 1);
                    const y = yCol ?? (nCols - 1);
                    const x = (xCols ?? []).filter((c) => c !== y && c !== weights);
                    setOptionsPatch({ weights_column: weights, x_columns: x });
                  }}
                >
                  <option value="">(none)</option>
                  {colOptions.map((c) => (
                    <option key={c.idx} value={c.idx} disabled={c.idx === (yCol ?? nCols - 1)}>
                      {c.idx}: {c.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <div className="label">validation fraction</div>
                <input
                  type="number"
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={options?.validation_fraction ?? 0}
                  onChange={(e) => setOptionsPatch({ validation_fraction: Number(e.target.value) })}
                />
              </label>
            </div>

            <div className="subTitle">X columns</div>
            <div className="checkboxGrid">
              {colOptions.map((c) => {
                const y = yCol ?? (nCols - 1);
                const isDisabled = c.idx === y || c.idx === (wCol ?? -1);
                const isChecked = (xCols ?? []).includes(c.idx) && !isDisabled;
                return (
                  <label key={c.idx} className={isDisabled ? "checkbox disabled" : "checkbox"}>
                    <input
                      type="checkbox"
                      disabled={isDisabled}
                      checked={isChecked}
                      onChange={() => {
                        const cur = xCols ?? [];
                        const next = cur.includes(c.idx) ? cur.filter((i) => i !== c.idx) : [...cur, c.idx];
                        setOptionsPatch({ x_columns: next.sort((a, b) => a - b) });
                      }}
                    />
                    {c.idx}: {c.name}
                  </label>
                );
              })}
            </div>

            <div className="row">
              <div className="muted">
                Parsed {parsed.rows.length} rows × {parsed.headers.length} columns.
              </div>
            </div>
          </>
        )}
      </div>

      <div className="grid2">
        <div className="card gridCell">
          <div className="cardTitle">Preview</div>
          {!parsed ? (
            <div className="muted">No preview yet.</div>
          ) : (
            <div className="tableWrap">
              <table className="table">
                <thead>
                  <tr>
                    {parsed.headers.slice(0, 12).map((h, i) => (
                      <th key={i}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {parsed.rows.slice(0, 12).map((r, i) => (
                    <tr key={i}>
                      {r.slice(0, 12).map((v, j) => (
                        <td key={j}>{Number.isFinite(v) ? v.toFixed(4) : String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card gridCell">
          <div className="cardTitle">X vs y</div>
          {!parsed ? (
            <div className="muted">Parse a CSV to see plots.</div>
          ) : xColsForPlot.length === 0 ? (
            <div className="muted">Select one or more X columns above.</div>
          ) : (
            <>
              <div className="row" style={{ marginBottom: 8 }}>
                <label className="field" style={{ minWidth: 260 }}>
                  <div className="label">x column</div>
                  <select value={inspectX} onChange={(e) => setInspectX(Number(e.target.value))}>
                    {xColsForPlot.map((j) => (
                      <option key={j} value={j}>
                        {headers[j] ?? `col_${j}`}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="plotWrap">
                <Plot
                  data={[
                    {
                      x: scatter.x,
                      y: scatter.y,
                      type: "scatter",
                      mode: "markers",
                      marker: { size: 6, opacity: 0.7 }
                    } as any
                  ]}
                  layout={{
                    ...plotLayoutBase(prefersDark),
                    autosize: true,
                    margin: { l: 50, r: 20, t: 10, b: 50 },
                    xaxis: { ...(plotLayoutBase(prefersDark) as any).xaxis, title: headers[inspectX] ?? `col_${inspectX}` },
                    yaxis: { ...(plotLayoutBase(prefersDark) as any).yaxis, title: headers[yColIdx] ?? `col_${yColIdx}` }
                  }}
                  style={{ width: "100%", height: "100%" }}
                  config={{ displayModeBar: false, responsive: true }}
                />
              </div>
            </>
          )}
        </div>
      </div>

      <div className="card">
        <div className="cardTitle">Column summary</div>
        {!parsed ? (
          <div className="muted">Parse a CSV to compute column summary statistics.</div>
        ) : (
          <div className="tableWrap" style={{ maxHeight: 320 }}>
            <table className="table fixed">
              <thead>
                <tr>
                  <th style={{ width: 180 }}>column</th>
                  <th className="num" style={{ width: 120 }}>
                    mean
                  </th>
                  <th className="num" style={{ width: 120 }}>
                    std
                  </th>
                  <th className="num" style={{ width: 120 }}>
                    min
                  </th>
                  <th className="num" style={{ width: 120 }}>
                    max
                  </th>
                </tr>
              </thead>
              <tbody>
                {columnSummary.map((s) => (
                  <tr key={s.name}>
                    <td className="mono monoEllipsis" title={s.name}>
                      {s.name}
                    </td>
                    <td className="mono num">{formatSci(s.mean)}</td>
                    <td className="mono num">{formatSci(s.std)}</td>
                    <td className="mono num">{formatSci(s.min)}</td>
                    <td className="mono num">{formatSci(s.max)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
