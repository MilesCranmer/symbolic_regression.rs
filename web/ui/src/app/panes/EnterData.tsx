import React from "react";
import { useMemo, useRef, useState } from "react";
import { useSessionStore } from "../../state/sessionStore";

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v | 0));
}

export function EnterData(): React.ReactElement {
  const csvText = useSessionStore((s) => s.csvText);
  const setCsvText = useSessionStore((s) => s.setCsvText);
  const parsed = useSessionStore((s) => s.parsed);
  const options = useSessionStore((s) => s.options);
  const setOptionsPatch = useSessionStore((s) => s.setOptionsPatch);
  const parseCsv = useSessionStore((s) => s.parseCsv);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const nCols = parsed?.headers.length ?? 0;
  const colOptions = useMemo(() => {
    const h = parsed?.headers ?? [];
    return h.map((name, idx) => ({ idx, name }));
  }, [parsed]);

  const yCol = options?.y_column ?? null;
  const wCol = options?.weights_column ?? null;
  const xCols = options?.x_columns ?? null;

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

      <div className="card">
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
    </div>
  );
}
