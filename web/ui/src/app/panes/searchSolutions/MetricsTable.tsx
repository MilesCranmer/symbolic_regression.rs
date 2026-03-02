import React, { useState } from "react";
import type { WasmEvalResult } from "../../../types/srTypes";
import { formatSci } from "./plotUtils";

export function MetricsTable({ m }: { m: WasmEvalResult["metrics"] }): React.ReactElement {
  const [showMore, setShowMore] = useState(false);

  return (
    <div>
      <div className="metricsGrid">
        <div className="metricCard">
          <div className="metricLabel">n</div>
          <div className="metricValue">{m.n}</div>
        </div>
        <div className="metricCard">
          <div className="metricLabel">rmse</div>
          <div className="metricValue">{formatSci(m.rmse)}</div>
        </div>
        <div className="metricCard">
          <div className="metricLabel">mae</div>
          <div className="metricValue">{formatSci(m.mae)}</div>
        </div>
        <div className="metricCard">
          <div className="metricLabel">r²</div>
          <div className="metricValue">{formatSci(m.r2)}</div>
        </div>
        <div className="metricCard">
          <div className="metricLabel">max |err|</div>
          <div className="metricValue">{formatSci(m.max_abs_err)}</div>
        </div>
        {showMore && (
          <>
            <div className="metricCard">
              <div className="metricLabel">mse</div>
              <div className="metricValue">{formatSci(m.mse)}</div>
            </div>
            <div className="metricCard">
              <div className="metricLabel">corr</div>
              <div className="metricValue">{formatSci(m.corr)}</div>
            </div>
            <div className="metricCard">
              <div className="metricLabel">min |err|</div>
              <div className="metricValue">{formatSci(m.min_abs_err)}</div>
            </div>
          </>
        )}
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <button onClick={() => setShowMore((v) => !v)}>{showMore ? "Hide" : "More metrics"}</button>
      </div>
    </div>
  );
}
