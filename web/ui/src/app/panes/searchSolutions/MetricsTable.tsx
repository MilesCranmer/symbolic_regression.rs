import React, { useState } from "react";
import type { WasmEvalResult } from "../../../types/srTypes";
import { formatSci } from "./plotUtils";

export function MetricsTable({ m }: { m: WasmEvalResult["metrics"] }): React.ReactElement {
  const [showMore, setShowMore] = useState(false);

  return (
    <div>
      <table className="table tight">
        <tbody>
          <tr>
            <td>n</td>
            <td className="mono num">{m.n}</td>
          </tr>
          <tr>
            <td>rmse</td>
            <td className="mono num">{formatSci(m.rmse)}</td>
          </tr>
          <tr>
            <td>mae</td>
            <td className="mono num">{formatSci(m.mae)}</td>
          </tr>
          <tr>
            <td>r²</td>
            <td className="mono num">{formatSci(m.r2)}</td>
          </tr>
          <tr>
            <td>max |err|</td>
            <td className="mono num">{formatSci(m.max_abs_err)}</td>
          </tr>
          {showMore && (
            <>
              <tr>
                <td>mse</td>
                <td className="mono num">{formatSci(m.mse)}</td>
              </tr>
              <tr>
                <td>corr</td>
                <td className="mono num">{formatSci(m.corr)}</td>
              </tr>
              <tr>
                <td>min |err|</td>
                <td className="mono num">{formatSci(m.min_abs_err)}</td>
              </tr>
            </>
          )}
        </tbody>
      </table>

      <div className="row" style={{ marginTop: 6 }}>
        <button onClick={() => setShowMore((v) => !v)}>{showMore ? "Hide" : "More"}</button>
      </div>
    </div>
  );
}

