import React from "react";
import type { EquationSummary, WasmEvalResult } from "../../../types/srTypes";
import { copyToClipboard, formatSci } from "./plotUtils";
import { MetricsTable } from "./MetricsTable";

export function QuickStatsCard(props: {
  selectedSummary: EquationSummary | null;
  evalTrain?: WasmEvalResult;
  evalVal?: WasmEvalResult;
  hasVal: boolean;
}): React.ReactElement {
  return (
    <div className="card gridCell resultsCard resultsCard--stats">
      <div className="cardTitle">Quick stats</div>
      {!props.selectedSummary ? (
        <div className="muted">Select a solution to compute metrics.</div>
      ) : (
        <>
          <div className="eqHeader">
            <div className="mono monoWrap bigEq" data-testid="selected-equation">
              {props.selectedSummary.equation}
            </div>
            <button onClick={() => copyToClipboard(props.selectedSummary!.equation)}>Copy</button>
          </div>
          <div className="eqMeta">
            <span className="eqMetaItem">
              <strong>complexity</strong> {props.selectedSummary.complexity}
            </span>
            <span className="eqMetaItem">
              <strong>loss</strong> {formatSci(props.selectedSummary.loss)}
            </span>
          </div>
          <div className="sectionLabel">Train</div>
          {props.evalTrain ? (
            <MetricsTable m={props.evalTrain.metrics} />
          ) : (
            <div className="muted" data-testid="no-metrics">
              No metrics yet (click solution to evaluate).
            </div>
          )}
          {props.hasVal && (
            <>
              <div className="sectionLabel">Validation</div>
              {props.evalVal ? <MetricsTable m={props.evalVal.metrics} /> : <div className="muted">No metrics yet.</div>}
            </>
          )}
        </>
      )}
    </div>
  );
}
