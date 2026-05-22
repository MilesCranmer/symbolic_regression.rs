import React from "react";
import type { EquationSummary } from "../../../types/srTypes";
import type { FitPlotMode } from "./types";
import { FitPlot } from "./FitPlot";

export function SelectedFitCard(props: {
  prefersDark: boolean;
  selectedSummary: EquationSummary | null;
  effectiveFitMode: FitPlotMode;
  hasVal: boolean;
  trainActual: number[];
  valActual: number[];
  trainYhat: number[];
  valYhat: number[];
  trainXY: { x: number[]; y: number[] };
  valXY: { x: number[]; y: number[] };
}): React.ReactElement {
  return (
    <div className="card gridCell resultsCard resultsCard--fit resultsFixed">
      <div className="cardTitle">Selected solution fit</div>
      {!props.selectedSummary ? (
        <div className="muted">Select a solution.</div>
      ) : (
        <div className="plotWrap">
          <FitPlot
            prefersDark={props.prefersDark}
            mode={props.effectiveFitMode}
            hasVal={props.hasVal}
            trainActual={props.trainActual}
            valActual={props.valActual}
            trainYhat={props.trainYhat}
            valYhat={props.valYhat}
            trainXY={props.trainXY}
            valXY={props.valXY}
          />
        </div>
      )}
    </div>
  );
}
