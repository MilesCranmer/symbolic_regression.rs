import React from "react";
import Plot from "react-plotly.js";
import type { EquationSummary } from "../../../types/srTypes";
import { plotLayoutBase } from "./plotUtils";

export function ParetoPlotCard(props: {
  prefersDark: boolean;
  points: { id: string; complexity: number; loss: number }[];
  selectedId: string | null;
  best: EquationSummary | null;
  selectEquation: (sel: { id: string; complexity: number }) => void;
}): React.ReactElement {
  const selectedFill = "#ff7c7c"; // visible in both light/dark
  const selectedLine = props.prefersDark ? "rgba(255,255,255,0.75)" : "rgba(0,0,0,0.55)";

  return (
    <div className="card gridCell resultsCard resultsCard--pareto resultsFixed">
      <div className="cardTitle">Pareto front</div>
      <div className="plotWrap">
        {props.points.length === 0 ? (
          <div className="muted">No points yet.</div>
        ) : (
          <Plot
            data={[
              {
                x: props.points.map((p) => p.complexity),
                y: props.points.map((p) => p.loss),
                text: props.points.map((p) => p.id),
                type: "scatter",
                mode: "markers",
                marker: {
                  size: props.points.map((p) => (p.id === props.selectedId ? 12 : 7)),
                  color: props.points.map((p) => (p.id === props.selectedId ? selectedFill : "rgba(100, 150, 255, 0.7)")),
                  line: props.points.map((p) =>
                    p.id === props.selectedId
                      ? { width: 2, color: selectedLine }
                      : props.best && p.id === props.best.id
                        ? { width: 2, color: "#ffd200" }
                        : { width: 0, color: "rgba(0,0,0,0)" }
                  )
                }
              } as any
            ]}
            layout={{
              ...plotLayoutBase(props.prefersDark),
              autosize: true,
              margin: { l: 50, r: 20, t: 20, b: 50 },
              xaxis: { ...(plotLayoutBase(props.prefersDark) as any).xaxis, title: "complexity" },
              yaxis: { ...(plotLayoutBase(props.prefersDark) as any).yaxis, title: "loss" }
            }}
            style={{ width: "100%", height: "100%" }}
            config={{ displayModeBar: false, responsive: true }}
            onClick={(ev) => {
              const idx = ev.points?.[0]?.pointIndex;
              if (idx == null) return;
              const p = props.points[idx as number];
              if (p) props.selectEquation({ id: p.id, complexity: p.complexity });
            }}
          />
        )}
      </div>
    </div>
  );
}
