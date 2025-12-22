import React from "react";
import Plot from "react-plotly.js";
import type { FitPlotMode } from "./types";
import { plotLayoutBase, sortXY } from "./plotUtils";

export function FitPlot(props: {
  prefersDark: boolean;
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
          ...plotLayoutBase(props.prefersDark),
          autosize: true,
          margin: { l: 50, r: 20, t: 20, b: 50 },
          xaxis: { ...(plotLayoutBase(props.prefersDark) as any).xaxis, title: "x" },
          yaxis: { ...(plotLayoutBase(props.prefersDark) as any).yaxis, title: "y" }
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
        ...plotLayoutBase(props.prefersDark),
        autosize: true,
        margin: { l: 50, r: 20, t: 20, b: 50 },
        xaxis: { ...(plotLayoutBase(props.prefersDark) as any).xaxis, title: "y (actual)" },
        yaxis: { ...(plotLayoutBase(props.prefersDark) as any).yaxis, title: "ŷ (predicted)" }
      }}
      style={{ width: "100%", height: "100%" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}

