import React from "react";
import type { SearchSnapshot } from "../../../types/srTypes";
import { formatSci } from "./plotUtils";

export function ControlsCard(props: {
  canInit: boolean;
  status: string;
  error: string | null;
  snap: SearchSnapshot | null;
  evalsPerSecond: number | null;
  niterations: number | null;
  setNiterations: (n: number) => void;
  canEditNiterations: boolean;

  initSearch: () => void;
  start: () => void;
  pause: () => void;
  reset: () => void;
}): React.ReactElement {
  return (
    <div className="card">
      <div className="cardTitle">Controls</div>
      <div className="controlsBar">
        <div className="buttonGroup">
          <button onClick={props.initSearch} disabled={!props.canInit} data-testid="search-init">
            Initialize
          </button>
          <button onClick={props.start} disabled={props.status !== "ready" && props.status !== "paused"} data-testid="search-start">
            Start / Resume
          </button>
          <button onClick={props.pause} disabled={props.status !== "running"}>
            Pause
          </button>
          <button onClick={props.reset}>Reset</button>
        </div>

        <label className="toolbarField">
          <span className="label">iterations</span>
          <input
            type="number"
            min={1}
            step={1}
            value={props.niterations ?? 1}
            className="itersInput"
            data-testid="opt-niterations"
            disabled={!props.canEditNiterations || props.niterations == null}
            onChange={(e) => props.setNiterations(Number(e.target.value))}
          />
        </label>

        <div className="spacer" />

        <div className="statusLine">
          <span className="statusChip" data-testid="search-status">
            {props.status}
          </span>
          {props.error && <span className="errorText">{props.error}</span>}
          {props.snap && (
            <span className="muted">
              cycles {props.snap.cycles_completed}/{props.snap.total_cycles} (
              {props.snap.total_cycles > 0 ? ((100 * props.snap.cycles_completed) / props.snap.total_cycles).toFixed(1) : "0"}%), evals=
              {props.snap.total_evals}
              {props.evalsPerSecond != null ? `, ${formatSci(props.evalsPerSecond)} eval/s` : ""}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
