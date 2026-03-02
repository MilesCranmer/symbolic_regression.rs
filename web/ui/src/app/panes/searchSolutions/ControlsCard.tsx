import React from "react";
import type { SearchSnapshot } from "../../../types/srTypes";
import { formatSci } from "./plotUtils";

function statusChipClass(status: string): string {
  const base = "statusChip";
  const variant = `statusChip--${status}`;
  return `${base} ${variant}`;
}

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
  const pct =
    props.snap && props.snap.total_cycles > 0
      ? (100 * props.snap.cycles_completed) / props.snap.total_cycles
      : 0;

  const isRunning = props.status === "running";
  const isDone = props.status === "done";
  const cardClass = isRunning ? "card cardRunning" : "card";

  return (
    <div className={cardClass}>
      <div className="cardTitle">Controls</div>
      <div className="controlsBar">
        <div className="buttonGroup">
          <button
            className="btnPrimary"
            onClick={props.initSearch}
            disabled={!props.canInit}
            data-testid="search-init"
          >
            Initialize
          </button>
          <button
            className="btnSuccess"
            onClick={props.start}
            disabled={props.status !== "ready" && props.status !== "paused"}
            data-testid="search-start"
          >
            Start / Resume
          </button>
          <button onClick={props.pause} disabled={props.status !== "running"}>
            Pause
          </button>
          <button className="btnDanger" onClick={props.reset}>
            Reset
          </button>
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
          <span className={statusChipClass(props.status)} data-testid="search-status">
            <span className="statusDot" />
            {props.status}
          </span>
          {props.error && <span className="errorText">{props.error}</span>}
        </div>
      </div>

      {props.snap && (
        <>
          <div className="progressOuter">
            <div
              className={`progressInner${isRunning ? " progressRunning" : ""}${isDone ? " progressDone" : ""}`}
              style={{ width: `${Math.min(100, pct)}%` }}
            />
          </div>

          <div className="statsBar">
            <div className="statItem">
              <span className="statLabel">Progress</span>
              <span className="statValue">
                {props.snap.cycles_completed} / {props.snap.total_cycles} ({pct.toFixed(1)}%)
              </span>
            </div>
            <div className="statItem">
              <span className="statLabel">Evaluations</span>
              <span className="statValue">{props.snap.total_evals.toLocaleString()}</span>
            </div>
            {props.evalsPerSecond != null && (
              <div className="statItem">
                <span className="statLabel">Throughput</span>
                <span className="statValue">{formatSci(props.evalsPerSecond)} eval/s</span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
