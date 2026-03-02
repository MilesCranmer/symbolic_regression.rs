import React from "react";
import type { EquationSummary } from "../../../types/srTypes";
import { formatSci } from "./plotUtils";

export function SolutionsTableCard(props: {
  front: EquationSummary[];
  selectedId: string | null;
  selectEquation: (sel: { id: string; complexity: number }) => void;
}): React.ReactElement {
  return (
    <div className="card gridCell resultsCard resultsCard--table resultsFixed">
      <div className="cardTitle">Current solutions</div>
      <div className="tableWrap">
        <table className="table fixed" data-testid="solutions-table">
          <thead>
            <tr>
              <th style={{ width: 50 }}>complexity</th>
              <th className="num" style={{ width: 100 }}>
                loss
              </th>
              <th>equation</th>
            </tr>
          </thead>
          <tbody>
            {props.front.length === 0 ? (
              <tr>
                <td colSpan={3} className="muted" style={{ textAlign: "center", padding: 24 }}>
                  No solutions yet. Initialize and start a search.
                </td>
              </tr>
            ) : (
              props.front
                .slice()
                .reverse()
                .map((m) => (
                  <tr
                    key={m.id}
                    className={`solutionRow${m.id === props.selectedId ? " selectedRow" : ""}`}
                    onClick={() => props.selectEquation({ id: m.id, complexity: m.complexity })}
                    data-testid={`solution-row-${m.id}`}
                  >
                    <td>
                      <span className="complexityBadge">{m.complexity}</span>
                    </td>
                    <td className="mono num">{formatSci(m.loss)}</td>
                    <td className="mono equationCell" title={m.equation}>
                      {m.equation}
                    </td>
                  </tr>
                ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
