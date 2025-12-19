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
              <th className="num" style={{ width: 120 }}>
                loss
              </th>
              <th>equation</th>
            </tr>
          </thead>
          <tbody>
            {props.front
              .slice()
              .reverse()
              .map((m) => (
                <tr
                  key={m.id}
                  className={m.id === props.selectedId ? "selectedRow" : ""}
                  onClick={() => props.selectEquation({ id: m.id, complexity: m.complexity })}
                  data-testid={`solution-row-${m.id}`}
                >
                  <td className="mono num">{formatSci(m.loss)}</td>
                  <td className="mono equationCell" title={m.equation}>
                    {m.equation}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
