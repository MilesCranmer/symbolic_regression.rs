import React from "react";
import { useSessionStore } from "../../state/sessionStore";
import { SrWorkerClient } from "../../worker/srWorkerClient";
import { ControlsCard } from "./searchSolutions/ControlsCard";
import { ParetoPlotCard } from "./searchSolutions/ParetoPlotCard";
import { QuickStatsCard } from "./searchSolutions/QuickStatsCard";
import { SelectedFitCard } from "./searchSolutions/SelectedFitCard";
import { SolutionsTableCard } from "./searchSolutions/SolutionsTableCard";
import { usePrefersDark } from "./searchSolutions/plotUtils";
import { useSearchController } from "./searchSolutions/useSearchController";

export function SearchSolutions(): React.ReactElement {
  const prefersDark = usePrefersDark();
  const c = useSearchController(SrWorkerClient);
  const options = useSessionStore((s) => s.options);
  const setOptionsPatch = useSessionStore((s) => s.setOptionsPatch);

  return (
    <div className="pane">
      <ControlsCard
        canInit={c.canInit}
        status={c.runtime.status}
        error={c.runtime.error}
        snap={c.snap}
        evalsPerSecond={c.runtime.evalsPerSecond}
        niterations={options?.niterations ?? null}
        setNiterations={(n) => setOptionsPatch({ niterations: n })}
        canEditNiterations={c.runtime.status === "idle" || c.runtime.status === "error"}
        initSearch={c.initSearch}
        start={c.start}
        pause={c.pause}
        reset={c.reset}
      />

      <div className="resultsColumns">
        <SolutionsTableCard front={c.runtime.front} selectedId={c.runtime.selectedId} selectEquation={c.selectEquation} />
        <SelectedFitCard
          prefersDark={prefersDark}
          selectedSummary={c.selectedSummary}
          effectiveFitMode={c.effectiveFitMode}
          hasVal={Boolean(c.split && c.split.val.length > 0)}
          trainActual={c.trainActual}
          valActual={c.valActual}
          trainYhat={c.trainYhat}
          valYhat={c.valYhat}
          trainXY={c.trainXY}
          valXY={c.valXY}
        />
        <QuickStatsCard
          selectedSummary={c.selectedSummary}
          evalTrain={c.evalTrain}
          evalVal={c.evalVal}
          hasVal={Boolean(c.split && c.split.val.length > 0)}
        />
        <ParetoPlotCard
          prefersDark={prefersDark}
          points={c.points}
          selectedId={c.runtime.selectedId}
          best={c.snap?.best ?? null}
          selectEquation={c.selectEquation}
        />
      </div>
    </div>
  );
}
