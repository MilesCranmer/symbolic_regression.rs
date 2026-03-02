import React from "react";
import { useEffect } from "react";
import { useSessionStore } from "../state/sessionStore";
import { EnterData } from "./panes/EnterData";
import { ModelingTask } from "./panes/ModelingTask";
import { SearchSolutions } from "./panes/SearchSolutions";

const tabs = [
  { key: "data" as const, label: "Data", num: 1 },
  { key: "configure" as const, label: "Configure", num: 2 },
  { key: "run" as const, label: "Run", num: 3 }
];

export function App(): React.ReactElement {
  const tab = useSessionStore((s) => s.tab);
  const setTab = useSessionStore((s) => s.setTab);
  const loadWasmMetadata = useSessionStore((s) => s.loadWasmMetadata);

  useEffect(() => {
    void loadWasmMetadata();
  }, [loadWasmMetadata]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="title">PySR Online</div>
        <div className="tabs">
          {tabs.map((t, i) => (
            <React.Fragment key={t.key}>
              {i > 0 && (
                <span className="tabSep" aria-hidden="true">
                  ›
                </span>
              )}
              <button className={tab === t.key ? "tab active" : "tab"} onClick={() => setTab(t.key)}>
                <span className="tabNumber">{t.num}</span>
                {t.label}
              </button>
            </React.Fragment>
          ))}
        </div>
        <nav className="topbarLinks" aria-label="Project links">
          <a
            className="topbarLink"
            href="https://github.com/astroautomata/symbolic_regression.rs"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <span className="topbarLinkSep" aria-hidden="true">
            ·
          </span>
          <a className="topbarLink" href="https://arxiv.org/abs/2305.01582" target="_blank" rel="noreferrer">
            arXiv
          </a>
        </nav>
      </header>

      <main className="main">
        <div className="mainInner">
          {tab === "data" && <EnterData />}
          {tab === "configure" && <ModelingTask />}
          {tab === "run" && <SearchSolutions />}
        </div>
      </main>
    </div>
  );
}
