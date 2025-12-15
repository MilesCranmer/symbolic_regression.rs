import React, { useMemo, useState } from "react";
import { useSessionStore } from "../../state/sessionStore";
import type { WasmOpInfo } from "../../types/srTypes";

function groupByArity(ops: WasmOpInfo[]): Record<number, WasmOpInfo[]> {
  const out: Record<number, WasmOpInfo[]> = {};
  for (const op of ops) {
    (out[op.arity] ??= []).push(op);
  }
  for (const k of Object.keys(out)) out[Number(k)].sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

export function ModelingTask(): React.ReactElement {
  const registry = useSessionStore((s) => s.operatorRegistry);
  const options = useSessionStore((s) => s.options);
  const setOptionsPatch = useSessionStore((s) => s.setOptionsPatch);

  const unaryOps = useSessionStore((s) => s.unaryOps);
  const binaryOps = useSessionStore((s) => s.binaryOps);
  const ternaryOps = useSessionStore((s) => s.ternaryOps);
  const toggleOp = useSessionStore((s) => s.toggleOp);
  const applyPreset = useSessionStore((s) => s.applyPreset);

  const [filter, setFilter] = useState("");
  const grouped = useMemo(() => groupByArity(registry), [registry]);
  const filterLower = filter.trim().toLowerCase();

  const visible = (ops: WasmOpInfo[]) =>
    filterLower.length === 0
      ? ops
      : ops.filter((o) => `${o.name} ${o.display} ${o.infix ?? ""}`.toLowerCase().includes(filterLower));

  const selectAll = (arity: 1 | 2 | 3) => {
    const ops = visible(grouped[arity] ?? []);
    for (const op of ops) toggleOp(arity, op.name);
  };

  const setAll = (arity: 1 | 2 | 3, enabled: boolean) => {
    const ops = visible(grouped[arity] ?? []);
    const selected = new Set((arity === 1 ? unaryOps : arity === 2 ? binaryOps : ternaryOps) ?? []);
    for (const op of ops) {
      const has = selected.has(op.name);
      if (enabled && !has) toggleOp(arity, op.name);
      if (!enabled && has) toggleOp(arity, op.name);
    }
  };

  const selectEverything = () => {
    setAll(1, true);
    setAll(2, true);
    setAll(3, true);
  };

  if (!options) {
    return (
      <div className="pane">
        <div className="card">
          <div className="muted">Loading defaults…</div>
        </div>
      </div>
    );
  }

  return (
    <div className="pane">
      <div className="card">
        <div className="cardTitle">Operators</div>
        <div className="row">
          <input className="input" placeholder="Filter (sin, log, …)" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <button onClick={() => applyPreset("basic")}>Preset: Basic</button>
          <button onClick={() => applyPreset("trig")}>Preset: Trig</button>
          <button onClick={() => applyPreset("explog")}>Preset: Exp/Log</button>
          <button onClick={selectEverything}>Select all</button>
        </div>

        <div className="opGroups">
          {[1, 2, 3].map((arity) => {
            const ops = visible(grouped[arity] ?? []);
            const selected = arity === 1 ? unaryOps : arity === 2 ? binaryOps : ternaryOps;
            return (
              <div key={arity} className="opGroup">
                <div className="subTitle">
                  {arity === 1 ? "Unary" : arity === 2 ? "Binary" : "Ternary"} ({selected.length} selected)
                </div>
                <div className="row">
                  <button onClick={() => setAll(arity as 1 | 2 | 3, true)}>Select visible</button>
                  <button onClick={() => setAll(arity as 1 | 2 | 3, false)}>Clear visible</button>
                </div>
                <div className="checkboxGrid">
                  {ops.map((op) => {
                    const checked = selected.includes(op.name);
                    const label = op.display === op.name ? op.name : `${op.name} (${op.display})`;
                    return (
                      <label key={`${op.arity}:${op.name}`} className="checkbox">
                        <input type="checkbox" checked={checked} onChange={() => toggleOp(arity as 1 | 2 | 3, op.name)} />
                        <span className="mono">{label}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <div className="cardTitle">Loss</div>
        <div className="row">
          <label className="field">
            <div className="label">loss_kind</div>
            <select value={options.loss_kind} onChange={(e) => setOptionsPatch({ loss_kind: e.target.value })}>
              <option value="mse">MSE</option>
              <option value="mae">MAE</option>
              <option value="rmse">RMSE</option>
              <option value="huber">Huber</option>
            </select>
          </label>
          {options.loss_kind === "huber" && (
            <label className="field">
              <div className="label">huber_delta</div>
              <input type="number" min={1e-6} step={0.1} value={options.huber_delta} onChange={(e) => setOptionsPatch({ huber_delta: Number(e.target.value) })} />
            </label>
          )}
        </div>
      </div>

      <div className="card">
        <div className="cardTitle">Hyperparameters</div>
        <div className="muted">All fields map 1:1 to `symbolic_regression::Options` (plus a few web-only ones).</div>

        <div className="grid2">
          <section className="section">
            <div className="subTitle">Search Budget</div>
            <label className="field">
              <div className="label">niterations</div>
              <input
                type="number"
                min={1}
                value={options.niterations}
                onChange={(e) => setOptionsPatch({ niterations: Number(e.target.value) })}
                data-testid="opt-niterations"
              />
            </label>
            <label className="field">
              <div className="label">populations</div>
              <input
                type="number"
                min={1}
                value={options.populations}
                onChange={(e) => setOptionsPatch({ populations: Number(e.target.value) })}
                data-testid="opt-populations"
              />
            </label>
            <label className="field">
              <div className="label">population_size</div>
              <input
                type="number"
                min={1}
                value={options.population_size}
                onChange={(e) => setOptionsPatch({ population_size: Number(e.target.value) })}
                data-testid="opt-population-size"
              />
            </label>
            <label className="field">
              <div className="label">ncycles_per_iteration</div>
              <input
                type="number"
                min={1}
                value={options.ncycles_per_iteration}
                onChange={(e) => setOptionsPatch({ ncycles_per_iteration: Number(e.target.value) })}
                data-testid="opt-ncycles"
              />
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Constraints</div>
            <label className="field">
              <div className="label">maxsize</div>
              <input type="number" min={1} value={options.maxsize} onChange={(e) => setOptionsPatch({ maxsize: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">maxdepth</div>
              <input type="number" min={1} value={options.maxdepth} onChange={(e) => setOptionsPatch({ maxdepth: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">warmup_maxsize_by</div>
              <input type="number" step={0.05} value={options.warmup_maxsize_by} onChange={(e) => setOptionsPatch({ warmup_maxsize_by: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">topn</div>
              <input type="number" min={1} value={options.topn} onChange={(e) => setOptionsPatch({ topn: Number(e.target.value) })} />
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Selection</div>
            <label className="field">
              <div className="label">tournament_selection_n</div>
              <input type="number" min={1} value={options.tournament_selection_n} onChange={(e) => setOptionsPatch({ tournament_selection_n: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">tournament_selection_p</div>
              <input type="number" min={0} max={1} step={0.001} value={options.tournament_selection_p} onChange={(e) => setOptionsPatch({ tournament_selection_p: Number(e.target.value) })} />
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.use_frequency} onChange={(e) => setOptionsPatch({ use_frequency: e.target.checked })} />
              use_frequency
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.use_frequency_in_tournament} onChange={(e) => setOptionsPatch({ use_frequency_in_tournament: e.target.checked })} />
              use_frequency_in_tournament
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Parsimony / Scaling</div>
            <label className="field">
              <div className="label">parsimony</div>
              <input type="number" step={0.001} value={options.parsimony} onChange={(e) => setOptionsPatch({ parsimony: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">adaptive_parsimony_scaling</div>
              <input type="number" step={0.1} value={options.adaptive_parsimony_scaling} onChange={(e) => setOptionsPatch({ adaptive_parsimony_scaling: Number(e.target.value) })} />
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.use_baseline} onChange={(e) => setOptionsPatch({ use_baseline: e.target.checked })} />
              use_baseline
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.annealing} onChange={(e) => setOptionsPatch({ annealing: e.target.checked })} />
              annealing
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Mutation / Crossover</div>
            <label className="field">
              <div className="label">crossover_probability</div>
              <input type="number" step={0.001} value={options.crossover_probability} onChange={(e) => setOptionsPatch({ crossover_probability: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">perturbation_factor</div>
              <input type="number" step={0.001} value={options.perturbation_factor} onChange={(e) => setOptionsPatch({ perturbation_factor: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">probability_negate_constant</div>
              <input type="number" step={0.001} value={options.probability_negate_constant} onChange={(e) => setOptionsPatch({ probability_negate_constant: Number(e.target.value) })} />
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.skip_mutation_failures} onChange={(e) => setOptionsPatch({ skip_mutation_failures: e.target.checked })} />
              skip_mutation_failures
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.should_simplify} onChange={(e) => setOptionsPatch({ should_simplify: e.target.checked })} />
              should_simplify
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Optimization</div>
            <label className="checkbox">
              <input type="checkbox" checked={options.should_optimize_constants} onChange={(e) => setOptionsPatch({ should_optimize_constants: e.target.checked })} />
              should_optimize_constants
            </label>
            <label className="field">
              <div className="label">optimizer_probability</div>
              <input type="number" step={0.001} value={options.optimizer_probability} onChange={(e) => setOptionsPatch({ optimizer_probability: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">optimizer_iterations</div>
              <input type="number" min={0} value={options.optimizer_iterations} onChange={(e) => setOptionsPatch({ optimizer_iterations: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">optimizer_nrestarts</div>
              <input type="number" min={0} value={options.optimizer_nrestarts} onChange={(e) => setOptionsPatch({ optimizer_nrestarts: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">optimizer_f_calls_limit</div>
              <input type="number" min={0} value={options.optimizer_f_calls_limit} onChange={(e) => setOptionsPatch({ optimizer_f_calls_limit: Number(e.target.value) })} />
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Migration / Replacement</div>
            <label className="checkbox">
              <input type="checkbox" checked={options.migration} onChange={(e) => setOptionsPatch({ migration: e.target.checked })} />
              migration
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.hof_migration} onChange={(e) => setOptionsPatch({ hof_migration: e.target.checked })} />
              hof_migration
            </label>
            <label className="field">
              <div className="label">fraction_replaced</div>
              <input type="number" step={0.0001} value={options.fraction_replaced} onChange={(e) => setOptionsPatch({ fraction_replaced: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">fraction_replaced_hof</div>
              <input type="number" step={0.0001} value={options.fraction_replaced_hof} onChange={(e) => setOptionsPatch({ fraction_replaced_hof: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">fraction_replaced_guesses</div>
              <input type="number" step={0.0001} value={options.fraction_replaced_guesses} onChange={(e) => setOptionsPatch({ fraction_replaced_guesses: Number(e.target.value) })} />
            </label>
          </section>

          <section className="section">
            <div className="subTitle">Misc</div>
            <label className="field">
              <div className="label">seed</div>
              <input type="number" min={0} value={options.seed} onChange={(e) => setOptionsPatch({ seed: Number(e.target.value) })} />
            </label>
            <label className="field">
              <div className="label">alpha</div>
              <input type="number" step={0.001} value={options.alpha} onChange={(e) => setOptionsPatch({ alpha: Number(e.target.value) })} />
            </label>
            <label className="checkbox">
              <input type="checkbox" checked={options.progress} onChange={(e) => setOptionsPatch({ progress: e.target.checked })} />
              progress (ignored in browser)
            </label>
          </section>
        </div>

        <details className="details">
          <summary>Mutation weights (expert)</summary>
          <div className="grid2">
            {Object.entries(options.mutation_weights).map(([k, v]) => (
              <label key={k} className="field">
                <div className="label">{k}</div>
                <input
                  type="number"
                  step={0.0001}
                  value={v}
                  onChange={(e) =>
                    setOptionsPatch({
                      mutation_weights: { ...options.mutation_weights, [k]: Number(e.target.value) } as any
                    })
                  }
                />
              </label>
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}
