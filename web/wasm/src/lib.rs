use std::io::Cursor;

use csv::ReaderBuilder;
use dynamic_expressions::operators::presets::BuiltinOpsF64;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use symbolic_regression::{Dataset, Operators, Options, SearchEngine};
use wasm_bindgen::prelude::*;

#[cfg(feature = "panic-hook")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WasmOptions {
    pub seed: u64,
    pub niterations: usize,
    pub populations: usize,
    pub population_size: usize,
    pub ncycles_per_iteration: usize,
    pub maxsize: usize,
    pub topn: usize,
    pub has_headers: bool,
}

impl Default for WasmOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            niterations: 100,
            populations: 8,
            population_size: 64,
            ncycles_per_iteration: 200,
            maxsize: 30,
            topn: 12,
            has_headers: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquationSummary {
    pub complexity: usize,
    pub loss: f64,
    pub cost: f64,
    pub equation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchSnapshot {
    pub total_cycles: usize,
    pub cycles_completed: usize,
    pub total_evals: u64,
    pub best: EquationSummary,
    pub pareto_front: Vec<EquationSummary>,
}

#[wasm_bindgen]
pub struct WasmSearch {
    engine: SearchEngine<f64, BuiltinOpsF64, 2>,
    pareto_k: usize,
}

#[wasm_bindgen]
impl WasmSearch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        csv_text: String,
        opts: JsValue,
        operator_tokens: JsValue,
    ) -> Result<WasmSearch, JsValue> {
        let opts: WasmOptions = from_value(opts)
            .or_else(|_| Ok::<WasmOptions, serde_wasm_bindgen::Error>(WasmOptions::default()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let tokens: Vec<String> =
            from_value(operator_tokens).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

        let dataset =
            parse_csv_to_dataset(&csv_text, opts.has_headers).map_err(|e| JsValue::from_str(&e))?;

        let operators = Operators::<2>::from_names::<BuiltinOpsF64>(&token_refs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let options = Options::<f64, 2> {
            seed: opts.seed,
            niterations: opts.niterations,
            populations: opts.populations,
            population_size: opts.population_size,
            ncycles_per_iteration: opts.ncycles_per_iteration,
            maxsize: opts.maxsize,
            topn: opts.topn,
            operators,
            // Keep browser output clean (and default-features disables progress anyway).
            progress: false,
            ..Default::default()
        };

        let engine = SearchEngine::<f64, BuiltinOpsF64, 2>::new(dataset, options);

        Ok(WasmSearch {
            engine,
            pareto_k: 25,
        })
    }

    pub fn set_pareto_k(&mut self, k: usize) {
        self.pareto_k = k;
    }

    pub fn is_finished(&self) -> bool {
        self.engine.is_finished()
    }

    pub fn step(&mut self, n_cycles: usize) -> Result<JsValue, JsValue> {
        let _ = self.engine.step(n_cycles);
        let snap = snapshot(&self.engine, self.pareto_k);
        to_value(&snap).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn best_equations(&self, k: usize) -> Result<JsValue, JsValue> {
        let mut pareto = self.engine.hall_of_fame().pareto_front();
        if pareto.len() > k {
            pareto.drain(0..(pareto.len() - k));
        }
        let out: Vec<EquationSummary> = pareto.into_iter().map(summary_from_member).collect();
        to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

fn snapshot(engine: &SearchEngine<f64, BuiltinOpsF64, 2>, pareto_k: usize) -> SearchSnapshot {
    let best = summary_from_member(engine.best().clone());
    let mut pareto = engine.hall_of_fame().pareto_front();
    if pareto.len() > pareto_k {
        pareto.drain(0..(pareto.len() - pareto_k));
    }
    let pareto_front: Vec<EquationSummary> = pareto.into_iter().map(summary_from_member).collect();

    SearchSnapshot {
        total_cycles: engine.total_cycles(),
        cycles_completed: engine.cycles_completed(),
        total_evals: engine.total_evals(),
        best,
        pareto_front,
    }
}

fn summary_from_member(
    m: symbolic_regression::PopMember<f64, BuiltinOpsF64, 2>,
) -> EquationSummary {
    EquationSummary {
        complexity: m.complexity,
        loss: m.loss,
        cost: m.cost,
        equation: m.expr.to_string(),
    }
}

fn parse_csv_to_dataset(csv_text: &str, has_headers: bool) -> Result<Dataset<f64>, String> {
    let mut rdr = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .has_headers(has_headers)
        .from_reader(Cursor::new(csv_text.as_bytes()));

    let headers: Vec<String> = if has_headers {
        rdr.headers()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let mut row: Vec<f64> = Vec::with_capacity(rec.len());
        for s in rec.iter() {
            let v = s
                .parse::<f64>()
                .map_err(|e| format!("failed to parse {s:?} as f64: {e}"))?;
            row.push(v);
        }
        rows.push(row);
    }

    let n_rows = rows.len();
    if n_rows == 0 {
        return Err("CSV had no data rows".to_string());
    }
    let n_cols = rows[0].len();
    if n_cols < 2 {
        return Err("CSV must have at least 2 columns (features..., target)".to_string());
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(format!(
                "row {i} has {found} columns but expected {n_cols}",
                found = row.len()
            ));
        }
    }

    let n_features = n_cols - 1;
    let mut x = Array2::<f64>::zeros((n_rows, n_features));
    let mut y = Array1::<f64>::zeros(n_rows);

    for (i, row) in rows.into_iter().enumerate() {
        for j in 0..n_features {
            x[(i, j)] = row[j];
        }
        y[i] = row[n_features];
    }

    let mut dataset = Dataset::new(x, y);
    if has_headers && headers.len() == n_cols {
        dataset.variable_names = headers[..n_features].to_vec();
    }
    Ok(dataset)
}
