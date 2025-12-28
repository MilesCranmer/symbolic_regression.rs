use std::io::Cursor;

use csv::ReaderBuilder;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{EvalOptions, OpId, OperatorSet, eval_plan_array_into};
use fastrand::Rng;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use symbolic_regression::{Dataset, LossKind, Operators, Options, SearchEngine, WasmOptionsShim};
use wasm_bindgen::prelude::*;

type FullDatasetParts = (Dataset<f64>, Vec<String>, Option<Array1<f64>>);
type SelectedRows = (Array2<f64>, Array1<f64>, Option<Array1<f64>>);

#[cfg(feature = "panic-hook")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn init_thread_pool(num_threads: usize) -> Result<(), JsValue> {
    wasm_bindgen_futures::JsFuture::from(wasm_bindgen_rayon::init_thread_pool(num_threads))
        .await
        .map(|_| ())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmOpInfo {
    pub arity: u8,
    pub name: String,
    pub display: String,
    pub infix: Option<String>,
    pub commutative: bool,
    pub associative: bool,
    pub complexity: u16,
}

#[wasm_bindgen]
pub fn builtin_operator_registry() -> Result<JsValue, JsValue> {
    let mut ops = Vec::new();
    for arity in 1..=BuiltinOpsF64::MAX_ARITY {
        for &id in BuiltinOpsF64::ops_with_arity(arity) {
            let op = OpId { arity, id };
            ops.push(WasmOpInfo {
                arity,
                name: BuiltinOpsF64::name(op).to_string(),
                display: BuiltinOpsF64::display(op).to_string(),
                infix: BuiltinOpsF64::infix(op).map(|s| s.to_string()),
                commutative: BuiltinOpsF64::commutative(op),
                associative: BuiltinOpsF64::associative(op),
                complexity: BuiltinOpsF64::complexity(op),
            });
        }
    }
    to_value(&ops).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn default_search_options() -> Result<JsValue, JsValue> {
    let opts = WasmSearchOptions::default();
    to_value(&opts).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn default_core_options() -> Result<JsValue, JsValue> {
    to_value(&WasmOptionsShim::default()).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn shuffle<T>(rng: &mut Rng, values: &mut [T]) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.usize(0..=i);
        values.swap(i, j);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WasmSearchOptions {
    // Dataset / UI plumbing
    pub has_headers: bool,
    pub x_columns: Option<Vec<usize>>,
    pub y_column: Option<usize>,
    pub weights_column: Option<usize>,
    pub validation_fraction: f64,

    // Loss
    pub loss_kind: String,
    pub huber_delta: f64,
    pub lp_p: f64,
    pub quantile_tau: f64,
    pub epsilon_insensitive_eps: f64,

    #[serde(flatten)]
    pub core: WasmOptionsShim,
}

impl Default for WasmSearchOptions {
    fn default() -> Self {
        Self {
            has_headers: true,
            x_columns: None,
            y_column: None,
            weights_column: None,
            validation_fraction: 0.0,

            loss_kind: "mse".to_string(),
            huber_delta: 1.0,
            lp_p: 2.0,
            quantile_tau: 0.5,
            epsilon_insensitive_eps: 0.1,
            core: WasmOptionsShim::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquationPoint {
    pub id: String,
    pub complexity: usize,
    pub loss: f64,
    pub cost: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquationSummary {
    pub id: String,
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
    pub pareto_points: Vec<EquationPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmMetrics {
    pub n: usize,
    pub mse: f64,
    pub mae: f64,
    pub rmse: f64,
    pub r2: f64,
    pub corr: f64,
    pub min_abs_err: f64,
    pub max_abs_err: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmEvalResult {
    pub metrics: WasmMetrics,
    pub yhat: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmSplitIndices {
    pub train: Vec<usize>,
    pub val: Vec<usize>,
}

#[wasm_bindgen]
pub struct WasmSearch {
    engine: SearchEngine<f64, BuiltinOpsF64, 3>,
    val_dataset: Option<Dataset<f64>>,
    split: WasmSplitIndices,
    pareto_k: usize,
}

#[wasm_bindgen]
impl WasmSearch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        csv_text: String,
        opts: JsValue,
        unary_tokens: JsValue,
        binary_tokens: JsValue,
        ternary_tokens: JsValue,
    ) -> Result<WasmSearch, JsValue> {
        let opts: WasmSearchOptions = from_value(opts)
            .or_else(|_| Ok::<WasmSearchOptions, serde_wasm_bindgen::Error>(WasmSearchOptions::default()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let unary: Vec<String> = from_value(unary_tokens).unwrap_or_default();
        let binary: Vec<String> = from_value(binary_tokens).unwrap_or_default();
        let ternary: Vec<String> = from_value(ternary_tokens).unwrap_or_default();
        let unary_refs: Vec<&str> = unary.iter().map(|s| s.as_str()).collect();
        let binary_refs: Vec<&str> = binary.iter().map(|s| s.as_str()).collect();
        let ternary_refs: Vec<&str> = ternary.iter().map(|s| s.as_str()).collect();

        let operators = Operators::<3>::from_names_by_arity::<BuiltinOpsF64>(&unary_refs, &binary_refs, &ternary_refs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let (headers, rows) = parse_csv_to_rows(&csv_text, opts.has_headers).map_err(|e| JsValue::from_str(&e))?;

        let (dataset_all, variable_names, weights_all) =
            build_full_dataset(&headers, &rows, &opts).map_err(|e| JsValue::from_str(&e))?;

        let split = make_split_indices(dataset_all.n_rows, opts.validation_fraction, opts.core.seed);
        let (train, val) = split_train_val(&dataset_all, weights_all.as_ref(), &variable_names, &split)
            .map_err(|e| JsValue::from_str(&e))?;

        let options = options_from_wasm(&opts, operators)?;

        let engine = SearchEngine::<f64, BuiltinOpsF64, 3>::new(train, options);

        Ok(WasmSearch {
            engine,
            val_dataset: val,
            split,
            pareto_k: 250,
        })
    }

    pub fn set_pareto_k(&mut self, k: usize) {
        self.pareto_k = k;
    }

    pub fn get_split_indices(&self) -> Result<JsValue, JsValue> {
        to_value(&self.split).map_err(|e| JsValue::from_str(&e.to_string()))
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
        let out: Vec<EquationSummary> = pareto
            .into_iter()
            .map(|m| summary_from_member(&self.engine, &m))
            .collect();
        to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn evaluate_member(&self, member_id: String, which: String) -> Result<JsValue, JsValue> {
        let member_id: u64 = member_id
            .parse()
            .map_err(|_| JsValue::from_str("member_id must be a base-10 integer string"))?;
        let dataset = match which.as_str() {
            "train" => self.engine.dataset(),
            "val" => self
                .val_dataset
                .as_ref()
                .ok_or_else(|| JsValue::from_str("no validation dataset"))?,
            _ => return Err(JsValue::from_str("which must be \"train\" or \"val\"")),
        };

        let m = find_member_by_id(&self.engine, member_id).ok_or_else(|| JsValue::from_str("member not found"))?;

        let yhat = eval_member_on_dataset(dataset, &m)?;
        let metrics = compute_metrics(&yhat, dataset);
        let out = WasmEvalResult { metrics, yhat };
        to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

fn options_from_wasm(opts: &WasmSearchOptions, operators: Operators<3>) -> Result<Options<f64, 3>, JsValue> {
    let loss_kind = match opts.loss_kind.trim().to_ascii_lowercase().as_str() {
        "mse" => LossKind::Mse,
        "mae" => LossKind::Mae,
        "rmse" => LossKind::Rmse,
        "huber" => LossKind::Huber {
            delta: opts.huber_delta,
        },
        "logcosh" => LossKind::LogCosh,
        "lp" => LossKind::Lp { p: opts.lp_p },
        "quantile" => LossKind::Quantile { tau: opts.quantile_tau },
        "epsilon-insensitive" | "epsilon_insensitive" | "eps-insensitive" | "eps_insensitive" => {
            LossKind::EpsilonInsensitive {
                eps: opts.epsilon_insensitive_eps,
            }
        }
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown loss_kind {other:?} (expected \"mse\", \"mae\", \"rmse\", \"huber\", \"logcosh\", \"lp\", \"quantile\", or \"epsilon-insensitive\")"
            )));
        }
    };

    let mut out = Options::<f64, 3> {
        operators,
        ..Default::default()
    };

    opts.core.apply_to(&mut out);
    // Keep browser output clean (and default-features disables progress anyway).
    out.progress = false;
    out.loss = symbolic_regression::make_loss::<f64>(loss_kind);

    Ok(out)
}

fn snapshot(engine: &SearchEngine<f64, BuiltinOpsF64, 3>, pareto_k: usize) -> SearchSnapshot {
    let best = summary_from_member(engine, engine.best());

    let mut pareto = engine.hall_of_fame().pareto_front();
    if pareto.len() > pareto_k {
        pareto.drain(0..(pareto.len() - pareto_k));
    }
    let pareto_points: Vec<EquationPoint> = pareto
        .into_iter()
        .map(|m| EquationPoint {
            id: m.id.0.to_string(),
            complexity: m.complexity,
            loss: m.loss,
            cost: m.cost,
        })
        .collect();

    SearchSnapshot {
        total_cycles: engine.total_cycles(),
        cycles_completed: engine.cycles_completed(),
        total_evals: engine.total_evals(),
        best,
        pareto_points,
    }
}

fn summary_from_member(
    engine: &SearchEngine<f64, BuiltinOpsF64, 3>,
    m: &symbolic_regression::PopMember<f64, BuiltinOpsF64, 3>,
) -> EquationSummary {
    let equation = string_tree::<f64, BuiltinOpsF64, 3>(
        &m.expr,
        StringTreeOptions {
            variable_names: Some(&engine.dataset().variable_names),
            pretty: false,
        },
    );
    EquationSummary {
        id: m.id.0.to_string(),
        complexity: m.complexity,
        loss: m.loss,
        cost: m.cost,
        equation,
    }
}

fn find_member_by_id(
    engine: &SearchEngine<f64, BuiltinOpsF64, 3>,
    member_id: u64,
) -> Option<symbolic_regression::PopMember<f64, BuiltinOpsF64, 3>> {
    engine
        .hall_of_fame()
        .members()
        .find(|m| m.id.0 == member_id)
        .cloned()
        .or_else(|| {
            let b = engine.best();
            (b.id.0 == member_id).then(|| b.clone())
        })
}

fn eval_member_on_dataset(
    dataset: &Dataset<f64>,
    m: &symbolic_regression::PopMember<f64, BuiltinOpsF64, 3>,
) -> Result<Vec<f64>, JsValue> {
    let mut yhat = vec![0.0_f64; dataset.n_rows];
    let mut scratch = ndarray::Array2::<f64>::zeros((0, 0));
    let eval_opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let ok = eval_plan_array_into::<f64, BuiltinOpsF64, 3>(
        &mut yhat,
        &m.plan,
        &m.expr,
        dataset.x.view(),
        &mut scratch,
        &eval_opts,
    );
    if !ok {
        return Err(JsValue::from_str("evaluation failed"));
    }
    Ok(yhat)
}

fn compute_metrics(yhat: &[f64], dataset: &Dataset<f64>) -> WasmMetrics {
    let y = dataset.y.as_slice().unwrap_or(&[]);
    let w = dataset.weights.as_ref().and_then(|w| w.as_slice());
    let n = y.len();

    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut sum_w = 0.0;
    let mut min_abs = f64::INFINITY;
    let mut max_abs = 0.0;

    match w {
        None => {
            for (&yh, &yi) in yhat.iter().zip_eq(y) {
                let r = yh - yi;
                let ar = r.abs();
                mse += r * r;
                mae += ar;
                if ar < min_abs {
                    min_abs = ar;
                }
                if ar > max_abs {
                    max_abs = ar;
                }
            }
            if n > 0 {
                mse /= n as f64;
                mae /= n as f64;
            }
        }
        Some(w) => {
            for ((&yh, &yi), &wi) in yhat.iter().zip_eq(y).zip_eq(w) {
                let r = yh - yi;
                let ar = r.abs();
                sum_w += wi;
                mse += wi * r * r;
                mae += wi * ar;
                if ar < min_abs {
                    min_abs = ar;
                }
                if ar > max_abs {
                    max_abs = ar;
                }
            }
            if sum_w > 0.0 {
                mse /= sum_w;
                mae /= sum_w;
            } else {
                mse = 0.0;
                mae = 0.0;
            }
        }
    }

    let rmse = mse.sqrt();
    let (r2, corr) = r2_and_corr(yhat, y, w);

    WasmMetrics {
        n,
        mse,
        mae,
        rmse,
        r2,
        corr,
        min_abs_err: if min_abs.is_finite() { min_abs } else { 0.0 },
        max_abs_err: max_abs,
    }
}

fn r2_and_corr(yhat: &[f64], y: &[f64], w: Option<&[f64]>) -> (f64, f64) {
    if y.is_empty() || yhat.len() != y.len() {
        return (f64::NAN, f64::NAN);
    }

    // Weighted mean for R^2; correlation unweighted (for UI purposes).
    let y_mean = match w {
        None => y.iter().sum::<f64>() / (y.len() as f64),
        Some(w) => {
            let sum_w = w.iter().sum::<f64>();
            if sum_w == 0.0 {
                0.0
            } else {
                y.iter().zip_eq(w).map(|(&yi, &wi)| yi * wi).sum::<f64>() / sum_w
            }
        }
    };

    let mut sse = 0.0;
    let mut sst = 0.0;
    match w {
        None => {
            for (&yh, &yi) in yhat.iter().zip_eq(y) {
                let r = yh - yi;
                sse += r * r;
                let d = yi - y_mean;
                sst += d * d;
            }
        }
        Some(w) => {
            for ((&yh, &yi), &wi) in yhat.iter().zip_eq(y).zip_eq(w) {
                let r = yh - yi;
                sse += wi * r * r;
                let d = yi - y_mean;
                sst += wi * d * d;
            }
        }
    }
    let r2 = if sst == 0.0 { 0.0 } else { 1.0 - (sse / sst) };

    // Unweighted Pearson correlation.
    let yh_mean = yhat.iter().sum::<f64>() / (yhat.len() as f64);
    let mut cov = 0.0;
    let mut vy = 0.0;
    let mut vyh = 0.0;
    for (&yh, &yi) in yhat.iter().zip_eq(y) {
        let dyh = yh - yh_mean;
        let dy = yi - y_mean;
        cov += dyh * dy;
        vyh += dyh * dyh;
        vy += dy * dy;
    }
    let corr = if vy == 0.0 || vyh == 0.0 {
        0.0
    } else {
        cov / (vy.sqrt() * vyh.sqrt())
    };

    (r2, corr)
}

fn parse_csv_to_rows(csv_text: &str, has_headers: bool) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
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
    Ok((headers, rows))
}

fn build_full_dataset(
    headers: &[String],
    rows: &[Vec<f64>],
    opts: &WasmSearchOptions,
) -> Result<FullDatasetParts, String> {
    if rows.is_empty() {
        return Err("CSV had no data rows".to_string());
    }
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    if n_cols < 2 {
        return Err("CSV must have at least 2 columns".to_string());
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(format!(
                "row {i} has {found} columns but expected {n_cols}",
                found = row.len()
            ));
        }
    }

    let y_col = opts.y_column.unwrap_or(n_cols - 1);
    if y_col >= n_cols {
        return Err(format!("y_column out of range: {y_col} (n_cols={n_cols})"));
    }

    let w_col = opts.weights_column;
    if let Some(wc) = w_col {
        if wc >= n_cols {
            return Err(format!("weights_column out of range: {wc} (n_cols={n_cols})"));
        }
        if wc == y_col {
            return Err("weights_column must be different from y_column".to_string());
        }
    }

    let mut x_cols = match &opts.x_columns {
        Some(v) => v.clone(),
        None => (0..n_cols).filter(|&c| c != y_col && Some(c) != w_col).collect(),
    };
    x_cols.sort_unstable();
    x_cols.dedup();
    if x_cols.is_empty() {
        return Err("x_columns is empty".to_string());
    }
    if x_cols.iter().any(|&c| c >= n_cols) {
        return Err("x_columns contains an out-of-range index".to_string());
    }
    if x_cols.contains(&y_col) {
        return Err("x_columns must not include y_column".to_string());
    }
    if let Some(wc) = w_col {
        if x_cols.contains(&wc) {
            return Err("x_columns must not include weights_column".to_string());
        }
    }

    let n_features = x_cols.len();

    let mut x = Array2::<f64>::zeros((n_features, n_rows));
    let mut y = Array1::<f64>::zeros(n_rows);
    let mut w = w_col.map(|_| Array1::<f64>::zeros(n_rows));

    for (i, row) in rows.iter().enumerate() {
        for (j, &c) in x_cols.iter().enumerate() {
            x[(j, i)] = row[c];
        }
        y[i] = row[y_col];
        if let (Some(wc), Some(w_arr)) = (w_col, w.as_mut()) {
            w_arr[i] = row[wc];
        }
    }

    let variable_names: Vec<String> = if opts.has_headers && headers.len() == n_cols {
        x_cols.iter().map(|&c| headers[c].clone()).collect()
    } else {
        (0..n_features).map(|i| format!("x{}", i)).collect()
    };

    let dataset = Dataset::with_weights_and_names(x, y, None, variable_names.clone());
    Ok((dataset, variable_names, w))
}

fn make_split_indices(n_rows: usize, validation_fraction: f64, seed: u64) -> WasmSplitIndices {
    if n_rows == 0 {
        return WasmSplitIndices {
            train: Vec::new(),
            val: Vec::new(),
        };
    }

    let vf = validation_fraction.clamp(0.0, 0.9);
    let n_val = ((vf * (n_rows as f64)).round() as usize).min(n_rows);
    if n_val == 0 {
        return WasmSplitIndices {
            train: (0..n_rows).collect(),
            val: Vec::new(),
        };
    }

    let mut idx: Vec<usize> = (0..n_rows).collect();
    let mut rng = Rng::with_seed(seed ^ 0x6a09_e667_f3bc_c909);
    shuffle(&mut rng, &mut idx);

    let val = idx[..n_val].to_vec();
    let train = idx[n_val..].to_vec();

    WasmSplitIndices { train, val }
}

fn split_train_val(
    dataset_all: &Dataset<f64>,
    weights_all: Option<&Array1<f64>>,
    variable_names: &[String],
    split: &WasmSplitIndices,
) -> Result<(Dataset<f64>, Option<Dataset<f64>>), String> {
    let (x_train, y_train, w_train) = select_rows(&dataset_all.x, &dataset_all.y, weights_all, &split.train)?;
    let train = Dataset::with_weights_and_names(x_train, y_train, w_train, variable_names.to_vec());

    if split.val.is_empty() {
        return Ok((train, None));
    }
    let (x_val, y_val, w_val) = select_rows(&dataset_all.x, &dataset_all.y, weights_all, &split.val)?;
    let val = Dataset::with_weights_and_names(x_val, y_val, w_val, variable_names.to_vec());
    Ok((train, Some(val)))
}

fn select_rows(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: Option<&Array1<f64>>,
    indices: &[usize],
) -> Result<SelectedRows, String> {
    let (n_features, n_rows) = x.dim();
    for &i in indices {
        if i >= n_rows {
            return Err(format!("index out of range while splitting: {i} (n_rows={n_rows})"));
        }
    }

    let mut xo = Array2::<f64>::zeros((n_features, indices.len()));
    let mut yo = Array1::<f64>::zeros(indices.len());
    let mut wo = w.map(|_| Array1::<f64>::zeros(indices.len()));

    for (i_new, &i_old) in indices.iter().enumerate() {
        for j in 0..n_features {
            xo[(j, i_new)] = x[(j, i_old)];
        }
        yo[i_new] = y[i_old];
        if let (Some(w_in), Some(w_out)) = (w, wo.as_mut()) {
            w_out[i_new] = w_in[i_old];
        }
    }
    Ok((xo, yo, wo))
}

#[cfg(test)]
mod tests {
    use super::WasmSearchOptions;

    #[test]
    fn wasm_options_shim_defaults_match_known_spec_values() {
        let core = WasmSearchOptions::default().core;
        assert_eq!(core.seed, 0);
        assert_eq!(core.niterations, 100);
        assert_eq!(core.populations, 31);
        assert_eq!(core.population_size, 27);
        assert_eq!(core.maxsize, 30);
        assert_eq!(core.maxdepth, 30);
        assert!(core.progress);
        assert!(!core.batching);
    }

    #[test]
    fn wasm_search_options_flattened_serde_shape_has_no_core_field() {
        let opts = WasmSearchOptions::default();
        let v = serde_json::to_value(&opts).unwrap();
        let obj = v.as_object().unwrap();
        assert!(obj.contains_key("seed"));
        assert!(obj.contains_key("mutation_weights"));
        assert!(!obj.contains_key("core"));
    }

    #[test]
    fn wasm_options_shim_apply_to_patches_core_options() {
        let mut core = WasmSearchOptions::default().core;
        core.seed = 123;
        core.niterations = 9;
        core.progress = false;

        let mut opt: symbolic_regression::Options<f64, 3> = Default::default();
        core.apply_to(&mut opt);

        assert_eq!(opt.seed, 123);
        assert_eq!(opt.niterations, 9);
        assert!(!opt.progress);
    }
}
