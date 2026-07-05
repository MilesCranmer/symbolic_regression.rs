use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use dynamic_expressions::{StringTreeOptions, node_utils};
use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;

const D: usize = 3;
const EXCLUDED: [&str; 4] = ["target", "target_low", "target_high", "weight"];

struct Frame {
    headers: Vec<String>,
    rows: Vec<Vec<f32>>,
}

fn read_csv(path: &str) -> Frame {
    let text = fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
    let mut lines = text.lines();
    let headers: Vec<String> = lines
        .next()
        .unwrap_or_else(|| panic!("empty csv: {path}"))
        .split(',')
        .map(str::to_string)
        .collect();
    let rows = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|v| v.parse::<f32>().unwrap_or_else(|err| panic!("bad float {v}: {err}")))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Frame { headers, rows }
}

fn column_map(headers: &[String]) -> HashMap<&str, usize> {
    headers
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect()
}

fn feature_names(headers: &[String]) -> Vec<String> {
    headers
        .iter()
        .filter(|name| !EXCLUDED.contains(&name.as_str()))
        .cloned()
        .collect()
}

fn arrays(frame: &Frame, features: &[String]) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    let columns = column_map(&frame.headers);
    let n_rows = frame.rows.len();
    let mut x = Array2::<f32>::zeros((features.len(), n_rows));
    let mut y = Array1::<f32>::zeros(n_rows);
    let mut w = Array1::<f32>::zeros(n_rows);
    for (row_idx, row) in frame.rows.iter().enumerate() {
        for (feature_idx, feature) in features.iter().enumerate() {
            x[(feature_idx, row_idx)] = row[columns[feature.as_str()]];
        }
        y[row_idx] = row[columns["target"]];
        w[row_idx] = row[columns["weight"]];
    }
    (x, y, w)
}

fn mse(pred: &[f32], y: &Array1<f32>, valid_start: usize) -> f64 {
    let pred = &pred[valid_start..];
    let y = &y.as_slice().unwrap()[valid_start..];
    pred.iter()
        .zip(y.iter())
        .map(|(&p, &t)| {
            let err = (p - t) as f64;
            err * err
        })
        .sum::<f64>()
        / pred.len() as f64
}

fn r2(pred: &[f32], y: &Array1<f32>, valid_start: usize) -> f64 {
    let pred = &pred[valid_start..];
    let y = &y.as_slice().unwrap()[valid_start..];
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
    let ss_res = pred
        .iter()
        .zip(y.iter())
        .map(|(&p, &t)| {
            let err = (p - t) as f64;
            err * err
        })
        .sum::<f64>();
    let ss_tot = y
        .iter()
        .map(|&t| {
            let err = t as f64 - mean;
            err * err
        })
        .sum::<f64>();
    1.0 - ss_res / ss_tot
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut train_path = String::new();
    let mut test_path = String::new();
    let mut niterations = 8usize;
    let mut populations = 4usize;
    let mut population_size = 30usize;
    let mut ncycles_per_iteration = 380usize;
    let mut optimizer_iterations = 8usize;
    let mut maxsize = 18usize;
    let mut max_delay = 0usize;
    let mut delay_probability = 0.0f64;
    let mut parsimony = 0.0f64;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train_path = args.next().expect("--train requires a path"),
            "--test" => test_path = args.next().expect("--test requires a path"),
            "--niterations" => {
                niterations = args
                    .next()
                    .expect("--niterations requires a value")
                    .parse()
                    .expect("bad --niterations")
            }
            "--populations" => {
                populations = args
                    .next()
                    .expect("--populations requires a value")
                    .parse()
                    .expect("bad --populations")
            }
            "--population-size" => {
                population_size = args
                    .next()
                    .expect("--population-size requires a value")
                    .parse()
                    .expect("bad --population-size")
            }
            "--cycles" => {
                ncycles_per_iteration = args
                    .next()
                    .expect("--cycles requires a value")
                    .parse()
                    .expect("bad --cycles")
            }
            "--optimizer-iterations" => {
                optimizer_iterations = args
                    .next()
                    .expect("--optimizer-iterations requires a value")
                    .parse()
                    .expect("bad --optimizer-iterations")
            }
            "--maxsize" => {
                maxsize = args
                    .next()
                    .expect("--maxsize requires a value")
                    .parse()
                    .expect("bad --maxsize")
            }
            "--max-delay" => {
                max_delay = args
                    .next()
                    .expect("--max-delay requires a value")
                    .parse()
                    .expect("bad --max-delay")
            }
            "--delay-probability" => {
                delay_probability = args
                    .next()
                    .expect("--delay-probability requires a value")
                    .parse()
                    .expect("bad --delay-probability")
            }
            "--parsimony" => {
                parsimony = args
                    .next()
                    .expect("--parsimony requires a value")
                    .parse()
                    .expect("bad --parsimony")
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    assert!(!train_path.is_empty(), "--train is required");
    assert!(!test_path.is_empty(), "--test is required");

    let train = read_csv(&train_path);
    let test = read_csv(&test_path);
    let features = feature_names(&train.headers);
    let (x_train, y_train, weights) = arrays(&train, &features);
    let (x_test, y_test, _) = arrays(&test, &features);

    let dataset = Dataset::with_weights_and_names(x_train, y_train, Some(weights), features.clone());
    let operators = BuiltinOpsF32::from_names(["cos", "sin", "+", "sub", "*", "/"]).unwrap();
    let options = Options::<f32, D> {
        seed: 1009,
        niterations,
        populations,
        population_size,
        ncycles_per_iteration,
        optimizer_iterations,
        maxsize,
        max_delay,
        delay_probability,
        parsimony,
        maxdepth: 8,
        progress: false,
        deterministic: true,
        operators,
        ..Default::default()
    };

    let start = Instant::now();
    let result = equation_search::<f32, BuiltinOpsF32, D>(&dataset, &options);
    let elapsed = start.elapsed().as_secs_f64();

    let (pred, complete) =
        eval_tree_array::<f32, BuiltinOpsF32, D>(&result.best.expr, x_test.view(), &EvalOptions::default());
    let expression = string_tree(
        &result.best.expr,
        StringTreeOptions {
            variable_names: Some(&features),
            ..Default::default()
        },
    );
    let valid_start = node_utils::max_delay(&result.best.expr.nodes).min(y_test.len().saturating_sub(1));

    println!("{{");
    println!("  \"engine\": \"symbolic_regression_rs\",");
    println!("  \"status\": \"ok\",");
    println!("  \"wall_seconds\": {elapsed},");
    println!("  \"feature_count\": {},", features.len());
    println!("  \"expression\": {:?},", expression);
    println!("  \"train_loss\": {},", result.best.loss);
    println!("  \"valid_start\": {},", valid_start);
    println!("  \"test_mse\": {},", mse(&pred, &y_test, valid_start));
    println!("  \"test_r2\": {},", r2(&pred, &y_test, valid_start));
    println!("  \"prediction_complete\": {}", complete);
    println!("}}");
}
