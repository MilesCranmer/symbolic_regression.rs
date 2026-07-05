use dynamic_expressions::StringTreeOptions;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;
use symbolic_regression::{Dataset, Options, equation_search};

#[test]
fn delay_leaf_evaluates_shifted_feature_without_expanded_columns() {
    let x = Array2::from_shape_vec((1, 5), vec![10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let expr = PostfixExpr::<f64, BuiltinOpsF64, 2>::new(
        vec![PNode::Delay { feature: 0, offset: 2 }],
        Vec::new(),
        Metadata {
            variable_names: vec!["x".into()],
        },
    );

    let (out, complete) = eval_tree_array::<f64, BuiltinOpsF64, 2>(&expr, x.view(), &EvalOptions::default());

    assert!(complete);
    assert_eq!(out, vec![10.0, 10.0, 10.0, 20.0, 30.0]);
    assert_eq!(
        string_tree(
            &expr,
            StringTreeOptions {
                variable_names: Some(&["x".to_string()]),
                ..Default::default()
            },
        ),
        "delay(x, 2)"
    );
}

#[test]
fn delay_validity_mask_reports_warmup_and_sequence_boundaries() {
    let expr = PostfixExpr::<f64, BuiltinOpsF64, 2>::new(
        vec![PNode::Delay { feature: 0, offset: 2 }],
        Vec::new(),
        Metadata::default(),
    );

    assert_eq!(
        dynamic_expressions::delay_validity_mask(&expr.nodes, 6, None),
        vec![false, false, true, true, true, true]
    );
    assert_eq!(
        dynamic_expressions::delay_validity_mask(&expr.nodes, 6, Some(&[0, 0, 0, 1, 1, 1])),
        vec![false, false, true, false, false, true]
    );
}

#[test]
fn search_can_use_delay_leaf_without_lag_expanded_inputs() {
    const D: usize = 3;
    let n_rows = 96usize;
    let mut x_values = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        x_values.push(((i as f64) / 7.0).sin());
    }

    let x = Array2::from_shape_vec((1, n_rows), x_values.clone()).unwrap();
    let mut y = Array1::<f64>::zeros(n_rows);
    for row in 0..n_rows {
        y[row] = x_values[row.saturating_sub(3)];
    }
    let dataset = Dataset::with_weights_and_names(x, y, None, vec!["x".into()]);

    let options = Options::<f64, D> {
        seed: 2,
        niterations: 25,
        populations: 4,
        population_size: 64,
        ncycles_per_iteration: 300,
        maxsize: 5,
        maxdepth: 3,
        max_delay: 4,
        delay_probability: 0.7,
        should_optimize_constants: false,
        progress: false,
        operators: BuiltinOpsF64::from_names(["+", "sub", "*"]).unwrap(),
        ..Default::default()
    };

    let result = equation_search::<f64, BuiltinOpsF64, D>(&dataset, &options);
    let expr = string_tree(
        &result.best.expr,
        StringTreeOptions {
            variable_names: Some(&["x".to_string()]),
            ..Default::default()
        },
    );

    assert!(
        expr.contains("delay(x, 3)") || result.best.loss < 1e-8,
        "best expression {expr:?} had loss {}",
        result.best.loss
    );
}

#[test]
fn delay_complexity_counts_variable_operation_and_offset() {
    let expr = PostfixExpr::<f64, BuiltinOpsF64, 2>::new(
        vec![PNode::Delay { feature: 0, offset: 3 }],
        Vec::new(),
        Metadata::default(),
    );
    let options = Options::<f64, 2> {
        complexity_of_variables: 1,
        complexity_of_delay: 1,
        complexity_of_delay_offset: 1,
        variable_complexities: Some(vec![1]),
        ..Default::default()
    };

    assert_eq!(symbolic_regression::compute_complexity(&expr.nodes, &options), 3);
}
