mod common;

use common::*;
use dynamic_expressions::{
    DiffContext, EvalOptions, GradContext, eval_diff_tree_array, eval_grad_tree_array, eval_tree_array,
};
use ndarray::Array2;

fn make_small_x() -> (Vec<f64>, Array2<f64>) {
    make_x(2, 21)
}

#[test]
fn covers_all_ops_in_eval_diff_and_grad() {
    let (x_data, x) = make_small_x();
    let n_rows = x.ncols();
    let n_features = x.nrows();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    // Unary ops: Log, Exp, Sin, Neg (via -(sin(exp(log(x1))))).
    let expr_unary = -dynamic_expressions::operators::sin(dynamic_expressions::operators::exp(
        dynamic_expressions::operators::log(var(0)),
    ));
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr_unary, x.view(), &opts);
    assert!(ok);
    let manual: Vec<f64> = (0..n_rows)
        .map(|row| {
            let v = x_data[row];
            -((v.ln()).exp().sin())
        })
        .collect();
    assert_close_vec(&out, &manual, 1e-12);
    let mut dctx = DiffContext::<f64, 3>::new(n_rows);
    let (_e, d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr_unary, x.view(), 0, &mut dctx, &opts);
    assert!(ok);
    let fd = finite_diff_dir(&expr_unary, &x_data, n_features, n_rows, 0, 1e-6);
    assert_close_vec(&d, &fd, 1e-6);
    let mut gctx = GradContext::<f64, 3>::new(n_rows);
    let (_e, _g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr_unary, x.view(), true, &mut gctx, &opts);
    assert!(ok);
    grad_matches_diffs(&expr_unary, &x.view(), true);

    // Binary ops: Add, Mul, Sub, Div (keep finite via +1.0).
    // expr_bin = ((x1 + x2) * (x1 - x2)) / (x2 + 1.0)
    let expr_bin = ((var(0) + var(1)) * (var(0) - var(1))) / (var(1) + 1.0);
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr_bin, x.view(), &opts);
    assert!(ok);
    let manual: Vec<f64> = (0..n_rows)
        .map(|row| {
            let a = x_data[row];
            let b = x_data[n_rows + row];
            ((a + b) * (a - b)) / (b + 1.0)
        })
        .collect();
    assert_close_vec(&out, &manual, 1e-12);
    let mut dctx = DiffContext::<f64, 3>::new(n_rows);
    let (_e, d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr_bin, x.view(), 1, &mut dctx, &opts);
    assert!(ok);
    let fd = finite_diff_dir(&expr_bin, &x_data, n_features, n_rows, 1, 1e-6);
    assert_close_vec(&d, &fd, 1e-6);
    grad_matches_diffs(&expr_bin, &x.view(), true);

    // Ternary op: Fma
    let expr_t3 = expr_ternary();
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr_t3, x.view(), &opts);
    assert!(ok);
    let c0 = expr_t3.consts[0];
    let manual: Vec<f64> = (0..n_rows)
        .map(|row| {
            let a = x_data[row];
            let b = x_data[n_rows + row];
            a * b + c0
        })
        .collect();
    assert_close_vec(&out, &manual, 1e-12);
    grad_matches_diffs(&expr_t3, &x.view(), true);
}
