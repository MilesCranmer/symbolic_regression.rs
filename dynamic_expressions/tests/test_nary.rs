mod common;

use common::*;
use dynamic_expressions::{
    eval_diff_tree_array, eval_grad_tree_array, eval_tree_array, DiffContext, EvalOptions,
    GradContext,
};

#[test]
fn nary_eval_matches_manual() {
    let expr = expr_ternary();
    let (x_data, x) = make_x(2, 31);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x_view, &opts);
    assert!(ok);

    let c0 = expr.consts[0];
    let mut manual = vec![0.0f64; x_view.nrows()];
    for row in 0..x_view.nrows() {
        let x1 = x_data[row * x_view.ncols()];
        let x2 = x_data[row * x_view.ncols() + 1];
        manual[row] = x1 * x2 + c0;
    }
    assert_close_vec(&out, &manual, 1e-12);
}

#[test]
fn nary_diff_matches_finite_difference() {
    let expr = expr_ternary();
    let (x_data, x) = make_x(2, 31);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut dctx = DiffContext::<f64, 3>::new(x_view.nrows());
    let (_eval, d, ok) =
        eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(ok);

    let fd = finite_diff_dir(&expr, &x_data, 2, x_view.nrows(), 0, 1e-6);
    assert_close_vec(&d, &fd, 1e-6);
}

#[test]
fn nary_grad_matches_diffs() {
    let expr = expr_ternary();
    let (_x_data, x) = make_x(2, 17);
    let x_view = x.view();
    grad_matches_diffs(&expr, &x_view, true);

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let mut gctx = GradContext::<f64, 3>::new(x_view.nrows());
    let (_eval, grad, ok) =
        eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, true, &mut gctx, &opts);
    assert!(ok);
    assert_eq!(grad.n_dir, x_view.ncols());
    assert_eq!(grad.n_rows, x_view.nrows());
}
