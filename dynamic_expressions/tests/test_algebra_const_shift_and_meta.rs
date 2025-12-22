mod common;

use common::{assert_close_vec, make_x, var, TestOps};
use dynamic_expressions::{eval_tree_array, EvalOptions};

#[test]
fn combining_constant_exprs_shifts_const_indices_and_merges_metadata() {
    let (x_data, x) = make_x(2, 32);
    let x_view = x.view();

    let a = var(0) + 1.0;
    let mut b = var(1) + 2.0;
    b.meta.variable_names = vec!["u".to_string(), "v".to_string()];

    let expr = a + b;
    assert_eq!(expr.consts.len(), 2);
    assert_eq!(
        expr.meta.variable_names,
        vec!["u".to_string(), "v".to_string()]
    );

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (y, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x_view, &opts);
    assert!(ok);

    let mut y_ref = vec![0.0f64; x_view.nrows()];
    for row in 0..x_view.nrows() {
        y_ref[row] =
            (x_data[row * x_view.ncols()] + 1.0) + (x_data[row * x_view.ncols() + 1] + 2.0);
    }
    assert_close_vec(&y, &y_ref, 1e-12);
}
