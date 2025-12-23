mod common;

use common::{TestOps, assert_close_vec, make_x, var};

#[test]
fn combining_constant_exprs_shifts_const_indices_and_merges_metadata() {
    let (x_data, x) = make_x(2, 32);
    let x_view = x.view();

    let a = var(0) + 1.0;
    let mut b = var(1) + 2.0;
    b.meta.variable_names = vec!["u".to_string(), "v".to_string()];

    let expr = a + b;
    assert_eq!(expr.consts.len(), 2);
    assert_eq!(expr.meta.variable_names, vec!["u".to_string(), "v".to_string()]);

    let opts = dynamic_expressions::EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (y, ok) = dynamic_expressions::eval_tree_array::<f64, TestOps, 3>(&expr, x_view, &opts);
    assert!(ok);

    let n_rows = x_view.ncols();
    let mut y_ref = vec![0.0f64; n_rows];
    for (row, yv) in y_ref.iter_mut().enumerate() {
        *yv = (x_data[row] + 1.0) + (x_data[n_rows + row] + 2.0);
    }
    assert_close_vec(&y, &y_ref, 1e-12);
}
