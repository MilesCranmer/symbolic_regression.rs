mod common;

use common::*;
use dynamic_expressions::DiffContext;

#[test]
fn diff_matches_finite_difference_readme_like() {
    let expr = expr_readme_like();
    let (x_data, x) = make_x(2, 37);
    let n_features = x.nrows();
    let n_rows = x.ncols();
    let x_view = x.view();
    let opts = dynamic_expressions::EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut dctx = DiffContext::<f64, 3>::new(x_view.ncols());
    let (_eval, d, ok) =
        dynamic_expressions::eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 1, &mut dctx, &opts);
    assert!(ok);

    let fd = finite_diff_dir(&expr, &x_data, n_features, n_rows, 1, 1e-6);
    assert_close_vec(&d, &fd, 1e-6);
}
