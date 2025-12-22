mod common;

use common::*;

#[test]
fn grad_variable_matches_stacked_diffs_readme_like() {
    let expr = expr_readme_like();
    let (_x_data, x) = make_x(2, 29);
    let x_view = x.view();
    grad_matches_diffs(&expr, &x_view, true);
}

#[test]
fn grad_constant_has_correct_shape() {
    let expr = expr_readme_like();
    let (_x_data, x) = make_x(2, 13);
    let x_view = x.view();
    let opts = dynamic_expressions::EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let mut gctx = dynamic_expressions::GradContext::<f64, 3>::new(x_view.nrows());
    let (_eval, grad, ok) =
        dynamic_expressions::eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, false, &mut gctx, &opts);
    assert!(ok);
    assert_eq!(grad.n_dir, expr.consts.len());
    assert_eq!(grad.n_rows, x_view.nrows());
    for v in grad.data {
        assert!(v.is_finite());
    }
}
