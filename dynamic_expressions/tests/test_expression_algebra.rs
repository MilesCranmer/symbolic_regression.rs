mod common;

use approx::assert_relative_eq;
use common::{assert_close_vec, expr_readme_like, make_x, var, TestOps};
use dynamic_expressions::{eval_tree_array, lit, EvalOptions};

#[test]
fn build_expr_with_algebra_and_eval() {
    use dynamic_expressions::math::cos;

    let (_x_data, x) = make_x(2, 64);
    let x_view = x.view();

    let x1 = var(0);
    let x2 = var(1);

    let ex = x1 * cos(x2 - 3.2);

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (y, ok) = eval_tree_array::<f64, TestOps, 3>(&ex, x_view, &opts);
    assert!(ok);

    let ref_ex = expr_readme_like();
    let (y2, ok2) = eval_tree_array::<f64, TestOps, 3>(&ref_ex, x_view, &opts);
    assert!(ok2);

    assert_close_vec(&y, &y2, 1e-12);
}

#[test]
fn lit_left_mul_works() {
    let (x_data, x) = make_x(2, 32);
    let x_view = x.view();

    let ex = lit(2.0) * var(0);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (y, ok) = eval_tree_array::<f64, TestOps, 3>(&ex, x_view, &opts);
    assert!(ok);

    for (i, &v) in y.iter().enumerate() {
        assert_relative_eq!(
            v,
            2.0 * x_data[i * x_view.ncols()],
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
}
