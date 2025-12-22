mod common;

use common::{TestOps, make_x, var};
use dynamic_expressions::EvalOptions;

#[test]
fn diff_early_exit_can_trigger_inside_instruction_loop() {
    let (_owned, x) = make_x(1, 16);
    let x_view = x.view();
    let expr = var(0) / 0.0;
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut ctx = dynamic_expressions::DiffContext::<f64, 3>::new(x_view.nrows());
    let (eval, der, ok) =
        dynamic_expressions::eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut ctx, &opts);
    assert!(!ok);
    assert!(eval.iter().all(|v| v.is_nan()));
    assert!(der.iter().all(|v| v.is_nan()));
}

#[test]
fn grad_early_exit_can_trigger_inside_instruction_loop() {
    let (_owned, x) = make_x(1, 16);
    let x_view = x.view();
    let expr = var(0) / 0.0;
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut ctx = dynamic_expressions::GradContext::<f64, 3>::new(x_view.nrows());
    let (eval, grad, ok) =
        dynamic_expressions::eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, true, &mut ctx, &opts);
    assert!(!ok);
    assert!(eval.iter().all(|v| v.is_nan()));
    assert!(grad.data.iter().all(|v| v.is_nan()));
}
