use dynamic_expressions::operator_enum::builtin::Add;
use dynamic_expressions::operator_enum::scalar::{
    diff_apply, diff_nary, eval_apply, eval_nary, grad_apply, grad_nary, GradKernelCtx, GradRef,
    SrcRef,
};
use dynamic_expressions::EvalOptions;
use std::hint::black_box;

fn id_eval(args: &[f64; 1]) -> f64 {
    args[0]
}

fn nan_on_second_row_eval(args: &[f64; 1]) -> f64 {
    if args[0] == 2.0 {
        f64::NAN
    } else {
        args[0]
    }
}

fn one_partial(_args: &[f64; 1], idx: usize) -> f64 {
    match idx {
        0 => 1.0,
        _ => unreachable!(),
    }
}

#[test]
fn eval_nary_no_checks_returns_true() {
    let xs = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: false,
    });
    assert!(eval_nary::<1, f64>(
        nan_on_second_row_eval,
        &mut out,
        &args,
        &opts
    ));
}

#[test]
fn eval_nary_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!eval_nary::<1, f64>(
        nan_on_second_row_eval,
        &mut out,
        &args,
        &opts
    ));
    assert!(out[1].is_nan());
}

#[test]
fn eval_nary_const_only_check_finite_false_returns_true() {
    let args = [SrcRef::Const(1.0)];
    let mut out = [0.0; 4];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(eval_nary::<1, f64>(id_eval, &mut out, &args, &opts));
}

#[test]
fn eval_nary_const_only_nonfinite_sets_complete_false() {
    let args = [SrcRef::Const(f64::NAN)];
    let mut out = [0.0; 4];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: true,
    });
    assert!(!eval_nary::<1, f64>(id_eval, &mut out, &args, &opts));
}

#[test]
fn eval_apply_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, f64::NAN, 3.0];
    let ys = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!eval_apply::<2, f64, Add>(&mut out, &args, &opts));
    assert!(out[1].is_nan());
}

#[test]
fn eval_apply_const_only_check_finite_false_returns_true() {
    let args = [SrcRef::Const(1.0), SrcRef::Const(2.0)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(eval_apply::<2, f64, Add>(&mut out, &args, &opts));
}

#[test]
fn diff_apply_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, f64::NAN, 3.0];
    let ys = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let dys = [0.0, 0.0, 0.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let dargs = [SrcRef::Slice(&dxs), SrcRef::Slice(&dys)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!diff_apply::<2, f64, Add>(
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
    assert!(out_val[1].is_nan());
}

#[test]
fn diff_nary_no_checks_returns_true() {
    let xs = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let args = [SrcRef::Slice(&xs)];
    let dargs = [SrcRef::Slice(&dxs)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(diff_nary::<1, f64>(
        nan_on_second_row_eval,
        one_partial,
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
}

#[test]
fn diff_nary_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let args = [SrcRef::Slice(&xs)];
    let dargs = [SrcRef::Slice(&dxs)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!diff_nary::<1, f64>(
        nan_on_second_row_eval,
        one_partial,
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
    assert!(out_val[1].is_nan());
}

#[test]
fn diff_apply_no_checks_returns_true() {
    let xs = [1.0, f64::NAN, 3.0];
    let ys = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let dys = [0.0, 0.0, 0.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let dargs = [SrcRef::Slice(&dxs), SrcRef::Slice(&dys)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(diff_apply::<2, f64, Add>(
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
}

#[test]
fn grad_apply_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, f64::NAN];
    let ys = [1.0, 2.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let arg_grads = [GradRef::Zero, GradRef::Zero];
    let mut out_val = [0.0; 2];
    let mut out_grad = [0.0; 2];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!grad_apply::<2, f64, Add>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 2,
        opts: &opts,
    }));
    assert!(out_val[1].is_nan());
}

#[test]
fn grad_apply_no_checks_returns_true() {
    let xs = [1.0, f64::NAN];
    let ys = [1.0, 2.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let arg_grads = [GradRef::Zero, GradRef::Zero];
    let mut out_val = [0.0; 2];
    let mut out_grad = [0.0; 2];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(grad_apply::<2, f64, Add>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 2,
        opts: &opts,
    }));
}

#[test]
fn grad_nary_no_checks_returns_true() {
    let xs = [1.0, 2.0];
    let args = [SrcRef::Slice(&xs)];
    let arg_grads = [GradRef::Zero];
    let mut out_val = [0.0; 2];
    let mut out_grad = [0.0; 2];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(grad_nary::<1, f64>(
        nan_on_second_row_eval,
        one_partial,
        GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir: 1,
            n_rows: 2,
            opts: &opts,
        }
    ));
}

#[test]
fn grad_nary_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs)];
    let arg_grads = [GradRef::Zero];
    let mut out_val = [0.0; 3];
    let mut out_grad = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!grad_nary::<1, f64>(
        nan_on_second_row_eval,
        one_partial,
        GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir: 1,
            n_rows: 3,
            opts: &opts,
        }
    ));
    assert!(out_val[1].is_nan());
}

#[test]
fn maybe_mark_nonfinite_can_early_exit() {
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: true,
    });
    let mut complete = true;
    let f: fn(f64, &EvalOptions, &mut bool) -> bool =
        dynamic_expressions::operator_enum::scalar::__maybe_mark_nonfinite::<f64>;
    let f = black_box(f);
    let ok = f(black_box(f64::INFINITY), &opts, &mut complete);
    assert!(!ok);
    assert!(!complete);
}

#[test]
fn maybe_mark_nonfinite_marks_complete_without_early_exit() {
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    let mut complete = true;
    let f: fn(f64, &EvalOptions, &mut bool) -> bool =
        dynamic_expressions::operator_enum::scalar::__maybe_mark_nonfinite::<f64>;
    let f = black_box(f);
    let ok = f(black_box(f64::INFINITY), &opts, &mut complete);
    assert!(ok);
    assert!(!complete);
}
