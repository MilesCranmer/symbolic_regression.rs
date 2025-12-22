mod common;

use common::{Op1, TestOps};
use dynamic_expressions::operator_enum::scalar::{
    DiffKernelCtx, EvalKernelCtx, GradKernelCtx, GradRef, OpId, ScalarOpSet, SrcRef,
};
use dynamic_expressions::strings::OpNames;
use dynamic_expressions::EvalOptions;

#[test]
fn opset_op_names_none_infix_branch_is_exercised() {
    // Cos is a non-infix operator, so OpNames should return its NAME.
    let name = <TestOps as OpNames>::op_name(OpId {
        arity: 1,
        id: Op1::Cos as u16,
    });
    assert_eq!(name, "cos");
}

#[test]
#[should_panic]
fn opset_eval_panics_on_unknown_id() {
    let mut out = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <TestOps as ScalarOpSet<f64>>::eval(
        OpId { arity: 1, id: 999 },
        EvalKernelCtx {
            out: &mut out,
            args: &args,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn opset_eval_panics_on_unsupported_arity() {
    let mut out = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <TestOps as ScalarOpSet<f64>>::eval(
        OpId { arity: 9, id: 0 },
        EvalKernelCtx {
            out: &mut out,
            args: &args,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn opset_diff_panics_on_unknown_id() {
    let mut out_val = [0.0f64];
    let mut out_der = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let dargs = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <TestOps as ScalarOpSet<f64>>::diff(
        OpId { arity: 1, id: 999 },
        DiffKernelCtx {
            out_val: &mut out_val,
            out_der: &mut out_der,
            args: &args,
            dargs: &dargs,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn opset_diff_panics_on_unsupported_arity() {
    let mut out_val = [0.0f64];
    let mut out_der = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let dargs = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <TestOps as ScalarOpSet<f64>>::diff(
        OpId { arity: 9, id: 0 },
        DiffKernelCtx {
            out_val: &mut out_val,
            out_der: &mut out_der,
            args: &args,
            dargs: &dargs,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn opset_grad_panics_on_unknown_id() {
    let mut out_val = [0.0f64];
    let mut out_grad = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let arg_grads = [GradRef::Zero];
    let opts = EvalOptions::default();
    <TestOps as ScalarOpSet<f64>>::grad(
        OpId { arity: 1, id: 999 },
        GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir: 1,
            n_rows: 1,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn opset_grad_panics_on_unsupported_arity() {
    let mut out_val = [0.0f64];
    let mut out_grad = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let arg_grads = [GradRef::Zero];
    let opts = EvalOptions::default();
    <TestOps as ScalarOpSet<f64>>::grad(
        OpId { arity: 9, id: 0 },
        GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir: 1,
            n_rows: 1,
            opts: &opts,
        },
    );
}
