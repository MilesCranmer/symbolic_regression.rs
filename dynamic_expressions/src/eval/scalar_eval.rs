use crate::compile::{compile_plan, EvalPlan};
use crate::eval::context::{EvalContext, EvalOptions};
use crate::eval::resolve::resolve_val_src;
use crate::expr::{PostfixExpr, Src};
use crate::operators::scalar::{EvalKernelCtx, OpId, ScalarOpSet, SrcRef};
use ndarray::ArrayView2;
use num_traits::Float;

pub fn eval_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    opts: &EvalOptions,
) -> (Vec<T>, bool)
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    let n_rows = x.nrows();
    let mut ctx = EvalContext::<T, D>::new(n_rows);
    let mut out = vec![T::zero(); n_rows];
    let complete = eval_tree_array_into::<T, Ops, D>(&mut out, expr, x.view(), &mut ctx, opts);
    (out, complete)
}

pub fn eval_plan_array_into<T, Ops, const D: usize>(
    out: &mut [T],
    plan: &EvalPlan<D>,
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    scratch: &mut Vec<Vec<T>>,
    opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    assert_eq!(out.len(), x.nrows());
    let n_rows = x.nrows();
    let x_data = x.as_slice().expect("X must be contiguous");
    let n_features = x.ncols();

    if scratch.len() < plan.n_slots {
        scratch.resize_with(plan.n_slots, Vec::new);
    }
    for slot in &mut scratch[..plan.n_slots] {
        if slot.len() != n_rows {
            slot.resize(n_rows, T::zero());
        }
    }

    let mut complete = true;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;
        let (before, rest) = scratch.split_at_mut(dst_slot);
        let (dst_buf, after) = rest.split_first_mut().unwrap();

        let mut args_refs: [SrcRef<'_, T>; D] = core::array::from_fn(|_| SrcRef::Const(T::zero()));
        for (j, dst) in args_refs.iter_mut().take(arity).enumerate() {
            *dst = resolve_val_src(
                instr.args[j],
                x_data,
                n_features,
                &expr.consts,
                dst_slot,
                before,
                after,
            );
        }

        let ok = Ops::eval(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            EvalKernelCtx {
                out: dst_buf,
                args: &args_refs[..arity],
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            return false;
        }
    }

    match plan.root {
        Src::Var(f) => {
            let offset = f as usize;
            for row in 0..n_rows {
                out[row] = x_data[row * n_features + offset];
            }
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    return false;
                }
            }
            out.fill(v);
        }
        Src::Slot(s) => out.copy_from_slice(&scratch[s as usize]),
    }

    complete
}

pub fn eval_tree_array_into<T, Ops, const D: usize>(
    out: &mut [T],
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    ctx: &mut EvalContext<T, D>,
    opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert_eq!(out.len(), x.nrows());
    assert_eq!(ctx.n_rows, x.nrows());

    let needs_recompile = ctx.plan.is_none()
        || ctx.plan_nodes_len != expr.nodes.len()
        || ctx.plan_n_consts != expr.consts.len()
        || ctx.plan_n_features != x.ncols();
    if needs_recompile {
        ctx.plan = Some(compile_plan::<D>(&expr.nodes, x.ncols(), expr.consts.len()));
        ctx.plan_nodes_len = expr.nodes.len();
        ctx.plan_n_consts = expr.consts.len();
        ctx.plan_n_features = x.ncols();
    }
    let plan = ctx.plan.as_ref().unwrap();

    eval_plan_array_into::<T, Ops, D>(out, plan, expr, x.view(), &mut ctx.scratch, opts)
}
