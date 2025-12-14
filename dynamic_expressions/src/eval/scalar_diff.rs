use crate::compile::{compile_plan, EvalPlan};
use crate::eval::context::{DiffContext, EvalOptions};
use crate::eval::resolve::{resolve_der_src, resolve_val_src};
use crate::expr::{PostfixExpr, Src};
use crate::operators::scalar::{DiffKernelCtx, OpId, ScalarOpSet, SrcRef};
use ndarray::ArrayView2;
use num_traits::Float;

pub fn eval_diff_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    direction: usize,
    ctx: &mut DiffContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, Vec<T>, bool)
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    assert!(direction < x.ncols());
    assert_eq!(ctx.n_rows, x.nrows());
    let n_rows = x.nrows();
    let x_data = x.as_slice().expect("X must be contiguous");
    let n_features = x.ncols();

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
    let n_slots = ctx.plan.as_ref().unwrap().n_slots;
    ctx.ensure_scratch(n_slots);
    let plan: &EvalPlan<D> = ctx.plan.as_ref().unwrap();

    let mut complete = true;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let (val_before, val_rest) = ctx.val_scratch.split_at_mut(dst_slot);
        let (dst_val, val_after) = val_rest.split_first_mut().unwrap();
        let (der_before, der_rest) = ctx.der_scratch.split_at_mut(dst_slot);
        let (dst_der, der_after) = der_rest.split_first_mut().unwrap();

        let mut args_refs: [SrcRef<'_, T>; D] = core::array::from_fn(|_| SrcRef::Const(T::zero()));
        let mut dargs_refs: [SrcRef<'_, T>; D] = core::array::from_fn(|_| SrcRef::Const(T::zero()));
        for (j, (dst_a, dst_da)) in args_refs
            .iter_mut()
            .take(arity)
            .zip(dargs_refs.iter_mut().take(arity))
            .enumerate()
        {
            *dst_a = resolve_val_src(
                instr.args[j],
                x_data,
                n_features,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *dst_da = resolve_der_src(instr.args[j], direction, dst_slot, der_before, der_after);
        }

        let ok = Ops::diff(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            DiffKernelCtx {
                out_val: dst_val,
                out_der: dst_der,
                args: &args_refs[..arity],
                dargs: &dargs_refs[..arity],
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            return (vec![T::nan(); n_rows], vec![T::nan(); n_rows], false);
        }
    }

    match plan.root {
        Src::Var(f) => {
            let eval = (0..n_rows)
                .map(|row| x_data[row * n_features + (f as usize)])
                .collect();
            let der = if f as usize == direction {
                vec![T::one(); n_rows]
            } else {
                vec![T::zero(); n_rows]
            };
            (eval, der, complete)
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    return (vec![T::nan(); n_rows], vec![T::nan(); n_rows], false);
                }
            }
            (vec![v; n_rows], vec![T::zero(); n_rows], complete)
        }
        Src::Slot(s) => {
            let eval = ctx.val_scratch[s as usize].clone();
            let der = ctx.der_scratch[s as usize].clone();
            (eval, der, complete)
        }
    }
}
