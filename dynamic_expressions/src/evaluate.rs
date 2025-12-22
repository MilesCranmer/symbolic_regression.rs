use crate::compile::{compile_plan, EvalPlan};
use crate::expression::PostfixExpr;
use crate::node::Src;
use crate::operator_enum::scalar::{EvalKernelCtx, GradRef, OpId, ScalarOpSet, SrcRef};
use ndarray::ArrayView2;
use num_traits::Float;

#[derive(Copy, Clone, Debug)]
pub struct EvalOptions {
    pub check_finite: bool,
    pub early_exit: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            check_finite: true,
            early_exit: true,
        }
    }
}

#[derive(Debug)]
pub struct EvalContext<T: Float, const D: usize> {
    pub scratch: Vec<Vec<T>>, // slot-major, each len n_rows
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> EvalContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            scratch: Vec::new(),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize) {
        if self.scratch.len() < n_slots {
            self.scratch.resize_with(n_slots, Vec::new);
        }
        for slot in &mut self.scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
    }
}

fn slot_slice<'a, T>(
    slot: usize,
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> &'a [T] {
    if slot < dst_slot {
        &before[slot]
    } else if slot > dst_slot {
        &after[slot - dst_slot - 1]
    } else {
        panic!("source references dst slot");
    }
}

pub(crate) fn resolve_val_src<'a, T: Float>(
    src: Src,
    x_data: &'a [T],
    n_features: usize,
    consts: &'a [T],
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> SrcRef<'a, T> {
    match src {
        Src::Var(f) => SrcRef::Strided {
            data: x_data,
            offset: f as usize,
            stride: n_features,
        },
        Src::Const(c) => SrcRef::Const(consts[c as usize]),
        Src::Slot(s) => SrcRef::Slice(slot_slice(s as usize, dst_slot, before, after)),
    }
}

pub(crate) fn resolve_der_src<'a, T: Float>(
    src: Src,
    direction: usize,
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> SrcRef<'a, T> {
    match src {
        Src::Var(f) => {
            if f as usize == direction {
                SrcRef::Const(T::one())
            } else {
                SrcRef::Const(T::zero())
            }
        }
        Src::Const(_) => SrcRef::Const(T::zero()),
        Src::Slot(s) => SrcRef::Slice(slot_slice(s as usize, dst_slot, before, after)),
    }
}

pub(crate) fn resolve_grad_src<'a, T: Float>(
    src: Src,
    variable: bool,
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> GradRef<'a, T> {
    match src {
        Src::Var(f) => {
            if variable {
                GradRef::Basis(f as usize)
            } else {
                GradRef::Zero
            }
        }
        Src::Const(c) => {
            if variable {
                GradRef::Zero
            } else {
                GradRef::Basis(c as usize)
            }
        }
        Src::Slot(s) => GradRef::Slice(slot_slice(s as usize, dst_slot, before, after)),
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_enum::scalar::SrcRef;
    use ndarray::Array2;

    #[test]
    fn slot_slice_panics_if_src_references_dst() {
        let x = Array2::from_shape_vec((2, 1), vec![0.0f64; 2]).unwrap();
        let x_data = x.as_slice().unwrap();
        let n_features = x.ncols();
        let consts: Vec<f64> = vec![];
        let before: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        let after: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];

        let dst_slot = 0usize;
        let res = std::panic::catch_unwind(|| {
            resolve_val_src(
                Src::Slot(dst_slot as u16),
                x_data,
                n_features,
                &consts,
                dst_slot,
                &before,
                &after,
            )
        });
        assert!(res.is_err());

        let r = resolve_val_src(
            Src::Slot(1),
            x_data,
            n_features,
            &consts,
            dst_slot,
            &before,
            &after,
        );
        assert!(matches!(r, SrcRef::Slice(s) if s.len() == 2));
    }

    #[test]
    fn slot_slice_uses_before_when_slot_is_less_than_dst() {
        let x = Array2::from_shape_vec((2, 1), vec![0.0f64; 2]).unwrap();
        let x_data = x.as_slice().unwrap();
        let n_features = x.ncols();
        let consts: Vec<f64> = vec![];
        let before: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        let after: Vec<Vec<f64>> = vec![vec![3.0, 4.0]];

        let dst_slot = 1usize;
        let r = resolve_val_src(
            Src::Slot(0),
            x_data,
            n_features,
            &consts,
            dst_slot,
            &before,
            &after,
        );
        assert!(matches!(r, SrcRef::Slice(s) if s.len() == 2 && s[0] == 1.0 && s[1] == 2.0));
    }
}
