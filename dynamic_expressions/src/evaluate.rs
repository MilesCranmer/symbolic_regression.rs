use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use crate::compile::{EvalPlan, build_node_hash, compile_plan};
use crate::expression::PostfixExpr;
use crate::node::Src;
use crate::operator_enum::scalar::{EvalKernelCtx, GradRef, OpId, ScalarOpSet, SrcRef};
use crate::subtree_caching::SubtreeCache;

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
    pub scratch: Option<Array2<T>>,
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> EvalContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            scratch: None,
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn setup<Ops>(&mut self, expr: &PostfixExpr<T, Ops, D>, x_columns: ArrayView2<'_, T>)
    where
        T: Float,
        Ops: ScalarOpSet<T>,
    {
        if self.needs_recompile(expr, x_columns) {
            self.plan = Some(compile_plan::<D>(&expr.nodes, x_columns.nrows(), expr.consts.len()));
            self.plan_nodes_len = expr.nodes.len();
            self.plan_n_consts = expr.consts.len();
            self.plan_n_features = x_columns.nrows();
        }
        let n_slots = self.plan.as_ref().unwrap().n_slots;
        self.ensure_scratch(n_slots);
    }

    fn needs_recompile<Ops>(&self, expr: &PostfixExpr<T, Ops, D>, x_columns: ArrayView2<'_, T>) -> bool
    where
        T: Float,
        Ops: ScalarOpSet<T>,
    {
        if self.plan.is_none()
            || self.plan_nodes_len != expr.nodes.len()
            || self.plan_n_consts != expr.consts.len()
            || self.plan_n_features != x_columns.nrows()
        {
            return true;
        }
        let Some(plan) = &self.plan else {
            return true;
        };
        plan.hash != build_node_hash(&expr.nodes)
    }

    fn ensure_scratch(&mut self, n_slots: usize) {
        let ok = self
            .scratch
            .as_ref()
            .is_some_and(|s| s.nrows() == n_slots && s.ncols() == self.n_rows);
        if !ok {
            self.scratch = Some(Array2::zeros((n_slots, self.n_rows)));
        }
    }
}

pub(crate) fn resolve_val_src<'a, T: Float>(
    src: Src,
    x_columns: &'a [T],
    n_rows: usize,
    consts: &'a [T],
    dst_slot: usize,
    before: &'a [T],
    after: &'a [T],
) -> SrcRef<'a, T> {
    match src {
        Src::Var(f) => {
            let start = f as usize * n_rows;
            let end = start + n_rows;
            SrcRef::Slice(&x_columns[start..end])
        }
        Src::Const(c) => SrcRef::Const(consts[c as usize]),
        Src::Slot(s) => {
            let slot = s as usize;
            if slot < dst_slot {
                let start = slot * n_rows;
                SrcRef::Slice(&before[start..start + n_rows])
            } else if slot > dst_slot {
                let start = (slot - dst_slot - 1) * n_rows;
                SrcRef::Slice(&after[start..start + n_rows])
            } else {
                panic!("source references dst slot");
            }
        }
    }
}

pub(crate) fn resolve_der_src<'a, T: Float>(
    src: Src,
    direction: usize,
    dst_slot: usize,
    before: &'a [T],
    after: &'a [T],
    n_rows: usize,
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
        Src::Slot(s) => {
            let slot = s as usize;
            if slot < dst_slot {
                let start = slot * n_rows;
                SrcRef::Slice(&before[start..start + n_rows])
            } else if slot > dst_slot {
                let start = (slot - dst_slot - 1) * n_rows;
                SrcRef::Slice(&after[start..start + n_rows])
            } else {
                panic!("source references dst slot");
            }
        }
    }
}

pub(crate) fn resolve_grad_src<'a, T: Float>(
    src: Src,
    variable: bool,
    dst_slot: usize,
    before: &'a [T],
    after: &'a [T],
    slot_stride: usize,
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
        Src::Slot(s) => {
            let slot = s as usize;
            if slot < dst_slot {
                let start = slot * slot_stride;
                GradRef::Slice(&before[start..start + slot_stride])
            } else if slot > dst_slot {
                let start = (slot - dst_slot - 1) * slot_stride;
                GradRef::Slice(&after[start..start + slot_stride])
            } else {
                panic!("source references dst slot");
            }
        }
    }
}

pub fn eval_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    opts: &EvalOptions,
) -> (Vec<T>, bool)
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(x_columns.is_standard_layout(), "X must be contiguous");
    let n_rows = x_columns.ncols();
    let mut ctx = EvalContext::<T, D>::new(n_rows);
    let mut out = vec![T::zero(); n_rows];
    let complete = eval_tree_array_into::<T, Ops, D>(&mut out, expr, x_columns, &mut ctx, opts);
    (out, complete)
}

pub fn eval_plan_array_into<T, Ops, const D: usize>(
    out: &mut [T],
    plan: &EvalPlan<D>,
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    scratch: &mut Array2<T>,
    opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(x_columns.is_standard_layout(), "X columns must be contiguous");
    let x_data = x_columns.as_slice().expect("X columns must be contiguous in memory");
    let n_rows = x_columns.ncols();
    assert_eq!(out.len(), n_rows);

    if scratch.nrows() != plan.n_slots || scratch.ncols() != n_rows {
        *scratch = Array2::zeros((plan.n_slots, n_rows));
    }
    let scratch_data = scratch.as_slice_mut().expect("scratch buffer must stay contiguous");

    let mut complete = true;
    let slot_stride = n_rows;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;
        let dst_start = dst_slot * slot_stride;
        let (before, rest) = scratch_data.split_at_mut(dst_start);
        let (dst_buf, after) = rest.split_at_mut(slot_stride);

        let mut args_refs: [SrcRef<'_, T>; D] = [SrcRef::Const(T::zero()); D];
        for (j, dst) in args_refs.iter_mut().take(arity).enumerate() {
            *dst = resolve_val_src(instr.args[j], x_data, n_rows, &expr.consts, dst_slot, before, after);
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
            out.copy_from_slice(&x_data[f as usize * n_rows..(f as usize + 1) * n_rows]);
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
        Src::Slot(s) => {
            let start = s as usize * n_rows;
            out.copy_from_slice(&scratch_data[start..start + n_rows]);
        }
    }

    complete
}

/// Evaluate a compiled plan, reusing cached evaluations for constant-free subtrees.
///
/// This is intended for the common case where `expr.nodes` stays fixed and only
/// `expr.consts` changes between evaluations (e.g. during constant optimization).
/// Any subtree that does **not** depend on constants can be cached once per
/// dataset and then reused across calls.
///
/// `dataset_key` must uniquely identify the dataset contents. If the dataset
/// changes, pass a different key (e.g. a version counter) to automatically
/// invalidate the cache.
#[allow(clippy::too_many_arguments)]
pub fn eval_plan_array_into_cached<T, Ops, const D: usize>(
    out: &mut [T],
    plan: &EvalPlan<D>,
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    scratch: &mut Array2<T>,
    opts: &EvalOptions,
    cache: &mut SubtreeCache<T>,
    dataset_key: u64,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(x_columns.is_standard_layout(), "X must be contiguous");
    let n_rows = x_columns.ncols();
    assert_eq!(out.len(), n_rows);

    let x_data = x_columns.as_slice().expect("X must be contiguous");

    if scratch.nrows() != plan.n_slots || scratch.ncols() != n_rows {
        *scratch = Array2::zeros((plan.n_slots, n_rows));
    }
    let scratch_data = scratch.as_slice_mut().expect("scratch buffer must stay contiguous");

    // Keep cache aligned with the expression and dataset.
    cache.ensure(&expr.nodes, dataset_key, n_rows);

    let mut complete = true;
    let slot_stride = n_rows;

    for (instr_index, instr) in plan.instrs.iter().copied().enumerate() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;
        let dst_start = dst_slot * slot_stride;
        let (before, rest) = scratch_data.split_at_mut(dst_start);
        let (dst_buf, after) = rest.split_at_mut(slot_stride);

        // Fast path: reuse cached constant-free subtree value.
        if let Some(cached) = cache.get(instr_index) {
            dst_buf.copy_from_slice(&cached.val);
            if opts.check_finite && !cached.all_finite {
                complete = false;
                if opts.early_exit {
                    return false;
                }
            }
            continue;
        }

        let mut args_refs: [SrcRef<'_, T>; D] = [SrcRef::Const(T::zero()); D];
        for (j, dst) in args_refs.iter_mut().take(arity).enumerate() {
            *dst = resolve_val_src(instr.args[j], x_data, n_rows, &expr.consts, dst_slot, before, after);
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

        // Populate cache lazily after the first computation.
        cache.maybe_store(instr_index, dst_buf);
    }

    match plan.root {
        Src::Var(f) => {
            out.copy_from_slice(&x_data[f as usize * n_rows..(f as usize + 1) * n_rows]);
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
        Src::Slot(s) => {
            let start = s as usize * n_rows;
            out.copy_from_slice(&scratch_data[start..start + n_rows]);
        }
    }

    complete
}
pub fn eval_tree_array_into<T, Ops, const D: usize>(
    out: &mut [T],
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    ctx: &mut EvalContext<T, D>,
    opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert_eq!(out.len(), x_columns.ncols());
    assert_eq!(ctx.n_rows, x_columns.ncols());

    ctx.setup(expr, x_columns);

    let plan = ctx.plan.as_ref().unwrap();
    let scratch = ctx.scratch.as_mut().unwrap();

    eval_plan_array_into::<T, Ops, D>(out, plan, expr, x_columns, scratch, opts)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::operator_enum::scalar::SrcRef;

    #[test]
    fn slot_slice_panics_if_src_references_dst() {
        let x_columns = Array2::<f64>::zeros((0, 0));
        let consts: Vec<f64> = vec![];
        let n_rows = 2usize;
        let before: Vec<f64> = vec![0.0, 0.0];
        let after: Vec<f64> = vec![0.0, 0.0];

        let dst_slot = 0usize;
        let res = std::panic::catch_unwind(|| {
            resolve_val_src(
                Src::Slot(dst_slot as u16),
                x_columns.as_slice().unwrap(),
                n_rows,
                &consts,
                dst_slot,
                &before,
                &after,
            )
        });
        assert!(res.is_err());

        let r = resolve_val_src(
            Src::Slot(1),
            x_columns.as_slice().unwrap(),
            n_rows,
            &consts,
            dst_slot,
            &before,
            &after,
        );
        assert!(matches!(r, SrcRef::Slice(s) if s.len() == 2));
    }
}
