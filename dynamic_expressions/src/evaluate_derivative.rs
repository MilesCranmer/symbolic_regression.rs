use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use crate::compile::{EvalPlan, build_node_hash, compile_plan};
use crate::evaluate::{EvalOptions, resolve_der_src, resolve_grad_src, resolve_val_src};
use crate::expression::PostfixExpr;
use crate::node::Src;
use crate::operator_enum::scalar::{DiffKernelCtx, GradKernelCtx, GradRef, OpId, ScalarOpSet, SrcRef};
use crate::utils::ZipEq;

#[derive(Debug)]
pub struct DiffContext<T: Float, const D: usize> {
    pub val_scratch: Array2<T>,
    pub der_scratch: Array2<T>,
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> DiffContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            val_scratch: Array2::zeros((0, 0)),
            der_scratch: Array2::zeros((0, 0)),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize) {
        if self.val_scratch.nrows() != n_slots || self.val_scratch.ncols() != self.n_rows {
            self.val_scratch = Array2::zeros((n_slots, self.n_rows));
        }
        if self.der_scratch.nrows() != n_slots || self.der_scratch.ncols() != self.n_rows {
            self.der_scratch = Array2::zeros((n_slots, self.n_rows));
        }
    }
}

#[derive(Debug)]
pub struct GradContext<T: Float, const D: usize> {
    pub val_scratch: Array2<T>,
    pub grad_scratch: Array2<T>, // slot-major, each len n_dir*n_rows
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> GradContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            val_scratch: Array2::zeros((0, 0)),
            grad_scratch: Array2::zeros((0, 0)),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize, n_dir: usize) {
        if self.val_scratch.nrows() != n_slots || self.val_scratch.ncols() != self.n_rows {
            self.val_scratch = Array2::zeros((n_slots, self.n_rows));
        }
        let grad_len = n_dir * self.n_rows;
        if self.grad_scratch.nrows() != n_slots || self.grad_scratch.ncols() != grad_len {
            self.grad_scratch = Array2::zeros((n_slots, grad_len));
        }
    }
}

#[derive(Clone, Debug)]
pub struct GradMatrix<T> {
    pub data: Vec<T>,
    pub n_dir: usize,
    pub n_rows: usize,
}

fn nan_grad_return<T: Float>(n_rows: usize, n_dir: usize) -> (Vec<T>, GradMatrix<T>, bool) {
    (
        vec![T::nan(); n_rows],
        GradMatrix {
            data: vec![T::nan(); n_dir * n_rows],
            n_dir,
            n_rows,
        },
        false,
    )
}

pub fn eval_diff_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    direction: usize,
    ctx: &mut DiffContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, Vec<T>, bool)
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(x_columns.is_standard_layout(), "X must be contiguous");
    assert!(direction < x_columns.nrows());
    assert_eq!(ctx.n_rows, x_columns.ncols());
    let n_rows = x_columns.ncols();
    let n_features = x_columns.nrows();

    let needs_recompile = ctx.plan_nodes_len != expr.nodes.len()
        || ctx.plan_n_consts != expr.consts.len()
        || ctx.plan_n_features != n_features
        || ctx.plan.as_ref().is_none_or(|p| p.hash != build_node_hash(&expr.nodes));
    if needs_recompile {
        ctx.plan = Some(compile_plan::<D>(&expr.nodes, n_features, expr.consts.len()));
        ctx.plan_nodes_len = expr.nodes.len();
        ctx.plan_n_consts = expr.consts.len();
        ctx.plan_n_features = n_features;
    }
    let n_slots = ctx.plan.as_ref().unwrap().n_slots;
    ctx.ensure_scratch(n_slots);
    let plan: &EvalPlan<D> = ctx.plan.as_ref().unwrap();

    let mut complete = true;
    let val_scratch = ctx
        .val_scratch
        .as_slice_mut()
        .expect("value scratch must be contiguous");
    let der_scratch = ctx
        .der_scratch
        .as_slice_mut()
        .expect("derivative scratch must be contiguous");
    let x_data = x_columns.as_slice().expect("X must be contiguous");
    let slot_stride = n_rows;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let dst_start = dst_slot * slot_stride;
        let (val_before, val_rest) = val_scratch.split_at_mut(dst_start);
        let (dst_val, val_after) = val_rest.split_at_mut(slot_stride);

        let (der_before, der_rest) = der_scratch.split_at_mut(dst_start);
        let (dst_der, der_after) = der_rest.split_at_mut(slot_stride);

        let mut args_refs: [SrcRef<'_, T>; D] = [SrcRef::Const(T::zero()); D];
        let mut dargs_refs: [SrcRef<'_, T>; D] = [SrcRef::Const(T::zero()); D];
        for (j, (dst_a, dst_da)) in args_refs
            .iter_mut()
            .take(arity)
            .zip_eq(dargs_refs.iter_mut().take(arity))
            .enumerate()
        {
            *dst_a = resolve_val_src(
                instr.args[j],
                x_data,
                n_rows,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *dst_da = resolve_der_src(instr.args[j], direction, dst_slot, der_before, der_after, n_rows);
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
            let nan = T::nan();
            return (vec![nan; n_rows], vec![nan; n_rows], false);
        }
    }

    let mut out = vec![T::zero(); n_rows];
    let mut der = vec![T::zero(); n_rows];
    match plan.root {
        Src::Var(f) => {
            let start = f as usize * n_rows;
            let end = start + n_rows;
            out.copy_from_slice(&x_data[start..end]);
            for v in &mut der {
                *v = if f as usize == direction { T::one() } else { T::zero() };
            }
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    let nan = T::nan();
                    return (vec![nan; n_rows], vec![nan; n_rows], false);
                }
            }
            out.fill(v);
            der.fill(T::zero());
        }
        Src::Slot(s) => {
            let start = s as usize * n_rows;
            let end = start + n_rows;
            out.copy_from_slice(&val_scratch[start..end]);
            der.copy_from_slice(&der_scratch[start..end]);
        }
    }

    (out, der, complete)
}

pub fn eval_grad_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    variable: bool,
    ctx: &mut GradContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, GradMatrix<T>, bool)
where
    T: Float + core::ops::AddAssign,
    Ops: ScalarOpSet<T>,
{
    assert!(x_columns.is_standard_layout(), "X must be contiguous");
    assert_eq!(ctx.n_rows, x_columns.ncols());

    let n_rows = x_columns.ncols();
    let n_dir = if variable { x_columns.nrows() } else { expr.consts.len() };

    let needs_recompile = ctx.plan_nodes_len != expr.nodes.len()
        || ctx.plan_n_consts != expr.consts.len()
        || ctx.plan_n_features != x_columns.nrows()
        || ctx.plan.as_ref().is_none_or(|p| p.hash != build_node_hash(&expr.nodes));

    if needs_recompile {
        ctx.plan = Some(compile_plan::<D>(&expr.nodes, x_columns.nrows(), expr.consts.len()));
        ctx.plan_nodes_len = expr.nodes.len();
        ctx.plan_n_consts = expr.consts.len();
        ctx.plan_n_features = x_columns.nrows();
    }

    ctx.ensure_scratch(ctx.plan.as_ref().unwrap().n_slots, n_dir);
    let plan = ctx.plan.as_ref().unwrap();
    let mut complete = true;

    let val_scratch = ctx
        .val_scratch
        .as_slice_mut()
        .expect("value scratch must be contiguous");
    let grad_scratch = ctx
        .grad_scratch
        .as_slice_mut()
        .expect("grad scratch must be contiguous");
    let x_data = x_columns.as_slice().expect("X must be contiguous");
    let slot_stride = n_rows;
    let grad_stride = n_rows * n_dir;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let dst_start = dst_slot * slot_stride;
        let (val_before, val_rest) = val_scratch.split_at_mut(dst_start);
        let (dst_val, val_after) = val_rest.split_at_mut(slot_stride);

        let grad_dst_start = dst_slot * grad_stride;
        let (grad_before, grad_rest) = grad_scratch.split_at_mut(grad_dst_start);
        let (dst_grad, grad_after) = grad_rest.split_at_mut(grad_stride);

        let mut args_refs: [SrcRef<'_, T>; D] = [SrcRef::Const(T::zero()); D];
        let mut arg_grads: [GradRef<'_, T>; D] = [GradRef::Zero; D];
        for (j, (dst_a, dst_ga)) in args_refs
            .iter_mut()
            .take(arity)
            .zip_eq(arg_grads.iter_mut().take(arity))
            .enumerate()
        {
            *dst_a = resolve_val_src(
                instr.args[j],
                x_data,
                n_rows,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *dst_ga = resolve_grad_src(instr.args[j], variable, dst_slot, grad_before, grad_after, grad_stride);
        }

        let ok = Ops::grad(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            GradKernelCtx {
                out_val: dst_val,
                out_grad: dst_grad,
                args: &args_refs[..arity],
                arg_grads: &arg_grads[..arity],
                n_dir,
                n_rows,
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            return nan_grad_return(n_rows, n_dir);
        }
    }

    let mut out_val = vec![T::zero(); n_rows];
    let mut out_grad = vec![T::zero(); n_dir * n_rows];
    match plan.root {
        Src::Var(f) => {
            let start = f as usize * n_rows;
            let end = start + n_rows;
            out_val.copy_from_slice(&x_data[start..end]);
            if variable {
                for (dir, grad_dir) in out_grad.chunks_mut(n_rows).enumerate() {
                    if dir == f as usize {
                        grad_dir.fill(T::one());
                    } else {
                        grad_dir.fill(T::zero());
                    }
                }
            }
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    return nan_grad_return(n_rows, n_dir);
                }
            }
            out_val.fill(v);
            if !variable {
                for (dir, grad_dir) in out_grad.chunks_mut(n_rows).enumerate() {
                    if dir == c as usize {
                        grad_dir.fill(T::one());
                    } else {
                        grad_dir.fill(T::zero());
                    }
                }
            }
        }
        Src::Slot(s) => {
            let start = s as usize * n_rows;
            let end = start + n_rows;
            out_val.copy_from_slice(&val_scratch[start..end]);

            let grad_start = s as usize * grad_stride;
            let grad_end = grad_start + grad_stride;
            out_grad.copy_from_slice(&grad_scratch[grad_start..grad_end]);
        }
    }

    (
        out_val,
        GradMatrix {
            data: out_grad,
            n_dir,
            n_rows,
        },
        complete,
    )
}
