use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use crate::compile::{EvalPlan, compile_plan};
use crate::dispatch::{GradKernelCtx, GradRef, SrcRef};
use crate::evaluate::{EvalOptions, resolve_val_src};
use crate::expression::PostfixExpr;
use crate::node::Src;
use crate::traits::{OpId, OperatorSet};
use crate::utils::ZipEq;

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

/// Directional derivatives are Jacobians with `n_dir = 1`.
pub type DiffContext<T, const D: usize> = GradContext<T, D>;

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

#[derive(Copy, Clone, Debug)]
enum JacTarget {
    Variables,
    Constants,
    VariableDir(usize),
}

#[inline]
fn n_dir_for_target(target: JacTarget, n_features: usize, n_consts: usize) -> usize {
    match target {
        JacTarget::Variables => n_features,
        JacTarget::Constants => n_consts,
        JacTarget::VariableDir(_) => 1,
    }
}

fn resolve_jac_src<'a, T: Float>(
    src: Src,
    target: JacTarget,
    dst_slot: usize,
    before: &'a [T],
    after: &'a [T],
    slot_stride: usize,
) -> GradRef<'a, T> {
    match src {
        Src::Var(f) => match target {
            JacTarget::Variables => GradRef::Basis(f as usize),
            JacTarget::Constants => GradRef::Zero,
            JacTarget::VariableDir(dir) => {
                if f as usize == dir {
                    GradRef::Basis(0)
                } else {
                    GradRef::Zero
                }
            }
        },
        Src::Const(c) => match target {
            JacTarget::Variables | JacTarget::VariableDir(_) => GradRef::Zero,
            JacTarget::Constants => GradRef::Basis(c as usize),
        },
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

fn eval_jac_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    target: JacTarget,
    ctx: &mut GradContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, GradMatrix<T>, bool)
where
    T: Float + core::ops::AddAssign,
    Ops: OperatorSet<T = T>,
{
    assert!(x_columns.is_standard_layout(), "X must be contiguous");
    assert_eq!(ctx.n_rows, x_columns.ncols());
    let n_rows = x_columns.ncols();
    let n_features = x_columns.nrows();
    let n_dir = n_dir_for_target(target, n_features, expr.consts.len());

    let needs_recompile = ctx.plan_nodes_len != expr.nodes.len()
        || ctx.plan_n_consts != expr.consts.len()
        || ctx.plan_n_features != n_features
        || ctx.plan.as_ref().is_none_or(|p| p.hash != expr.hash_nodes());
    if needs_recompile {
        ctx.plan = Some(compile_plan::<D>(&expr.nodes, n_features, expr.consts.len()));
        ctx.plan_nodes_len = expr.nodes.len();
        ctx.plan_n_consts = expr.consts.len();
        ctx.plan_n_features = n_features;
    }
    let n_slots = ctx.plan.as_ref().unwrap().n_slots;
    ctx.ensure_scratch(n_slots, n_dir);
    let plan: &EvalPlan<D> = ctx.plan.as_ref().unwrap();

    let mut complete = true;
    let val_scratch = ctx
        .val_scratch
        .as_slice_mut()
        .expect("value scratch must be contiguous");
    let jac_scratch = ctx
        .grad_scratch
        .as_slice_mut()
        .expect("grad scratch must be contiguous");
    let x_data = x_columns.as_slice().expect("X must be contiguous");
    let slot_stride = n_rows;
    let jac_stride = n_rows * n_dir;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let dst_start = dst_slot * slot_stride;
        let (val_before, val_rest) = val_scratch.split_at_mut(dst_start);
        let (dst_val, val_after) = val_rest.split_at_mut(slot_stride);

        let jac_dst_start = dst_slot * jac_stride;
        let (jac_before, jac_rest) = jac_scratch.split_at_mut(jac_dst_start);
        let (dst_jac, jac_after) = jac_rest.split_at_mut(jac_stride);

        let mut args_refs: [SrcRef<'_, T>; D] = [SrcRef::Const(T::zero()); D];
        let mut arg_jacs: [GradRef<'_, T>; D] = [GradRef::Zero; D];
        for (j, (dst_a, dst_ja)) in args_refs
            .iter_mut()
            .take(arity)
            .zip_eq(arg_jacs.iter_mut().take(arity))
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
            *dst_ja = resolve_jac_src(instr.args[j], target, dst_slot, jac_before, jac_after, jac_stride);
        }

        let ok = Ops::grad(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            GradKernelCtx {
                out_val: dst_val,
                out_grad: dst_jac,
                args: &args_refs[..arity],
                arg_grads: &arg_jacs[..arity],
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
    let mut out_jac = vec![T::zero(); n_dir * n_rows];
    match plan.root {
        Src::Var(f) => {
            let start = f as usize * n_rows;
            let end = start + n_rows;
            out_val.copy_from_slice(&x_data[start..end]);
            match target {
                JacTarget::Variables => {
                    for (dir, jac_dir) in out_jac.chunks_mut(n_rows).enumerate() {
                        if dir == f as usize {
                            jac_dir.fill(T::one());
                        } else {
                            jac_dir.fill(T::zero());
                        }
                    }
                }
                JacTarget::Constants => {}
                JacTarget::VariableDir(dir) => {
                    if f as usize == dir {
                        out_jac.fill(T::one());
                    } else {
                        out_jac.fill(T::zero());
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
            match target {
                JacTarget::Variables | JacTarget::VariableDir(_) => {}
                JacTarget::Constants => {
                    for (dir, jac_dir) in out_jac.chunks_mut(n_rows).enumerate() {
                        if dir == c as usize {
                            jac_dir.fill(T::one());
                        } else {
                            jac_dir.fill(T::zero());
                        }
                    }
                }
            }
        }
        Src::Slot(s) => {
            let start = s as usize * n_rows;
            let end = start + n_rows;
            out_val.copy_from_slice(&val_scratch[start..end]);

            let jac_start = s as usize * jac_stride;
            let jac_end = jac_start + jac_stride;
            out_jac.copy_from_slice(&jac_scratch[jac_start..jac_end]);
        }
    }

    (
        out_val,
        GradMatrix {
            data: out_jac,
            n_dir,
            n_rows,
        },
        complete,
    )
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
    Ops: OperatorSet<T = T>,
{
    let target = if variable {
        JacTarget::Variables
    } else {
        JacTarget::Constants
    };
    eval_jac_tree_array(expr, x_columns, target, ctx, opts)
}

pub fn eval_diff_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    direction: usize,
    ctx: &mut DiffContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, Vec<T>, bool)
where
    T: Float + core::ops::AddAssign,
    Ops: OperatorSet<T = T>,
{
    assert!(x_columns.is_standard_layout(), "X must be contiguous");
    assert!(direction < x_columns.nrows());
    let (out_val, jac, complete) = eval_jac_tree_array(expr, x_columns, JacTarget::VariableDir(direction), ctx, opts);
    debug_assert_eq!(jac.n_dir, 1);
    (out_val, jac.data, complete)
}
