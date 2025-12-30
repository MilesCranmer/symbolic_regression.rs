use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use crate::compile::{EvalPlan, compile_plan};
use crate::dispatch::{EvalKernelCtx, SrcRef};
use crate::expression::PostfixExpr;
use crate::node::Src;
use crate::traits::{OpId, OperatorSet};

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

    pub fn setup<Ops>(&mut self, expr: &PostfixExpr<T, Ops, D>, x_columns: ArrayView2<'_, T>) {
        if self.needs_recompile(expr, x_columns) {
            self.plan = Some(compile_plan::<D>(&expr.nodes, x_columns.nrows(), expr.consts.len()));
            self.plan_nodes_len = expr.nodes.len();
            self.plan_n_consts = expr.consts.len();
            self.plan_n_features = x_columns.nrows();
        }
        let n_slots = self.plan.as_ref().unwrap().n_slots;
        self.ensure_scratch(n_slots);
    }

    fn needs_recompile<Ops>(&self, expr: &PostfixExpr<T, Ops, D>, x_columns: ArrayView2<'_, T>) -> bool {
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
        plan.hash != expr.hash_nodes()
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

pub fn eval_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    opts: &EvalOptions,
) -> (Vec<T>, bool)
where
    T: Float,
    Ops: OperatorSet<T = T>,
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
    Ops: OperatorSet<T = T>,
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
pub fn eval_tree_array_into<T, Ops, const D: usize>(
    out: &mut [T],
    expr: &PostfixExpr<T, Ops, D>,
    x_columns: ArrayView2<'_, T>,
    ctx: &mut EvalContext<T, D>,
    opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    assert_eq!(out.len(), x_columns.ncols());
    assert_eq!(ctx.n_rows, x_columns.ncols());

    ctx.setup(expr, x_columns);

    let plan = ctx.plan.as_ref().unwrap();
    let scratch = ctx.scratch.as_mut().unwrap();

    eval_plan_array_into::<T, Ops, D>(out, plan, expr, x_columns, scratch, opts)
}

#[doc(hidden)]
pub mod kernels {
    use num_traits::Float;

    use super::EvalOptions;
    use crate::dispatch::{GradKernelCtx, GradRef, SrcRef, grad_at};
    use crate::utils::ZipEq;

    #[inline]
    fn __src_val<T: Float>(src: SrcRef<'_, T>, row: usize) -> T {
        match src {
            SrcRef::Slice(s) => s[row],
            SrcRef::Const(c) => c,
        }
    }

    #[inline]
    fn __all_finite<T: Float>(out: &[T]) -> bool {
        let mut ok = true;
        for v in out {
            ok &= v.is_finite();
        }
        ok
    }

    #[derive(Clone, Copy)]
    enum ArgView<'a, T> {
        Slice(&'a [T]),
        Const(T),
    }

    impl<'a, T: Float> From<SrcRef<'a, T>> for ArgView<'a, T> {
        fn from(value: SrcRef<'a, T>) -> Self {
            match value {
                SrcRef::Slice(s) => Self::Slice(s),
                SrcRef::Const(c) => Self::Const(c),
            }
        }
    }

    impl<'a, T: Float> ArgView<'a, T> {
        #[inline]
        fn get(&self, row: usize) -> T {
            match self {
                Self::Slice(s) => s[row],
                Self::Const(c) => *c,
            }
        }
    }

    #[inline]
    fn vals2<T: Float, const A: usize>(a: T, b: T) -> [T; A] {
        let mut vals = [T::zero(); A];
        vals[0] = a;
        vals[1] = b;
        vals
    }

    #[inline]
    fn vals1<T: Float, const A: usize>(a: T) -> [T; A] {
        let mut vals = [T::zero(); A];
        vals[0] = a;
        vals
    }

    #[inline]
    fn grad_dir_view<'a, T: Float>(g: GradRef<'a, T>, dir: usize, n_rows: usize) -> ArgView<'a, T> {
        match g {
            GradRef::Slice(s) => ArgView::Slice(&s[dir * n_rows..(dir + 1) * n_rows]),
            GradRef::Basis(k) => ArgView::Const(if dir == k { T::one() } else { T::zero() }),
            GradRef::Zero => ArgView::Const(T::zero()),
        }
    }

    #[inline]
    fn grad_unary_loop<T: Float, F, const A: usize>(out: &mut [T], x: ArgView<'_, T>, dx: ArgView<'_, T>, partial: F)
    where
        F: Fn(&[T; A], usize) -> T,
    {
        match (x, dx) {
            (_, ArgView::Const(dx_c)) if dx_c.is_zero() => out.fill(T::zero()),
            (ArgView::Slice(x_s), ArgView::Slice(dx_s)) => {
                for ((outg, &xv), &dxv) in out.iter_mut().zip_eq(x_s).zip_eq(dx_s) {
                    let vals = vals1(xv);
                    *outg = partial(&vals, 0) * dxv;
                }
            }
            (ArgView::Slice(x_s), ArgView::Const(dx_c)) => {
                for (outg, &xv) in out.iter_mut().zip_eq(x_s) {
                    let vals = vals1(xv);
                    *outg = partial(&vals, 0) * dx_c;
                }
            }
            (ArgView::Const(x_c), ArgView::Const(dx_c)) => {
                let vals = vals1(x_c);
                let p = partial(&vals, 0);
                out.fill(p * dx_c);
            }
            _ => unreachable!("malformed expression"),
        }
    }

    #[inline]
    fn grad_binary_loop<T: Float, F, const A: usize>(
        out: &mut [T],
        x: ArgView<'_, T>,
        y: ArgView<'_, T>,
        dx: ArgView<'_, T>,
        dy: ArgView<'_, T>,
        partial: F,
    ) where
        F: Fn(&[T; A], usize) -> T,
    {
        match (x, y, dx, dy) {
            (ArgView::Slice(x_s), ArgView::Slice(y_s), ArgView::Slice(dx_s), ArgView::Slice(dy_s)) => {
                let data = (x_s.iter().zip_eq(y_s)).zip_eq(dx_s.iter().zip_eq(dy_s));
                for (outv, ((&xv, &yv), (&dxv, &dyv))) in out.iter_mut().zip_eq(data) {
                    let vals = vals2(xv, yv);
                    *outv = partial(&vals, 0) * dxv + partial(&vals, 1) * dyv;
                }
            }
            (ArgView::Slice(x_s), ArgView::Slice(y_s), ArgView::Slice(dx_s), ArgView::Const(dy_c)) => {
                let data = x_s.iter().zip_eq(y_s).zip_eq(dx_s);
                for (outv, ((&xv, &yv), &dxv)) in out.iter_mut().zip_eq(data) {
                    let vals = vals2(xv, yv);
                    *outv = partial(&vals, 0) * dxv + partial(&vals, 1) * dy_c;
                }
            }
            (ArgView::Slice(x_s), ArgView::Slice(y_s), ArgView::Const(dx_c), ArgView::Slice(dy_s)) => {
                let data = x_s.iter().zip_eq(y_s).zip_eq(dy_s);
                for (outv, ((&xv, &yv), &dyv)) in out.iter_mut().zip_eq(data) {
                    let vals = vals2(xv, yv);
                    *outv = partial(&vals, 0) * dx_c + partial(&vals, 1) * dyv;
                }
            }
            (ArgView::Slice(x_s), ArgView::Slice(y_s), ArgView::Const(dx_c), ArgView::Const(dy_c)) => {
                for (outv, (&xv, &yv)) in out.iter_mut().zip_eq(x_s.iter().zip_eq(y_s)) {
                    let vals = vals2(xv, yv);
                    *outv = partial(&vals, 0) * dx_c + partial(&vals, 1) * dy_c;
                }
            }

            (ArgView::Slice(x_s), ArgView::Const(y_c), ArgView::Slice(dx_s), ArgView::Const(dy_c)) => {
                for (outv, (&xv, &dxv)) in out.iter_mut().zip_eq(x_s.iter().zip_eq(dx_s)) {
                    let vals = vals2(xv, y_c);
                    *outv = partial(&vals, 0) * dxv + partial(&vals, 1) * dy_c;
                }
            }
            (ArgView::Slice(x_s), ArgView::Const(y_c), ArgView::Const(dx_c), ArgView::Const(dy_c)) => {
                for (outv, &xv) in out.iter_mut().zip_eq(x_s) {
                    let vals = vals2(xv, y_c);
                    *outv = partial(&vals, 0) * dx_c + partial(&vals, 1) * dy_c;
                }
            }

            (ArgView::Const(x_c), ArgView::Slice(y_s), ArgView::Const(dx_c), ArgView::Slice(dy_s)) => {
                for (outv, (&yv, &dyv)) in out.iter_mut().zip_eq(y_s.iter().zip_eq(dy_s)) {
                    let vals = vals2(x_c, yv);
                    *outv = partial(&vals, 0) * dx_c + partial(&vals, 1) * dyv;
                }
            }
            (ArgView::Const(x_c), ArgView::Slice(y_s), ArgView::Const(dx_c), ArgView::Const(dy_c)) => {
                for (outv, &yv) in out.iter_mut().zip_eq(y_s) {
                    let vals = vals2(x_c, yv);
                    *outv = partial(&vals, 0) * dx_c + partial(&vals, 1) * dy_c;
                }
            }
            (ArgView::Const(x_c), ArgView::Const(y_c), ArgView::Const(dx_c), ArgView::Const(dy_c)) => {
                let vals = vals2(x_c, y_c);
                out.fill(partial(&vals, 0) * dx_c + partial(&vals, 1) * dy_c);
            }
            _ => unreachable!("malformed expression"),
        }
    }

    #[inline]
    fn make_arg_views<'a, const A: usize, T: Float>(args: &'a [SrcRef<'a, T>]) -> [ArgView<'a, T>; A] {
        core::array::from_fn(|j| args[j].into())
    }

    #[inline]
    fn eval_unary_loop<T: Float>(out: &mut [T], x: ArgView<'_, T>, eval: impl Copy + Fn(T) -> T) {
        match x {
            ArgView::Slice(x_s) => {
                for (outv, &xv) in out.iter_mut().zip_eq(x_s) {
                    *outv = eval(xv);
                }
            }
            ArgView::Const(x_c) => out.fill(eval(x_c)),
        }
    }

    #[inline]
    fn eval_binary_loop<T: Float>(
        out: &mut [T],
        x: ArgView<'_, T>,
        y: ArgView<'_, T>,
        eval: impl Copy + Fn(T, T) -> T,
    ) {
        match (x, y) {
            (ArgView::Slice(x_s), ArgView::Slice(y_s)) => {
                for (outv, (&xv, &yv)) in out.iter_mut().zip_eq(x_s.iter().zip_eq(y_s)) {
                    *outv = eval(xv, yv);
                }
            }
            (ArgView::Slice(x_s), ArgView::Const(y_c)) => {
                for (outv, &xv) in out.iter_mut().zip_eq(x_s) {
                    *outv = eval(xv, y_c);
                }
            }
            (ArgView::Const(x_c), ArgView::Slice(y_s)) => {
                for (outv, &yv) in out.iter_mut().zip_eq(y_s) {
                    *outv = eval(x_c, yv);
                }
            }
            (ArgView::Const(x_c), ArgView::Const(y_c)) => out.fill(eval(x_c, y_c)),
        }
    }

    pub fn eval_nary<const A: usize, T: Float, Op: crate::traits::Operator<T, A>>(
        out: &mut [T],
        args: &[SrcRef<'_, T>],
        opts: &EvalOptions,
    ) -> bool {
        debug_assert_eq!(args.len(), A);
        let check_finite = opts.check_finite;
        let early_exit = opts.early_exit;

        if args.iter().all(|a| matches!(a, SrcRef::Const(_))) {
            let vals: [T; A] = core::array::from_fn(|j| __src_val(args[j], 0));
            let v = Op::eval(&vals);
            out.fill(v);
            if !check_finite {
                return true;
            }
            let finite = v.is_finite();
            if !finite && early_exit {
                out.fill(T::nan());
            }
            return finite;
        }

        let mut complete = true;
        let arg_views: [ArgView<'_, T>; A] = make_arg_views(args);

        if A == 1 {
            eval_unary_loop(out, arg_views[0], move |a| Op::eval(&vals1(a)));
        } else if A == 2 {
            eval_binary_loop(out, arg_views[0], arg_views[1], move |a, b| Op::eval(&vals2(a, b)));
        } else {
            let mut vals: [T; A] = [T::zero(); A];
            for (row, outv) in out.iter_mut().enumerate() {
                for (v, view) in vals.iter_mut().zip_eq(arg_views) {
                    *v = view.get(row);
                }
                let v = Op::eval(&vals);
                *outv = v;
            }
        }

        if !check_finite {
            return true;
        }

        let finite = __all_finite(out);
        complete &= finite;
        if !finite && early_exit {
            out.fill(T::nan());
            return false;
        }

        complete
    }

    pub fn diff_nary<const A: usize, T: Float + core::ops::AddAssign, Op: crate::traits::Operator<T, A>>(
        out_val: &mut [T],
        out_der: &mut [T],
        args: &[SrcRef<'_, T>],
        dargs: &[SrcRef<'_, T>],
        opts: &EvalOptions,
    ) -> bool {
        debug_assert_eq!(args.len(), A);
        debug_assert_eq!(dargs.len(), A);
        let check_finite = opts.check_finite;
        let early_exit = opts.early_exit;
        let mut complete = true;

        let arg_views: [ArgView<'_, T>; A] = make_arg_views(args);
        let darg_views: [ArgView<'_, T>; A] = make_arg_views(dargs);

        if A == 1 {
            eval_unary_loop(out_val, arg_views[0], move |a| Op::eval(&vals1(a)));
            grad_unary_loop::<T, _, A>(out_der, arg_views[0], darg_views[0], Op::partial);
        } else if A == 2 {
            eval_binary_loop(out_val, arg_views[0], arg_views[1], move |a, b| Op::eval(&vals2(a, b)));
            grad_binary_loop::<T, _, A>(
                out_der,
                arg_views[0],
                arg_views[1],
                darg_views[0],
                darg_views[1],
                Op::partial,
            );
        } else {
            let mut vals: [T; A] = [T::zero(); A];
            let mut dvals: [T; A] = [T::zero(); A];

            for (row, (outv, outd)) in out_val.iter_mut().zip_eq(out_der.iter_mut()).enumerate() {
                for (v, view) in vals.iter_mut().zip_eq(arg_views) {
                    *v = view.get(row);
                }
                for (dv, view) in dvals.iter_mut().zip_eq(darg_views) {
                    *dv = view.get(row);
                }
                let v = Op::eval(&vals);
                *outv = v;

                let mut dv_out = T::zero();
                for (j, dvj) in dvals.iter().copied().enumerate() {
                    dv_out += Op::partial(&vals, j) * dvj;
                }
                *outd = dv_out;
            }
        }

        if check_finite {
            let finite_val = __all_finite(out_val);
            let finite = if A > 2 {
                finite_val && __all_finite(out_der)
            } else {
                finite_val
            };
            complete &= finite;
            if !finite && early_exit {
                out_val.fill(T::nan());
                out_der.fill(T::nan());
                return false;
            }
        }

        complete
    }

    pub fn grad_nary<const A: usize, T: Float + core::ops::AddAssign, Op: crate::traits::Operator<T, A>>(
        ctx: GradKernelCtx<'_, '_, T>,
    ) -> bool {
        debug_assert_eq!(ctx.args.len(), A);
        debug_assert_eq!(ctx.arg_grads.len(), A);

        let check_finite = ctx.opts.check_finite;
        let early_exit = ctx.opts.early_exit;
        let mut complete = true;
        let arg_views: [ArgView<'_, T>; A] = make_arg_views(ctx.args);

        if A == 1 {
            eval_unary_loop(ctx.out_val, arg_views[0], move |a| Op::eval(&vals1(a)));
            let x = arg_views[0];
            let dx_ref = ctx.arg_grads[0];
            let n_rows = ctx.n_rows;
            for dir in 0..ctx.n_dir {
                let grad_dir = &mut ctx.out_grad[dir * n_rows..(dir + 1) * n_rows];
                let dx = grad_dir_view(dx_ref, dir, n_rows);
                grad_unary_loop::<T, _, A>(grad_dir, x, dx, Op::partial);
            }
        } else if A == 2 {
            eval_binary_loop(ctx.out_val, arg_views[0], arg_views[1], move |a, b| {
                Op::eval(&vals2(a, b))
            });

            let x = arg_views[0];
            let y = arg_views[1];
            let dx_ref = ctx.arg_grads[0];
            let dy_ref = ctx.arg_grads[1];
            let n_rows = ctx.n_rows;

            for dir in 0..ctx.n_dir {
                let grad_dir = &mut ctx.out_grad[dir * n_rows..(dir + 1) * n_rows];
                let dx = grad_dir_view(dx_ref, dir, n_rows);
                let dy = grad_dir_view(dy_ref, dir, n_rows);
                if matches!((dx, dy), (ArgView::Const(dx_c), ArgView::Const(dy_c)) if dx_c.is_zero() && dy_c.is_zero())
                {
                    grad_dir.fill(T::zero());
                    continue;
                }
                grad_binary_loop::<T, _, A>(grad_dir, x, y, dx, dy, Op::partial);
            }
        } else {
            let mut vals: [T; A] = [T::zero(); A];

            for (row, outv) in ctx.out_val.iter_mut().enumerate() {
                for (v, view) in vals.iter_mut().zip_eq(arg_views) {
                    *v = view.get(row);
                }
                *outv = Op::eval(&vals);
            }

            for (dir, grad_dir) in ctx.out_grad.chunks_mut(ctx.n_rows).enumerate().take(ctx.n_dir) {
                for (row, outg) in grad_dir.iter_mut().enumerate() {
                    for (v, view) in vals.iter_mut().zip_eq(arg_views) {
                        *v = view.get(row);
                    }
                    let mut g = T::zero();
                    for (j, ag) in ctx.arg_grads.iter().copied().enumerate() {
                        g += Op::partial(&vals, j) * grad_at(ag, dir, row, ctx.n_rows);
                    }
                    *outg = g;
                }
            }
        }

        if check_finite {
            let finite = __all_finite(ctx.out_val);
            complete &= finite;
            if !finite && early_exit {
                ctx.out_val.fill(T::nan());
                ctx.out_grad.fill(T::nan());
                return false;
            }
        }

        complete
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::dispatch::SrcRef;

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
