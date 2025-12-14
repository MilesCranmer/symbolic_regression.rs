use crate::eval::EvalOptions;
use crate::operators::builtin::BuiltinOp;
use num_traits::Float;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct OpId {
    pub arity: u8,
    pub id: u16,
}

pub trait HasOp<Tag, const A: usize> {
    const ID: u16;
}

#[derive(Copy, Clone, Debug)]
pub enum SrcRef<'a, T> {
    Slice(&'a [T]),
    /// Strided view into a flat row-major matrix `data[row*stride + offset]`.
    Strided {
        data: &'a [T],
        offset: usize,
        stride: usize,
    },
    Const(T),
}

#[derive(Copy, Clone, Debug)]
pub enum GradRef<'a, T> {
    /// Dir-major gradient slab: `grad[dir*n_rows + row]`.
    Slice(&'a [T]),
    /// One-hot basis direction (value is 1 if `dir == basis_dir` else 0).
    Basis(usize),
    /// All zeros.
    Zero,
}

pub fn grad_at<T: Float>(g: GradRef<'_, T>, dir: usize, row: usize, n_rows: usize) -> T {
    match g {
        GradRef::Slice(s) => s[dir * n_rows + row],
        GradRef::Basis(k) => {
            if dir == k {
                T::one()
            } else {
                T::zero()
            }
        }
        GradRef::Zero => T::zero(),
    }
}

pub struct EvalKernelCtx<'a, 'b, T> {
    pub out: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub opts: &'b EvalOptions,
}

pub struct DiffKernelCtx<'a, 'b, T> {
    pub out_val: &'b mut [T],
    pub out_der: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub dargs: &'b [SrcRef<'a, T>],
    pub opts: &'b EvalOptions,
}

pub struct GradKernelCtx<'a, 'b, T> {
    pub out_val: &'b mut [T],
    /// Dir-major buffer: `out_grad[dir*n_rows + row]`.
    pub out_grad: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub arg_grads: &'b [GradRef<'a, T>],
    pub n_dir: usize,
    pub n_rows: usize,
    pub opts: &'b EvalOptions,
}

pub trait ScalarOpSet<T: Float> {
    fn eval(op: OpId, ctx: EvalKernelCtx<'_, '_, T>) -> bool;
    fn diff(op: OpId, ctx: DiffKernelCtx<'_, '_, T>) -> bool;
    fn grad(op: OpId, ctx: GradKernelCtx<'_, '_, T>) -> bool;
}

#[doc(hidden)]
pub fn __src_val<T: Float>(src: SrcRef<'_, T>, row: usize) -> T {
    match src {
        SrcRef::Slice(s) => s[row],
        SrcRef::Strided {
            data,
            offset,
            stride,
        } => data[row * stride + offset],
        SrcRef::Const(c) => c,
    }
}

#[doc(hidden)]
pub fn __maybe_mark_nonfinite<T: Float>(v: T, opts: &EvalOptions, complete: &mut bool) -> bool {
    if opts.check_finite && !v.is_finite() {
        *complete = false;
        if opts.early_exit {
            return false;
        }
    }
    true
}

fn __all_finite<T: Float>(vals: &[T]) -> bool {
    vals.iter().all(|v| v.is_finite())
}

pub fn eval_nary<const A: usize, T: Float>(
    eval: fn(&[T; A]) -> T,
    out: &mut [T],
    args: &[SrcRef<'_, T>],
    opts: &EvalOptions,
) -> bool {
    debug_assert_eq!(args.len(), A);
    let check_finite = opts.check_finite;
    let early_exit = opts.early_exit;
    let mut complete = true;

    if args.iter().all(|a| matches!(a, SrcRef::Const(_))) {
        let vals: [T; A] = core::array::from_fn(|j| __src_val(args[j], 0));
        let v = eval(&vals);
        out.fill(v);
        if !check_finite {
            return true;
        }
        return v.is_finite();
    }

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    if check_finite && early_exit {
        for (row, outv) in out.iter_mut().enumerate() {
            for (j, v) in vals.iter_mut().enumerate() {
                *v = __src_val(args[j], row);
            }
            let v = eval(&vals);
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                return false;
            }
            *outv = v;
        }
        return complete;
    }

    for (row, outv) in out.iter_mut().enumerate() {
        for (j, v) in vals.iter_mut().enumerate() {
            *v = __src_val(args[j], row);
        }
        let v = eval(&vals);
        *outv = v;
    }
    if !check_finite {
        return true;
    }
    __all_finite(out)
}

pub fn eval_apply<const A: usize, T: Float, Op: BuiltinOp<T, A>>(
    out: &mut [T],
    args: &[SrcRef<'_, T>],
    opts: &EvalOptions,
) -> bool {
    debug_assert_eq!(args.len(), A);
    let check_finite = opts.check_finite;
    let early_exit = opts.early_exit;
    let mut complete = true;

    if args.iter().all(|a| matches!(a, SrcRef::Const(_))) {
        let vals: [T; A] = core::array::from_fn(|j| __src_val(args[j], 0));
        let v = Op::eval(&vals);
        out.fill(v);
        if !check_finite {
            return true;
        }
        if !v.is_finite() {
            complete = false;
        }
        return complete;
    }

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    if check_finite && early_exit {
        for (row, outv) in out.iter_mut().enumerate() {
            for (j, v) in vals.iter_mut().enumerate() {
                *v = __src_val(args[j], row);
            }
            let v = Op::eval(&vals);
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                return false;
            }
            *outv = v;
        }
        return complete;
    }

    for (row, outv) in out.iter_mut().enumerate() {
        for (j, v) in vals.iter_mut().enumerate() {
            *v = __src_val(args[j], row);
        }
        let v = Op::eval(&vals);
        *outv = v;
    }
    if !check_finite {
        return true;
    }
    __all_finite(out)
}

pub fn diff_nary<const A: usize, T: Float + core::ops::AddAssign>(
    eval: fn(&[T; A]) -> T,
    partial: fn(&[T; A], usize) -> T,
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

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    let mut dvals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
            for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
                *v = __src_val(src, row);
            }
            for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
                *dv = __src_val(dsrc, row);
            }
            let v = eval(&vals);
            let mut d = T::zero();
            for (j, dv) in dvals.iter().enumerate() {
                d += partial(&vals, j) * *dv;
            }
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                *outd = d;
                return false;
            }
            *outv = v;
            *outd = d;
        }
        return complete;
    }

    for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
        for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
            *v = __src_val(src, row);
        }
        for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
            *dv = __src_val(dsrc, row);
        }
        let v = eval(&vals);
        let mut d = T::zero();
        for (j, dv) in dvals.iter().enumerate() {
            d += partial(&vals, j) * *dv;
        }
        *outv = v;
        *outd = d;
    }

    if !check_finite {
        return true;
    }
    __all_finite(out_val)
}

pub fn diff_apply<const A: usize, T: Float + core::ops::AddAssign, Op: BuiltinOp<T, A>>(
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

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    let mut dvals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
            for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
                *v = __src_val(src, row);
            }
            for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
                *dv = __src_val(dsrc, row);
            }
            let v = Op::eval(&vals);
            let mut d = T::zero();
            for (j, dv) in dvals.iter().enumerate() {
                d += Op::partial(&vals, j) * *dv;
            }
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                *outd = d;
                return false;
            }
            *outv = v;
            *outd = d;
        }
        return complete;
    }

    for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
        for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
            *v = __src_val(src, row);
        }
        for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
            *dv = __src_val(dsrc, row);
        }
        let v = Op::eval(&vals);
        let mut d = T::zero();
        for (j, dv) in dvals.iter().enumerate() {
            d += Op::partial(&vals, j) * *dv;
        }
        *outv = v;
        *outd = d;
    }

    if !check_finite {
        return true;
    }
    __all_finite(out_val)
}

pub fn grad_nary<const A: usize, T: Float + core::ops::AddAssign>(
    eval: fn(&[T; A]) -> T,
    partial: fn(&[T; A], usize) -> T,
    ctx: GradKernelCtx<'_, '_, T>,
) -> bool {
    debug_assert_eq!(ctx.args.len(), A);
    debug_assert_eq!(ctx.arg_grads.len(), A);

    let check_finite = ctx.opts.check_finite;
    let early_exit = ctx.opts.early_exit;
    let mut complete = true;
    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = eval(&vals);
            if !__maybe_mark_nonfinite(v, ctx.opts, &mut complete) {
                *outv = v;
                ctx.out_grad.fill(T::nan());
                return false;
            }
            *outv = v;
        }
    } else {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = eval(&vals);
            *outv = v;
        }
    }

    for (dir, grad_dir) in ctx
        .out_grad
        .chunks_mut(ctx.n_rows)
        .enumerate()
        .take(ctx.n_dir)
    {
        for (row, outg) in grad_dir.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let mut g = T::zero();
            for (j, ag) in ctx.arg_grads.iter().copied().enumerate() {
                g += partial(&vals, j) * grad_at(ag, dir, row, ctx.n_rows);
            }
            *outg = g;
        }
    }

    if !check_finite {
        return true;
    }
    if early_exit {
        return complete;
    }
    __all_finite(ctx.out_val)
}

pub fn grad_apply<const A: usize, T: Float + core::ops::AddAssign, Op: BuiltinOp<T, A>>(
    ctx: GradKernelCtx<'_, '_, T>,
) -> bool {
    debug_assert_eq!(ctx.args.len(), A);
    debug_assert_eq!(ctx.arg_grads.len(), A);

    let check_finite = ctx.opts.check_finite;
    let early_exit = ctx.opts.early_exit;
    let mut complete = true;
    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = Op::eval(&vals);
            if !__maybe_mark_nonfinite(v, ctx.opts, &mut complete) {
                *outv = v;
                ctx.out_grad.fill(T::nan());
                return false;
            }
            *outv = v;
        }
    } else {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = Op::eval(&vals);
            *outv = v;
        }
    }

    for (dir, grad_dir) in ctx
        .out_grad
        .chunks_mut(ctx.n_rows)
        .enumerate()
        .take(ctx.n_dir)
    {
        for (row, outg) in grad_dir.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let mut g = T::zero();
            for (j, ag) in ctx.arg_grads.iter().copied().enumerate() {
                g += Op::partial(&vals, j) * grad_at(ag, dir, row, ctx.n_rows);
            }
            *outg = g;
        }
    }

    if !check_finite {
        return true;
    }
    if early_exit {
        return complete;
    }
    __all_finite(ctx.out_val)
}

#[macro_export]
#[doc(hidden)]
macro_rules! __default_op_str {
    (Add) => {
        "+"
    };
    (Sub) => {
        "-"
    };
    (Mul) => {
        "*"
    };
    (Div) => {
        "/"
    };
    (Neg) => {
        "-"
    };
    ($other:ident) => {
        $crate::paste::paste! { stringify!([<$other:snake>]) }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __maybe_impl_role {
    ($Ops:ident, $arity:literal, $enum_name:ident, Add) => {
        impl $crate::algebra::HasAdd for $Ops {
            const ADD: $crate::operators::scalar::OpId = $crate::operators::scalar::OpId {
                arity: $arity as u8,
                id: $enum_name::Add as u16,
            };
        }
    };
    ($Ops:ident, $arity:literal, $enum_name:ident, Sub) => {
        impl $crate::algebra::HasSub for $Ops {
            const SUB: $crate::operators::scalar::OpId = $crate::operators::scalar::OpId {
                arity: $arity as u8,
                id: $enum_name::Sub as u16,
            };
        }
    };
    ($Ops:ident, $arity:literal, $enum_name:ident, Mul) => {
        impl $crate::algebra::HasMul for $Ops {
            const MUL: $crate::operators::scalar::OpId = $crate::operators::scalar::OpId {
                arity: $arity as u8,
                id: $enum_name::Mul as u16,
            };
        }
    };
    ($Ops:ident, $arity:literal, $enum_name:ident, Div) => {
        impl $crate::algebra::HasDiv for $Ops {
            const DIV: $crate::operators::scalar::OpId = $crate::operators::scalar::OpId {
                arity: $arity as u8,
                id: $enum_name::Div as u16,
            };
        }
    };
    ($Ops:ident, $arity:literal, $enum_name:ident, Neg) => {
        impl $crate::algebra::HasNeg for $Ops {
            const NEG: $crate::operators::scalar::OpId = $crate::operators::scalar::OpId {
                arity: $arity as u8,
                id: $enum_name::Neg as u16,
            };
        }
    };
    ($Ops:ident, $arity:literal, $enum_name:ident, $other:ident) => {};
}

#[macro_export]
#[doc(hidden)]
macro_rules! define_scalar_ops {
    (
        $vis:vis struct $Ops:ident<$t:ty>;
        ops {
            $(($arity:literal, $enum_name:ident) {
                $($op_name:ident => ($op_eval:path, $op_partial:path),)*
            })*
        }
    ) => {
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::operators::scalar::ScalarOpSet<$t> for $Ops {
            fn eval(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::EvalKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::eval_nary::<$arity, $t>($op_eval, ctx.out, ctx.args, ctx.opts),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn diff(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::DiffKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::diff_nary::<$arity, $t>(
                                        $op_eval,
                                        $op_partial,
                                        ctx.out_val,
                                        ctx.out_der,
                                        ctx.args,
                                        ctx.dargs,
                                        ctx.opts,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn grad(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::GradKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::grad_nary::<$arity, $t>(
                                        $op_eval,
                                        $op_partial,
                                        ctx,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }
        }

        $(
            $(
                $crate::__maybe_impl_role!($Ops, $arity, $enum_name, $op_name);
            )*
        )*

        impl $crate::strings::OpNames for $Ops {
            fn op_name(op: $crate::operators::scalar::OpId) -> &'static str {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) => $crate::__default_op_str!($op_name),
                            )*
                            _ => "unknown_op",
                        },
                    )*
                    _ => "unknown_op",
                }
            }
        }
    };
}

#[macro_export]
macro_rules! opset {
    (
        $vis:vis struct $Ops:ident<$t:ty>;
        ops {
            $(($arity:literal, $enum_name:ident) { $($op_name:ident,)* })*
        }
    ) => {
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::operators::scalar::ScalarOpSet<$t> for $Ops {
            fn eval(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::EvalKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::eval_apply::<$arity, $t, $crate::operators::builtin::$op_name>(
                                        ctx.out, ctx.args, ctx.opts,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn diff(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::DiffKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::diff_apply::<$arity, $t, $crate::operators::builtin::$op_name>(
                                        ctx.out_val, ctx.out_der, ctx.args, ctx.dargs, ctx.opts,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn grad(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::GradKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::grad_apply::<$arity, $t, $crate::operators::builtin::$op_name>(ctx),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }
        }

        $(
            $(
                impl $crate::operators::scalar::HasOp<$crate::operators::builtin::$op_name, $arity> for $Ops {
                    const ID: u16 = $enum_name::$op_name as u16;
                }

                $crate::__maybe_impl_role!($Ops, $arity, $enum_name, $op_name);
            )*
        )*

        impl $crate::strings::OpNames for $Ops {
            fn op_name(op: $crate::operators::scalar::OpId) -> &'static str {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) => {
                                    <$crate::operators::builtin::$op_name as $crate::operators::builtin::BuiltinOp<$t, $arity>>::DISPLAY
                                }
                            )*
                            _ => "unknown_op",
                        },
                    )*
                    _ => "unknown_op",
                }
            }
        }
    };
}
