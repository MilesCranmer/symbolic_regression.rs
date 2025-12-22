pub mod builtin {
    use num_traits::Float;

    pub trait BuiltinOp<T: Float, const A: usize> {
        const NAME: &'static str;
        const INFIX: Option<&'static str> = None;
        const DISPLAY: &'static str = Self::NAME;

        fn eval(args: &[T; A]) -> T;
        fn partial(args: &[T; A], idx: usize) -> T;
    }

    pub trait OpMeta<const A: usize> {
        const COMMUTATIVE: bool = false;
        const ASSOCIATIVE: bool = false;
        const COMPLEXITY: u16 = 1;
    }

    fn two<T: Float>() -> T {
        T::one() + T::one()
    }

    macro_rules! builtin_op {
    (@name $Op:ident, $name:literal) => {
        $name
    };
    (@name $Op:ident) => {
        crate::paste::paste! { stringify!([<$Op:snake>]) }
    };
    (@infix $v:expr) => {
        $v
    };
    (@infix) => {
        None
    };
    (@commutative $v:expr) => {
        $v
    };
    (@commutative) => {
        false
    };
    (@associative $v:expr) => {
        $v
    };
    (@associative) => {
        false
    };
    (@complexity $v:expr) => {
        $v
    };
    (@complexity) => {
        1u16
    };
    (
        $(#[$meta:meta])*
        $Op:ident : $A:literal {
            $(name: $name:literal,)?
            $(infix: $infix:expr,)?
            $(commutative: $commutative:expr,)?
            $(associative: $associative:expr,)?
            $(complexity: $complexity:expr,)?
            eval($args:ident) $eval:block,
            partial($pargs:ident, $idx:ident) $partial:block $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $Op;

        impl OpMeta<$A> for $Op {
            const COMMUTATIVE: bool = builtin_op!(@commutative $($commutative)?);
            const ASSOCIATIVE: bool = builtin_op!(@associative $($associative)?);
            const COMPLEXITY: u16 = builtin_op!(@complexity $($complexity)?);
        }

        impl<T: Float> BuiltinOp<T, $A> for $Op {
            const NAME: &'static str = builtin_op!(@name $Op $(, $name)?);
            const INFIX: Option<&'static str> = builtin_op!(@infix $($infix)?);
            const DISPLAY: &'static str = match builtin_op!(@infix $($infix)?) {
                Some(s) => s,
                None => builtin_op!(@name $Op $(, $name)?),
            };

            fn eval(args: &[T; $A]) -> T {
                let $args = args;
                $eval
            }

            fn partial(args: &[T; $A], idx: usize) -> T {
                let $pargs = args;
                let $idx = idx;
                $partial
            }
        }
    };
}

    builtin_op!(Cos: 1 {
        eval(args) { args[0].cos() },
        partial(args, idx) {
            match idx {
                0 => -args[0].sin(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Sin: 1 {
        eval(args) { args[0].sin() },
        partial(args, idx) {
            match idx {
                0 => args[0].cos(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Tan: 1 {
        eval(args) { args[0].tan() },
        partial(args, idx) {
            match idx {
                0 => {
                    let c = args[0].cos();
                    T::one() / (c * c)
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Asin: 1 {
        eval(args) { args[0].asin() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (T::one() - args[0] * args[0]).sqrt(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Acos: 1 {
        eval(args) { args[0].acos() },
        partial(args, idx) {
            match idx {
                0 => -T::one() / (T::one() - args[0] * args[0]).sqrt(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Atan: 1 {
        eval(args) { args[0].atan() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (T::one() + args[0] * args[0]),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Sinh: 1 {
        eval(args) { args[0].sinh() },
        partial(args, idx) {
            match idx {
                0 => args[0].cosh(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Cosh: 1 {
        eval(args) { args[0].cosh() },
        partial(args, idx) {
            match idx {
                0 => args[0].sinh(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Tanh: 1 {
        eval(args) { args[0].tanh() },
        partial(args, idx) {
            match idx {
                0 => {
                    let c = args[0].cosh();
                    T::one() / (c * c)
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Asinh: 1 {
        eval(args) { args[0].asinh() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (args[0] * args[0] + T::one()).sqrt(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Acosh: 1 {
        eval(args) { args[0].acosh() },
        partial(args, idx) {
            match idx {
                0 => {
                    let x = args[0];
                    T::one() / ((x - T::one()).sqrt() * (x + T::one()).sqrt())
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Atanh: 1 {
        eval(args) { args[0].atanh() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (T::one() - args[0] * args[0]),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Sec: 1 {
        eval(args) { T::one() / args[0].cos() },
        partial(args, idx) {
            match idx {
                0 => {
                    let sec = T::one() / args[0].cos();
                    sec * args[0].tan()
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Csc: 1 {
        eval(args) { T::one() / args[0].sin() },
        partial(args, idx) {
            match idx {
                0 => {
                    let csc = T::one() / args[0].sin();
                    let cot = T::one() / args[0].tan();
                    -csc * cot
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Cot: 1 {
        eval(args) { T::one() / args[0].tan() },
        partial(args, idx) {
            match idx {
                0 => {
                    let s = args[0].sin();
                    -T::one() / (s * s)
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Exp: 1 {
        eval(args) { args[0].exp() },
        partial(args, idx) {
            match idx {
                0 => args[0].exp(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Log: 1 {
        eval(args) { args[0].ln() },
        partial(args, idx) {
            match idx {
                0 => T::one() / args[0],
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Log2: 1 {
        eval(args) { args[0].log2() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (args[0] * two::<T>().ln()),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Log10: 1 {
        eval(args) { args[0].log10() },
        partial(args, idx) {
            match idx {
                0 => {
                    let ten = T::from(10.0).expect("Float can represent 10.0");
                    T::one() / (args[0] * ten.ln())
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Log1p: 1 {
        eval(args) { args[0].ln_1p() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (T::one() + args[0]),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Exp2: 1 {
        eval(args) { args[0].exp2() },
        partial(args, idx) {
            match idx {
                0 => args[0].exp2() * two::<T>().ln(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Expm1: 1 {
        eval(args) { args[0].exp_m1() },
        partial(args, idx) {
            match idx {
                0 => args[0].exp(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Sqrt: 1 {
        eval(args) { args[0].sqrt() },
        partial(args, idx) {
            match idx {
                0 => T::one() / (two::<T>() * args[0].sqrt()),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Cbrt: 1 {
        eval(args) { args[0].cbrt() },
        partial(args, idx) {
            match idx {
                0 => {
                    let three = T::from(3.0).expect("Float can represent 3.0");
                    T::one() / (three * args[0].cbrt().powi(2))
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Abs: 1 {
        eval(args) { args[0].abs() },
        partial(args, idx) {
            match idx {
                0 => {
                    let x = args[0];
                    if x > T::zero() {
                        T::one()
                    } else if x < T::zero() {
                        -T::one()
                    } else {
                        T::zero()
                    }
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Abs2: 1 {
        eval(args) { args[0] * args[0] },
        partial(args, idx) {
            match idx {
                0 => two::<T>() * args[0],
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Inv: 1 {
        eval(args) { args[0].recip() },
        partial(args, idx) {
            match idx {
                0 => -T::one() / (args[0] * args[0]),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Sign: 1 {
        eval(args) { args[0].signum() },
        partial(_args, idx) {
            match idx {
                0 => T::zero(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Identity: 1 {
        eval(args) { args[0] },
        partial(_args, idx) {
            match idx {
                0 => T::one(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Neg: 1 {
        infix: Some("-"),
        eval(args) { -args[0] },
        partial(_args, idx) {
            match idx {
                0 => -T::one(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Add: 2 {
        infix: Some("+"),
        commutative: true,
        associative: true,
        eval(args) { args[0] + args[1] },
        partial(_args, idx) {
            match idx {
                0 | 1 => T::one(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Sub: 2 {
        infix: Some("-"),
        eval(args) { args[0] - args[1] },
        partial(_args, idx) {
            match idx {
                0 => T::one(),
                1 => -T::one(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Mul: 2 {
        infix: Some("*"),
        commutative: true,
        associative: true,
        eval(args) { args[0] * args[1] },
        partial(args, idx) {
            match idx {
                0 => args[1],
                1 => args[0],
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Div: 2 {
        infix: Some("/"),
        eval(args) { args[0] / args[1] },
        partial(args, idx) {
            match idx {
                0 => T::one() / args[1],
                1 => -args[0] / (args[1] * args[1]),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Pow: 2 {
        eval(args) { args[0].powf(args[1]) },
        partial(args, idx) {
            let x = args[0];
            let y = args[1];
            let f = x.powf(y);
            match idx {
                0 => y * x.powf(y - T::one()),
                1 => f * x.ln(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Atan2: 2 {
        eval(args) { args[0].atan2(args[1]) },
        partial(args, idx) {
            let y = args[0];
            let x = args[1];
            let denom = x * x + y * y;
            match idx {
                0 => x / denom,
                1 => -y / denom,
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Min: 2 {
        commutative: true,
        associative: true,
        eval(args) { args[0].min(args[1]) },
        partial(args, idx) {
            let half = T::from(0.5).expect("Float can represent 0.5");
            if args[0].is_nan() && args[1].is_nan() {
                return half;
            }
            if args[0].is_nan() {
                return match idx {
                    0 => T::zero(),
                    1 => T::one(),
                    _ => unreachable!(),
                };
            }
            if args[1].is_nan() {
                return match idx {
                    0 => T::one(),
                    1 => T::zero(),
                    _ => unreachable!(),
                };
            }
            match idx {
                0 => {
                    if args[0] < args[1] {
                        T::one()
                    } else if args[0] > args[1] {
                        T::zero()
                    } else {
                        half
                    }
                }
                1 => {
                    if args[1] < args[0] {
                        T::one()
                    } else if args[1] > args[0] {
                        T::zero()
                    } else {
                        half
                    }
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Max: 2 {
        commutative: true,
        associative: true,
        eval(args) { args[0].max(args[1]) },
        partial(args, idx) {
            let half = T::from(0.5).expect("Float can represent 0.5");
            if args[0].is_nan() && args[1].is_nan() {
                return half;
            }
            if args[0].is_nan() {
                return match idx {
                    0 => T::zero(),
                    1 => T::one(),
                    _ => unreachable!(),
                };
            }
            if args[1].is_nan() {
                return match idx {
                    0 => T::one(),
                    1 => T::zero(),
                    _ => unreachable!(),
                };
            }
            match idx {
                0 => {
                    if args[0] > args[1] {
                        T::one()
                    } else if args[0] < args[1] {
                        T::zero()
                    } else {
                        half
                    }
                }
                1 => {
                    if args[1] > args[0] {
                        T::one()
                    } else if args[1] < args[0] {
                        T::zero()
                    } else {
                        half
                    }
                }
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Fma: 3 {
        eval(args) { args[0].mul_add(args[1], args[2]) },
        partial(args, idx) {
            match idx {
                0 => args[1],
                1 => args[0],
                2 => T::one(),
                _ => unreachable!(),
            }
        },
    });

    builtin_op!(Clamp: 3 {
        eval(args) {
            let x = args[0];
            let lo = args[1];
            let hi = args[2];
            // `Float::clamp` may panic if `lo > hi`, so keep old behavior as fallback.
            if lo <= hi {
                x.clamp(lo, hi)
            } else if x < lo {
                lo
            } else if x > hi {
                hi
            } else {
                x
            }
        },
        partial(args, idx) {
            let x = args[0];
            let lo = args[1];
            let hi = args[2];
            match idx {
                0 => {
                    if x < lo || x > hi { T::zero() } else { T::one() }
                }
                1 => {
                    if x < lo { T::one() } else { T::zero() }
                }
                2 => {
                    if x > hi { T::one() } else { T::zero() }
                }
                _ => unreachable!(),
            }
        },
    });
}

pub mod names {
    use super::scalar::OpId;

    pub trait OpNames {
        fn op_name(op: OpId) -> &'static str;

        fn op_pretty_name(op: OpId) -> &'static str {
            Self::op_name(op)
        }
    }
}

pub mod presets {
    use crate::opset;

    // A convenient, batteries-included opset so downstream crates (like `symbolic_regression`)
    // don't need to define their own `Ops` type for common scalar use cases.

    opset! {
        pub struct BuiltinOpsF32<f32>;
        ops {
            (1, UOpsF32) {
                Abs, Abs2, Acos, Acosh, Asin, Asinh, Atan, Atanh,
                Cbrt, Cos, Cosh, Cot, Csc, Exp, Exp2, Expm1,
                Identity, Inv, Log, Log1p, Log2, Log10,
                Neg, Sec, Sign, Sin, Sinh, Sqrt, Tan, Tanh,
            }
            (2, BOpsF32) { Add, Atan2, Div, Max, Min, Mul, Pow, Sub, }
            (3, TOpsF32) { Clamp, Fma, }
        }
    }

    opset! {
        pub struct BuiltinOpsF64<f64>;
        ops {
            (1, UOpsF64) {
                Abs, Abs2, Acos, Acosh, Asin, Asinh, Atan, Atanh,
                Cbrt, Cos, Cosh, Cot, Csc, Exp, Exp2, Expm1,
                Identity, Inv, Log, Log1p, Log2, Log10,
                Neg, Sec, Sign, Sin, Sinh, Sqrt, Tan, Tanh,
            }
            (2, BOpsF64) { Add, Atan2, Div, Max, Min, Mul, Pow, Sub, }
            (3, TOpsF64) { Clamp, Fma, }
        }
    }
}

pub mod registry {
    pub use crate::operator_registry::{LookupError, OpInfo, OpRegistry};
}

pub mod scalar {
    mod types {
        use crate::evaluate::EvalOptions;
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
        pub fn __maybe_mark_nonfinite<T: Float>(
            v: T,
            opts: &EvalOptions,
            complete: &mut bool,
        ) -> bool {
            if opts.check_finite && !v.is_finite() {
                *complete = false;
                if opts.early_exit {
                    return false;
                }
            }
            true
        }
    }

    mod kernels {
        use crate::evaluate::EvalOptions;
        use crate::operator_enum::builtin::BuiltinOp;
        use num_traits::Float;

        use super::{grad_at, GradKernelCtx, SrcRef, __maybe_mark_nonfinite, __src_val};

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
    }

    mod macros {
        /// Define a minimal custom operator set using inline functions for evaluation and derivatives.
        ///
        /// Example:
        ///
        /// ```
        /// dynamic_expressions::custom_opset! {
        ///     /// Example custom operators.
        ///     struct CustomOps<f64> {
        ///         1 {
        ///             square {
        ///                 eval(args) { args[0] * args[0] },
        ///                 partial(args, idx) {
        ///                     match idx {
        ///                         0 => 2.0 * args[0],
        ///                         _ => unreachable!(),
        ///                     }
        ///                 },
        ///             }
        ///         }
        ///     }
        /// }
        /// ```
        #[macro_export]
        macro_rules! custom_opset {
            (
                $(#[$meta:meta])* $vis:vis struct $Ops:ident<$t:ty> {
                    $( $arity:literal {
                        $( $op_name:ident { $($op_body:tt)* })*
                    })*
                }
            ) => {
                $(#[$meta])*
                #[derive(Copy, Clone, Debug, Default)]
                $vis struct $Ops;

                $crate::paste::paste! {
                    $(
                        #[repr(u16)]
                        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
                        #[allow(non_camel_case_types)]
                        enum [<__ $Ops _op_id_ $arity>] { $( [<$op_name:camel>], )* }
                    )*

                        $(
                            $(
                                #[allow(non_snake_case)]
                                fn [<__ $Ops _ $op_name _eval>](args: &[$t; $arity]) -> $t {
                                    $crate::custom_opset!(@eval_body args { $($op_body)* })
                                }

                                #[allow(non_snake_case)]
                                fn [<__ $Ops _ $op_name _partial>](args: &[$t; $arity], idx: usize) -> $t {
                                    $crate::custom_opset!(@partial_body args, idx { $($op_body)* })
                                }
                            )*
                        )*
                }

                $crate::paste::paste! {
                    impl $crate::operator_enum::scalar::ScalarOpSet<$t> for $Ops {
                        fn eval(
                            op: $crate::operator_enum::scalar::OpId,
                            ctx: $crate::operator_enum::scalar::EvalKernelCtx<'_, '_, $t>,
                        ) -> bool {
                            match op.arity {
                                $(
                                    $arity => match op.id {
                                        $(
                                            x if x == ([<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16) =>
                                                $crate::operator_enum::scalar::eval_nary::<$arity, $t>(
                                                    [<__ $Ops _ $op_name _eval>],
                                                    ctx.out,
                                                    ctx.args,
                                                    ctx.opts,
                                                ),
                                        )*
                                        _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                                    },
                                )*
                                _ => panic!("unsupported arity {}", op.arity),
                            }
                        }

                        fn diff(
                            op: $crate::operator_enum::scalar::OpId,
                            ctx: $crate::operator_enum::scalar::DiffKernelCtx<'_, '_, $t>,
                        ) -> bool {
                            match op.arity {
                                $(
                                    $arity => match op.id {
                                        $(
                                            x if x == ([<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16) =>
                                                $crate::operator_enum::scalar::diff_nary::<$arity, $t>(
                                                    [<__ $Ops _ $op_name _eval>],
                                                    [<__ $Ops _ $op_name _partial>],
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
                            op: $crate::operator_enum::scalar::OpId,
                            ctx: $crate::operator_enum::scalar::GradKernelCtx<'_, '_, $t>,
                        ) -> bool {
                            match op.arity {
                                $(
                                    $arity => match op.id {
                                        $(
                                            x if x == ([<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16) =>
                                                $crate::operator_enum::scalar::grad_nary::<$arity, $t>(
                                                    [<__ $Ops _ $op_name _eval>],
                                                    [<__ $Ops _ $op_name _partial>],
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
                }

                $crate::paste::paste! {
                    impl $crate::operator_registry::OpRegistry for $Ops {
                        fn registry() -> &'static [$crate::operator_registry::OpInfo] {
                            const REGISTRY: &[$crate::operator_registry::OpInfo] = &[
                                $(
                                    $(
                                        $crate::operator_registry::OpInfo {
                                            op: $crate::operator_enum::scalar::OpId {
                                                arity: $arity as u8,
                                                id: [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16,
                                            },
                                            name: stringify!($op_name),
                                            display: $crate::custom_opset!(@display $op_name { $($op_body)* }),
                                            infix: $crate::custom_opset!(@infix { $($op_body)* }),
                                            commutative: $crate::custom_opset!(@bool [commutative] { $($op_body)* }),
                                            associative: $crate::custom_opset!(@bool [associative] { $($op_body)* }),
                                            complexity: $crate::custom_opset!(@complexity { $($op_body)* }),
                                        },
                                    )*
                                )*
                            ];
                            REGISTRY
                        }
                    }
                }

                $crate::paste::paste! {
                    impl $crate::strings::OpNames for $Ops {
                        fn op_name(op: $crate::operator_enum::scalar::OpId) -> &'static str {
                            match op.arity {
                                $(
                                    $arity => match op.id {
                                        $(
                                            x if x == ([<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16) =>
                                                $crate::custom_opset!(@display $op_name { $($op_body)* }),
                                        )*
                                        _ => "unknown_op",
                                    },
                                )*
                                _ => "unknown_op",
                            }
                        }
                    }
                }
            };

            (@eval_body $args:ident { eval: $eval:expr, $($rest:tt)* }) => {
                ($eval)($args)
            };
            (@eval_body $args:ident { eval($eval_args:ident) $eval:block, $($rest:tt)* }) => {{
                let $eval_args = $args;
                $eval
            }};
            (@eval_body $args:ident { $head:tt $($tail:tt)* }) => {
                $crate::custom_opset!(@eval_body $args { $($tail)* })
            };
            (@eval_body $args:ident { }) => {
                compile_error!("custom_opset!: missing eval")
            };

            (@partial_body $args:ident, $idx:ident { partial: $partial:expr, $($rest:tt)* }) => {
                ($partial)($args, $idx)
            };
            (@partial_body $args:ident, $idx:ident { partial($p_args:ident, $p_idx:ident) $partial:block, $($rest:tt)* }) => {{
                let $p_args = $args;
                let $p_idx = $idx;
                $partial
            }};
            (@partial_body $args:ident, $idx:ident { $head:tt $($tail:tt)* }) => {
                $crate::custom_opset!(@partial_body $args, $idx { $($tail)* })
            };
            (@partial_body $args:ident, $idx:ident { }) => {
                compile_error!("custom_opset!: missing partial")
            };

            (@display $op_name:ident { display: $display:expr, $($rest:tt)* }) => {
                $display
            };
            (@display $op_name:ident { $head:tt $($tail:tt)* }) => {
                $crate::custom_opset!(@display $op_name { $($tail)* })
            };
            (@display $op_name:ident { }) => {
                stringify!($op_name)
            };

            (@infix { infix: $infix:expr, $($rest:tt)* }) => {
                Some($infix)
            };
            (@infix { $head:tt $($tail:tt)* }) => {
                $crate::custom_opset!(@infix { $($tail)* })
            };
            (@infix { }) => {
                None
            };

            (@complexity { complexity: $complexity:expr, $($rest:tt)* }) => {
                $complexity
            };
            (@complexity { $head:tt $($tail:tt)* }) => {
                $crate::custom_opset!(@complexity { $($tail)* })
            };
            (@complexity { }) => {
                1u16
            };

            (@bool [commutative] { commutative: $value:expr, $($rest:tt)* }) => {
                $value
            };
            (@bool [associative] { associative: $value:expr, $($rest:tt)* }) => {
                $value
            };
            (@bool [$key:ident] { $head:tt $($tail:tt)* }) => {
                $crate::custom_opset!(@bool [$key] { $($tail)* })
            };
            (@bool [$key:ident] { }) => {
                false
            };

            (@max [$first:literal $(, $rest:literal)*]) => {
                (($first as usize) $(.max($rest as usize))* )
            };
            (@max []) => { 0 };

            ($($other:tt)*) => {
                compile_error!(concat!("could not parse custom_opset! input: ", stringify!($($other)*)));
            };
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
        macro_rules! define_scalar_ops {
    (
        $vis:vis struct $Ops:ident<$t:ty>;
        ops {
            $(($arity:literal, $enum_name:ident) {
                $($op_name:ident => ($op_eval:path, $op_partial:path),)*
            })*
        }
    ) => {
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::operator_enum::scalar::ScalarOpSet<$t> for $Ops {
            fn eval(
                op: $crate::operator_enum::scalar::OpId,
                ctx: $crate::operator_enum::scalar::EvalKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operator_enum::scalar::eval_nary::<$arity, $t>($op_eval, ctx.out, ctx.args, ctx.opts),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn diff(
                op: $crate::operator_enum::scalar::OpId,
                ctx: $crate::operator_enum::scalar::DiffKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operator_enum::scalar::diff_nary::<$arity, $t>(
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
                op: $crate::operator_enum::scalar::OpId,
                ctx: $crate::operator_enum::scalar::GradKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operator_enum::scalar::grad_nary::<$arity, $t>(
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
                impl $crate::operator_enum::scalar::HasOp<$crate::operator_enum::builtin::$op_name, $arity>
                    for $Ops
                {
                    const ID: u16 = $enum_name::$op_name as u16;
                }
            )*
        )*

        impl $crate::operator_enum::names::OpNames for $Ops {
            fn op_name(op: $crate::operator_enum::scalar::OpId) -> &'static str {
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
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::operator_enum::scalar::ScalarOpSet<$t> for $Ops {
            fn eval(
                op: $crate::operator_enum::scalar::OpId,
                ctx: $crate::operator_enum::scalar::EvalKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operator_enum::scalar::eval_apply::<$arity, $t, $crate::operator_enum::builtin::$op_name>(
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
                op: $crate::operator_enum::scalar::OpId,
                ctx: $crate::operator_enum::scalar::DiffKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operator_enum::scalar::diff_apply::<$arity, $t, $crate::operator_enum::builtin::$op_name>(
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
                op: $crate::operator_enum::scalar::OpId,
                ctx: $crate::operator_enum::scalar::GradKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operator_enum::scalar::grad_apply::<$arity, $t, $crate::operator_enum::builtin::$op_name>(ctx),
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
                impl $crate::operator_enum::scalar::HasOp<$crate::operator_enum::builtin::$op_name, $arity> for $Ops {
                    const ID: u16 = $enum_name::$op_name as u16;
                }
            )*
        )*

        impl $crate::operator_enum::names::OpNames for $Ops {
            fn op_name(op: $crate::operator_enum::scalar::OpId) -> &'static str {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) => {
                                    <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::BuiltinOp<$t, $arity>>::DISPLAY
                                }
                            )*
                            _ => "unknown_op",
                        },
                    )*
                    _ => "unknown_op",
                }
            }
        }

        impl $Ops {
            pub const REGISTRY: &'static [$crate::operator_registry::OpInfo] = &[
                $(
                    $(
                        $crate::operator_registry::OpInfo {
                            op: $crate::operator_enum::scalar::OpId {
                                arity: $arity as u8,
                                id: $enum_name::$op_name as u16,
                            },
                            name: <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::BuiltinOp<$t, $arity>>::NAME,
                            display: <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::BuiltinOp<$t, $arity>>::DISPLAY,
                            infix: <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::BuiltinOp<$t, $arity>>::INFIX,
                            commutative: <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::OpMeta<$arity>>::COMMUTATIVE,
                            associative: <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::OpMeta<$arity>>::ASSOCIATIVE,
                            complexity: <$crate::operator_enum::builtin::$op_name as $crate::operator_enum::builtin::OpMeta<$arity>>::COMPLEXITY,
                        },
                    )*
                )*
            ];
        }

        impl $crate::operator_registry::OpRegistry for $Ops {
            #[inline]
            fn registry() -> &'static [$crate::operator_registry::OpInfo] {
                Self::REGISTRY
            }
        }
    };
}
    }

    pub use kernels::*;
    pub use types::*;
}

pub use registry::{LookupError, OpInfo, OpRegistry};
