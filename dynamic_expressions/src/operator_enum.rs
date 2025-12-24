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
        use num_traits::Float;

        use crate::evaluate::EvalOptions;

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
                SrcRef::Const(c) => c,
            }
        }

        #[derive(Clone, Copy)]
        pub(super) enum ArgView<'a, T> {
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
            pub(super) fn get(&self, row: usize) -> T {
                match self {
                    Self::Slice(s) => s[row],
                    Self::Const(c) => *c,
                }
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
    }

    mod kernels {
        use num_traits::Float;

        use super::{__src_val, ArgView, GradKernelCtx, GradRef, SrcRef, grad_at};
        use crate::evaluate::EvalOptions;
        use crate::operator_enum::builtin::BuiltinOp;
        use crate::utils::ZipEq;

        fn __all_finite<T: Float>(vals: &[T]) -> bool {
            vals.iter().all(|v| v.is_finite())
        }

        #[inline]
        fn finish_complete<T: Float>(out: &mut [T], check_finite: bool, early_exit: bool) -> bool {
            if !check_finite {
                return true;
            }
            let complete = __all_finite(out);
            if !complete && early_exit {
                out.fill(T::nan());
                return false;
            }
            complete
        }

        #[inline]
        fn make_arg_views<'a, T: Float, const A: usize>(args: &[SrcRef<'a, T>]) -> [ArgView<'a, T>; A] {
            core::array::from_fn(|j| args[j].into())
        }

        #[inline]
        fn eval_unary_loop<T: Float, F: Fn(T) -> T>(out: &mut [T], arg: ArgView<'_, T>, eval: F) {
            match arg {
                ArgView::Slice(s) => {
                    for (outv, &av) in out.iter_mut().zip_eq(s) {
                        *outv = eval(av);
                    }
                }
                ArgView::Const(c) => {
                    let v = eval(c);
                    out.fill(v);
                }
            }
        }

        #[inline]
        fn eval_binary_loop<T: Float, F: Fn(T, T) -> T>(
            out: &mut [T],
            lhs: ArgView<'_, T>,
            rhs: ArgView<'_, T>,
            eval: F,
        ) {
            match (lhs, rhs) {
                (ArgView::Slice(a), ArgView::Slice(b)) => {
                    for ((outv, &av), &bv) in out.iter_mut().zip_eq(a).zip_eq(b) {
                        *outv = eval(av, bv);
                    }
                }
                (ArgView::Slice(a), ArgView::Const(bc)) => {
                    for (outv, &av) in out.iter_mut().zip_eq(a) {
                        *outv = eval(av, bc);
                    }
                }
                (ArgView::Const(ac), ArgView::Slice(b)) => {
                    for (outv, &bv) in out.iter_mut().zip_eq(b) {
                        *outv = eval(ac, bv);
                    }
                }
                (ArgView::Const(ac), ArgView::Const(bc)) => {
                    let v = eval(ac, bc);
                    out.fill(v);
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
        fn grad_unary_loop<T: Float, F, const A: usize>(
            out: &mut [T],
            x: ArgView<'_, T>,
            dx: ArgView<'_, T>,
            partial: F,
        ) where
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

        pub fn eval_nary<const A: usize, T: Float>(
            eval: fn(&[T; A]) -> T,
            out: &mut [T],
            args: &[SrcRef<'_, T>],
            opts: &EvalOptions,
        ) -> bool {
            debug_assert_eq!(args.len(), A);
            let check_finite = opts.check_finite;
            let early_exit = opts.early_exit;

            if args.iter().all(|a| matches!(a, SrcRef::Const(_))) {
                let vals: [T; A] = core::array::from_fn(|j| __src_val(args[j], 0));
                let v = eval(&vals);
                out.fill(v);
                if !check_finite {
                    return true;
                }
                return finish_complete(out, check_finite, early_exit);
            }

            let views: [ArgView<'_, T>; A] = make_arg_views(args);

            if A == 1 {
                eval_unary_loop(out, views[0], |a| eval(&vals1(a)));
            } else if A == 2 {
                eval_binary_loop(out, views[0], views[1], |a, b| eval(&vals2(a, b)));
            } else {
                let mut vals: [T; A] = [T::zero(); A];
                for (row, outv) in out.iter_mut().enumerate() {
                    for (v, view) in vals.iter_mut().zip_eq(views) {
                        *v = view.get(row);
                    }
                    *outv = eval(&vals);
                }
            }

            finish_complete(out, check_finite, early_exit)
        }

        pub fn eval_apply<const A: usize, T: Float, Op: BuiltinOp<T, A>>(
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
                return finish_complete(out, check_finite, early_exit);
            }

            let views: [ArgView<'_, T>; A] = make_arg_views(args);

            if A == 1 {
                eval_unary_loop(out, views[0], |a| {
                    let vals = vals1::<T, A>(a);
                    Op::eval(&vals)
                });
                return finish_complete(out, check_finite, early_exit);
            }
            if A == 2 {
                eval_binary_loop(out, views[0], views[1], |a, b| {
                    let vals = vals2(a, b);
                    Op::eval(&vals)
                });
                return finish_complete(out, check_finite, early_exit);
            }

            let mut vals: [T; A] = [T::zero(); A];
            for (row, outv) in out.iter_mut().enumerate() {
                for (v, view) in vals.iter_mut().zip_eq(views) {
                    *v = view.get(row);
                }
                *outv = Op::eval(&vals);
            }

            finish_complete(out, check_finite, early_exit)
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

            let mut vals: [T; A] = [T::zero(); A];
            let mut dvals: [T; A] = [T::zero(); A];
            let val_views: [ArgView<'_, T>; A] = make_arg_views(args);
            let dval_views: [ArgView<'_, T>; A] = make_arg_views(dargs);

            for ((row, outv), outd) in out_val.iter_mut().enumerate().zip_eq(out_der.iter_mut()) {
                for (v, view) in vals.iter_mut().zip_eq(val_views) {
                    *v = view.get(row);
                }
                for (dv, view) in dvals.iter_mut().zip_eq(dval_views) {
                    *dv = view.get(row);
                }
                let v = eval(&vals);
                let mut d = T::zero();
                for (j, dv) in dvals.iter().enumerate() {
                    d += partial(&vals, j) * *dv;
                }
                *outv = v;
                *outd = d;
            }

            if check_finite {
                let finite = __all_finite(out_val);
                complete &= finite;
                if !finite && early_exit {
                    out_val.fill(T::nan());
                    out_der.fill(T::nan());
                    return false;
                }
            }

            complete
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

            let mut vals: [T; A] = [T::zero(); A];
            let mut dvals: [T; A] = [T::zero(); A];
            let val_views: [ArgView<'_, T>; A] = make_arg_views(args);
            let dval_views: [ArgView<'_, T>; A] = make_arg_views(dargs);

            for ((row, outv), outd) in out_val.iter_mut().enumerate().zip_eq(out_der.iter_mut()) {
                for (v, view) in vals.iter_mut().zip_eq(val_views) {
                    *v = view.get(row);
                }
                for (dv, view) in dvals.iter_mut().zip_eq(dval_views) {
                    *dv = view.get(row);
                }
                let v = Op::eval(&vals);
                let mut d = T::zero();
                for (j, dv) in dvals.iter().enumerate() {
                    d += Op::partial(&vals, j) * *dv;
                }
                *outv = v;
                *outd = d;
            }

            if check_finite {
                let finite = __all_finite(out_val);
                complete &= finite;
                if !finite && early_exit {
                    out_val.fill(T::nan());
                    out_der.fill(T::nan());
                    return false;
                }
            }

            complete
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
            let arg_views: [ArgView<'_, T>; A] = make_arg_views(ctx.args);

            if A == 1 {
                eval_unary_loop(ctx.out_val, arg_views[0], |a| eval(&vals1(a)));
                let x = arg_views[0];
                let dx_ref = ctx.arg_grads[0];
                let n_rows = ctx.n_rows;
                for dir in 0..ctx.n_dir {
                    let grad_dir = &mut ctx.out_grad[dir * n_rows..(dir + 1) * n_rows];
                    let dx = grad_dir_view(dx_ref, dir, n_rows);
                    grad_unary_loop::<T, _, A>(grad_dir, x, dx, partial);
                }
            } else if A == 2 {
                eval_binary_loop(ctx.out_val, arg_views[0], arg_views[1], |a, b| eval(&vals2(a, b)));

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
                    grad_binary_loop::<T, _, A>(grad_dir, x, y, dx, dy, partial);
                }
            } else {
                let mut vals: [T; A] = [T::zero(); A];

                for (row, outv) in ctx.out_val.iter_mut().enumerate() {
                    for (v, view) in vals.iter_mut().zip_eq(arg_views) {
                        *v = view.get(row);
                    }
                    *outv = eval(&vals);
                }

                for (dir, grad_dir) in ctx.out_grad.chunks_mut(ctx.n_rows).enumerate().take(ctx.n_dir) {
                    for (row, outg) in grad_dir.iter_mut().enumerate() {
                        for (v, view) in vals.iter_mut().zip_eq(arg_views) {
                            *v = view.get(row);
                        }
                        let mut g = T::zero();
                        for (j, ag) in ctx.arg_grads.iter().copied().enumerate() {
                            g += partial(&vals, j) * grad_at(ag, dir, row, ctx.n_rows);
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

        pub fn grad_apply<const A: usize, T: Float + core::ops::AddAssign, Op: BuiltinOp<T, A>>(
            ctx: GradKernelCtx<'_, '_, T>,
        ) -> bool {
            debug_assert_eq!(ctx.args.len(), A);
            debug_assert_eq!(ctx.arg_grads.len(), A);

            let check_finite = ctx.opts.check_finite;
            let early_exit = ctx.opts.early_exit;
            let mut complete = true;
            let arg_views: [ArgView<'_, T>; A] = make_arg_views(ctx.args);

            if A == 1 {
                eval_unary_loop(ctx.out_val, arg_views[0], |a| {
                    let vals = vals1::<T, A>(a);
                    Op::eval(&vals)
                });

                let x = arg_views[0];
                let dx_ref = ctx.arg_grads[0];
                let n_rows = ctx.n_rows;
                for dir in 0..ctx.n_dir {
                    let grad_dir = &mut ctx.out_grad[dir * n_rows..(dir + 1) * n_rows];
                    let dx = grad_dir_view(dx_ref, dir, n_rows);
                    grad_unary_loop::<T, _, A>(grad_dir, x, dx, Op::partial);
                }
            } else if A == 2 {
                eval_binary_loop(ctx.out_val, arg_views[0], arg_views[1], |a, b| {
                    let vals = vals2(a, b);
                    Op::eval(&vals)
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
