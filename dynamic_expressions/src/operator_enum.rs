pub mod builtin {
    use num_traits::Float;

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

        impl $crate::traits::OpTag for $Op {
            const ARITY: u8 = $A as u8;
        }

        impl<T: Float> $crate::traits::Operator<T, $A> for $Op {
            const NAME: &'static str = builtin_op!(@name $Op $(, $name)?);
            const INFIX: Option<&'static str> = builtin_op!(@infix $($infix)?);
            const DISPLAY: &'static str = match builtin_op!(@infix $($infix)?) {
                Some(s) => s,
                None => builtin_op!(@name $Op $(, $name)?),
            };
            const COMMUTATIVE: bool = builtin_op!(@commutative $($commutative)?);
            const ASSOCIATIVE: bool = builtin_op!(@associative $($associative)?);
            const COMPLEXITY: u16 = builtin_op!(@complexity $($complexity)?);

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

// -------------------------------------------------------------------------------------------------
// Operator set macros
// -------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! custom_opset {
    // Compute the maximum arity.
    (@max_arity $a:literal) => { $a as u8 };
    (@max_arity $a:literal, $($rest:literal),+ $(,)?) => {
        if $a as u8 > $crate::custom_opset!(@max_arity $($rest),+) {
            $a as u8
        } else {
            $crate::custom_opset!(@max_arity $($rest),+)
        }
    };

    (
        $(#[$meta:meta])* $vis:vis struct $Ops:ident<$t:ty> {
            $( $arity:literal {
                $( $op_name:ident { $($op_body:tt)* } )*
            } )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops;

        // Per-arity IDs and eval/partial helpers.
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

        // Marker types implementing `Operator`.
        $crate::paste::paste! {
            $(
                $(
                    #[derive(Copy, Clone, Debug, Default)]
                    $vis struct [<$Ops $op_name:camel>];

                    impl $crate::traits::OpTag for [<$Ops $op_name:camel>] {
                        const ARITY: u8 = $arity as u8;
                    }

                    impl $crate::traits::Operator<$t, $arity> for [<$Ops $op_name:camel>] {
                        const NAME: &'static str = stringify!($op_name);
                        const DISPLAY: &'static str = $crate::custom_opset!(@display $op_name { $($op_body)* });
                        const INFIX: Option<&'static str> = $crate::custom_opset!(@infix { $($op_body)* });
                        const COMMUTATIVE: bool = $crate::custom_opset!(@bool [commutative] { $($op_body)* });
                        const ASSOCIATIVE: bool = $crate::custom_opset!(@bool [associative] { $($op_body)* });
                        const COMPLEXITY: u16 = $crate::custom_opset!(@complexity { $($op_body)* });

                        #[inline]
                        fn eval(args: &[$t; $arity]) -> $t {
                            [<__ $Ops _ $op_name _eval>](args)
                        }

                        #[inline]
                        fn partial(args: &[$t; $arity], idx: usize) -> $t {
                            [<__ $Ops _ $op_name _partial>](args, idx)
                        }
                    }
                )*
            )*
        }

        impl $crate::traits::OperatorSet for $Ops {
            type T = $t;
            const MAX_ARITY: u8 = $crate::custom_opset!(@max_arity $($arity),*);

            fn ops_with_arity(arity: u8) -> &'static [u16] {
                match arity {
                    $(
                        $arity => &[$( $crate::paste::paste! { [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16 } , )*],
                    )*
                    _ => &[],
                }
            }

            fn meta(op: $crate::traits::OpId) -> Option<&'static $crate::traits::OpMeta> {
                match op.arity {
                    $(
                        $arity => {
                            $crate::paste::paste! {
                                const META: &[$crate::traits::OpMeta] = &[
                                    $(
                                        $crate::traits::OpMeta {
                                            arity: $arity as u8,
                                            id: [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16,
                                            name: <[<$Ops $op_name:camel>] as $crate::traits::Operator<$t, $arity>>::NAME,
                                            display: <[<$Ops $op_name:camel>] as $crate::traits::Operator<$t, $arity>>::DISPLAY,
                                            infix: <[<$Ops $op_name:camel>] as $crate::traits::Operator<$t, $arity>>::INFIX,
                                            commutative: <[<$Ops $op_name:camel>] as $crate::traits::Operator<$t, $arity>>::COMMUTATIVE,
                                            associative: <[<$Ops $op_name:camel>] as $crate::traits::Operator<$t, $arity>>::ASSOCIATIVE,
                                            complexity: <[<$Ops $op_name:camel>] as $crate::traits::Operator<$t, $arity>>::COMPLEXITY,
                                        },
                                    )*
                                ];
                                META.get(op.id as usize)
                            }
                        }
                    )*
                    _ => None,
                }
            }

            fn eval(op: $crate::traits::OpId, ctx: $crate::dispatch::EvalKernelCtx<'_, '_, $t>) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($crate::paste::paste! { [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16 }) =>
                                    $crate::paste::paste! {
                                        $crate::evaluate::kernels::eval_nary::<$arity, $t, [<$Ops $op_name:camel>]>(
                                            ctx.out,
                                            ctx.args,
                                            ctx.opts,
                                        )
                                    },
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn diff(op: $crate::traits::OpId, ctx: $crate::dispatch::DiffKernelCtx<'_, '_, $t>) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($crate::paste::paste! { [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16 }) =>
                                    $crate::paste::paste! {
                                        $crate::evaluate::kernels::diff_nary::<$arity, $t, [<$Ops $op_name:camel>]>(
                                            ctx.out_val,
                                            ctx.out_der,
                                            ctx.args,
                                            ctx.dargs,
                                            ctx.opts,
                                        )
                                    },
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn grad(op: $crate::traits::OpId, ctx: $crate::dispatch::GradKernelCtx<'_, '_, $t>) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($crate::paste::paste! { [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16 }) =>
                                    $crate::paste::paste! {
                                        $crate::evaluate::kernels::grad_nary::<$arity, $t, [<$Ops $op_name:camel>]>(ctx)
                                    },
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }
        }

        $crate::paste::paste! {
            $(
                $(
                    impl $crate::traits::HasOp<[<$Ops $op_name:camel>]> for $Ops {
                        const ID: u16 = [<__ $Ops _op_id_ $arity>]::[<$op_name:camel>] as u16;
                    }
                )*
            )*
        }
    };

    // ---- helper parsers ----

    (@eval_body $args:ident { eval: $eval:expr, $($rest:tt)* }) => { ($eval)($args) };
    (@eval_body $args:ident { eval($pat:pat) $body:block, $($rest:tt)* }) => {{ let $pat = $args; $body }};
    (@eval_body $args:ident { $head:tt $($tail:tt)* }) => { $crate::custom_opset!(@eval_body $args { $($tail)* }) };
    (@eval_body $args:ident { }) => { compile_error!("custom_opset!: missing eval") };

    (@partial_body $args:ident, $idx:ident { partial: $partial:expr, $($rest:tt)* }) => { ($partial)($args, $idx) };
    (@partial_body $args:ident, $idx:ident { partial($pat:pat, $ipat:pat) $body:block, $($rest:tt)* }) => {{
        let $pat = $args;
        let $ipat = $idx;
        $body
    }};
    (@partial_body $args:ident, $idx:ident { $head:tt $($tail:tt)* }) => { $crate::custom_opset!(@partial_body $args, $idx { $($tail)* }) };
    (@partial_body $args:ident, $idx:ident { }) => { compile_error!("custom_opset!: missing partial") };

    (@display $op_name:ident { display: $d:expr, $($rest:tt)* }) => { $d };
    (@display $op_name:ident { $head:tt $($tail:tt)* }) => { $crate::custom_opset!(@display $op_name { $($tail)* }) };
    (@display $op_name:ident { }) => { stringify!($op_name) };

    (@infix { infix: $i:expr, $($rest:tt)* }) => { Some($i) };
    (@infix { $head:tt $($tail:tt)* }) => { $crate::custom_opset!(@infix { $($tail)* }) };
    (@infix { }) => { None };

    (@complexity { complexity: $c:expr, $($rest:tt)* }) => { $c };
    (@complexity { $head:tt $($tail:tt)* }) => { $crate::custom_opset!(@complexity { $($tail)* }) };
    (@complexity { }) => { 1u16 };

    (@bool [commutative] { commutative: $b:expr, $($rest:tt)* }) => { $b };
    (@bool [associative] { associative: $b:expr, $($rest:tt)* }) => { $b };
    (@bool [$key:ident] { $head:tt $($tail:tt)* }) => { $crate::custom_opset!(@bool [$key] { $($tail)* }) };
    (@bool [$key:ident] { }) => { false };

    ($($other:tt)*) => {
        compile_error!(concat!("could not parse custom_opset! input: ", stringify!($($other)*)));
    };
}

#[macro_export]
macro_rules! opset {
    // Compute the maximum arity.
    (@max_arity $a:literal) => { $a as u8 };
    (@max_arity $a:literal, $($rest:literal),+ $(,)?) => {
        if $a as u8 > $crate::opset!(@max_arity $($rest),+) {
            $a as u8
        } else {
            $crate::opset!(@max_arity $($rest),+)
        }
    };

    (
        $(#[$meta:meta])* $vis:vis struct $Ops:ident<$t:ty>;
        ops {
            $( ($arity:literal, $enum_name:ident) { $($op_name:ident),* $(,)? } )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            #[allow(non_camel_case_types)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::traits::OperatorSet for $Ops {
            type T = $t;
            const MAX_ARITY: u8 = $crate::opset!(@max_arity $($arity),*);

            fn ops_with_arity(arity: u8) -> &'static [u16] {
                match arity {
                    $(
                        $arity => &[$($enum_name::$op_name as u16,)*],
                    )*
                    _ => &[],
                }
            }

            fn meta(op: $crate::traits::OpId) -> Option<&'static $crate::traits::OpMeta> {
                match op.arity {
                    $(
                        $arity => {
                            const META: &[$crate::traits::OpMeta] = &[
                                $(
                                    $crate::traits::OpMeta {
                                        arity: $arity as u8,
                                        id: $enum_name::$op_name as u16,
                                        name: <$crate::operator_enum::builtin::$op_name as $crate::traits::Operator<$t, $arity>>::NAME,
                                        display: <$crate::operator_enum::builtin::$op_name as $crate::traits::Operator<$t, $arity>>::DISPLAY,
                                        infix: <$crate::operator_enum::builtin::$op_name as $crate::traits::Operator<$t, $arity>>::INFIX,
                                        commutative: <$crate::operator_enum::builtin::$op_name as $crate::traits::Operator<$t, $arity>>::COMMUTATIVE,
                                        associative: <$crate::operator_enum::builtin::$op_name as $crate::traits::Operator<$t, $arity>>::ASSOCIATIVE,
                                        complexity: <$crate::operator_enum::builtin::$op_name as $crate::traits::Operator<$t, $arity>>::COMPLEXITY,
                                    },
                                )*
                            ];
                            META.get(op.id as usize)
                        }
                    )*
                    _ => None,
                }
            }

            fn eval(op: $crate::traits::OpId, ctx: $crate::dispatch::EvalKernelCtx<'_, '_, $t>) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::evaluate::kernels::eval_nary::<$arity, $t, $crate::operator_enum::builtin::$op_name>(
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

            fn diff(op: $crate::traits::OpId, ctx: $crate::dispatch::DiffKernelCtx<'_, '_, $t>) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::evaluate::kernels::diff_nary::<$arity, $t, $crate::operator_enum::builtin::$op_name>(
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

            fn grad(op: $crate::traits::OpId, ctx: $crate::dispatch::GradKernelCtx<'_, '_, $t>) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::evaluate::kernels::grad_nary::<$arity, $t, $crate::operator_enum::builtin::$op_name>(ctx),
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
                impl $crate::traits::HasOp<$crate::operator_enum::builtin::$op_name> for $Ops {
                    const ID: u16 = $enum_name::$op_name as u16;
                }
            )*
        )*
    };
}
