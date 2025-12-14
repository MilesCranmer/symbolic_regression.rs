use num_traits::Float;

pub trait BuiltinOp<T: Float, const A: usize> {
    const NAME: &'static str;
    const INFIX: Option<&'static str> = None;
    const DISPLAY: &'static str = Self::NAME;

    fn eval(args: &[T; A]) -> T;
    fn partial(args: &[T; A], idx: usize) -> T;
}

fn two<T: Float>() -> T {
    T::one() + T::one()
}

macro_rules! builtin_op {
    (
        $(#[$meta:meta])*
        $Op:ident : $A:literal {
            name: $name:literal,
            infix: $infix:expr,
            eval($args:ident) $eval:block,
            partial($pargs:ident, $idx:ident) $partial:block $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $Op;

        impl<T: Float> BuiltinOp<T, $A> for $Op {
            const NAME: &'static str = $name;
            const INFIX: Option<&'static str> = $infix;
            const DISPLAY: &'static str = match $infix {
                Some(s) => s,
                None => $name,
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
    name: "cos",
    infix: None,
    eval(args) { args[0].cos() },
    partial(args, idx) {
        match idx {
            0 => -args[0].sin(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sin: 1 {
    name: "sin",
    infix: None,
    eval(args) { args[0].sin() },
    partial(args, idx) {
        match idx {
            0 => args[0].cos(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Tan: 1 {
    name: "tan",
    infix: None,
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
    name: "asin",
    infix: None,
    eval(args) { args[0].asin() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() - args[0] * args[0]).sqrt(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Acos: 1 {
    name: "acos",
    infix: None,
    eval(args) { args[0].acos() },
    partial(args, idx) {
        match idx {
            0 => -T::one() / (T::one() - args[0] * args[0]).sqrt(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Atan: 1 {
    name: "atan",
    infix: None,
    eval(args) { args[0].atan() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() + args[0] * args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sinh: 1 {
    name: "sinh",
    infix: None,
    eval(args) { args[0].sinh() },
    partial(args, idx) {
        match idx {
            0 => args[0].cosh(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Cosh: 1 {
    name: "cosh",
    infix: None,
    eval(args) { args[0].cosh() },
    partial(args, idx) {
        match idx {
            0 => args[0].sinh(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Tanh: 1 {
    name: "tanh",
    infix: None,
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
    name: "asinh",
    infix: None,
    eval(args) { args[0].asinh() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (args[0] * args[0] + T::one()).sqrt(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Acosh: 1 {
    name: "acosh",
    infix: None,
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
    name: "atanh",
    infix: None,
    eval(args) { args[0].atanh() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() - args[0] * args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sec: 1 {
    name: "sec",
    infix: None,
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
    name: "csc",
    infix: None,
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
    name: "cot",
    infix: None,
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
    name: "exp",
    infix: None,
    eval(args) { args[0].exp() },
    partial(args, idx) {
        match idx {
            0 => args[0].exp(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log: 1 {
    name: "log",
    infix: None,
    eval(args) { args[0].ln() },
    partial(args, idx) {
        match idx {
            0 => T::one() / args[0],
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log2: 1 {
    name: "log2",
    infix: None,
    eval(args) { args[0].log2() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (args[0] * two::<T>().ln()),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log10: 1 {
    name: "log10",
    infix: None,
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
    name: "log1p",
    infix: None,
    eval(args) { args[0].ln_1p() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() + args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Exp2: 1 {
    name: "exp2",
    infix: None,
    eval(args) { args[0].exp2() },
    partial(args, idx) {
        match idx {
            0 => args[0].exp2() * two::<T>().ln(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Expm1: 1 {
    name: "expm1",
    infix: None,
    eval(args) { args[0].exp_m1() },
    partial(args, idx) {
        match idx {
            0 => args[0].exp(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sqrt: 1 {
    name: "sqrt",
    infix: None,
    eval(args) { args[0].sqrt() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (two::<T>() * args[0].sqrt()),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Cbrt: 1 {
    name: "cbrt",
    infix: None,
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
    name: "abs",
    infix: None,
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
    name: "abs2",
    infix: None,
    eval(args) { args[0] * args[0] },
    partial(args, idx) {
        match idx {
            0 => two::<T>() * args[0],
            _ => unreachable!(),
        }
    },
});

builtin_op!(Inv: 1 {
    name: "inv",
    infix: None,
    eval(args) { T::one() / args[0] },
    partial(args, idx) {
        match idx {
            0 => -T::one() / (args[0] * args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sign: 1 {
    name: "sign",
    infix: None,
    eval(args) { args[0].signum() },
    partial(_args, idx) {
        match idx {
            0 => T::zero(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Identity: 1 {
    name: "identity",
    infix: None,
    eval(args) { args[0] },
    partial(_args, idx) {
        match idx {
            0 => T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Neg: 1 {
    name: "neg",
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
    name: "add",
    infix: Some("+"),
    eval(args) { args[0] + args[1] },
    partial(_args, idx) {
        match idx {
            0 | 1 => T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sub: 2 {
    name: "sub",
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
    name: "mul",
    infix: Some("*"),
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
    name: "div",
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
    name: "pow",
    infix: None,
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
    name: "atan2",
    infix: None,
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
    name: "min",
    infix: None,
    eval(args) {
        if args[0] < args[1] { args[0] } else { args[1] }
    },
    partial(args, idx) {
        let half = T::from(0.5).expect("Float can represent 0.5");
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
    name: "max",
    infix: None,
    eval(args) {
        if args[0] > args[1] { args[0] } else { args[1] }
    },
    partial(args, idx) {
        let half = T::from(0.5).expect("Float can represent 0.5");
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
    name: "fma",
    infix: None,
    eval(args) { args[0] * args[1] + args[2] },
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
    name: "clamp",
    infix: None,
    eval(args) {
        let x = args[0];
        let lo = args[1];
        let hi = args[2];
        if x < lo { lo } else if x > hi { hi } else { x }
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
