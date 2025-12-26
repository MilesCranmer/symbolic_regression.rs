use dynamic_expressions::Operator;
use dynamic_expressions::operator_enum::builtin::*;

macro_rules! unary_bad_idx_panics {
    ($name:ident, $x:expr) => {
        dynamic_expressions::paste::paste! {
            #[test]
            #[should_panic]
            fn [<$name:snake _bad_idx_panics>]() {
                let args = [$x];
                let _ = <$name as Operator<f64, 1>>::partial(&args, 1);
            }
        }
    };
}

macro_rules! binary_bad_idx_panics {
    ($name:ident, $x:expr, $y:expr) => {
        dynamic_expressions::paste::paste! {
            #[test]
            #[should_panic]
            fn [<$name:snake _bad_idx_panics>]() {
                let args = [$x, $y];
                let _ = <$name as Operator<f64, 2>>::partial(&args, 2);
            }
        }
    };
}

macro_rules! ternary_bad_idx_panics {
    ($name:ident, $x:expr, $y:expr, $z:expr) => {
        dynamic_expressions::paste::paste! {
            #[test]
            #[should_panic]
            fn [<$name:snake _bad_idx_panics>]() {
                let args = [$x, $y, $z];
                let _ = <$name as Operator<f64, 3>>::partial(&args, 3);
            }
        }
    };
}

unary_bad_idx_panics!(Cos, 0.0f64);
unary_bad_idx_panics!(Sin, 0.0f64);
unary_bad_idx_panics!(Tan, 0.0f64);
unary_bad_idx_panics!(Asin, 0.0f64);
unary_bad_idx_panics!(Acos, 0.0f64);
unary_bad_idx_panics!(Atan, 0.0f64);
unary_bad_idx_panics!(Sinh, 0.0f64);
unary_bad_idx_panics!(Cosh, 0.0f64);
unary_bad_idx_panics!(Tanh, 0.0f64);
unary_bad_idx_panics!(Asinh, 0.0f64);
unary_bad_idx_panics!(Acosh, 2.0f64);
unary_bad_idx_panics!(Atanh, 0.0f64);
unary_bad_idx_panics!(Sec, 0.0f64);
unary_bad_idx_panics!(Csc, 1.0f64);
unary_bad_idx_panics!(Cot, 1.0f64);
unary_bad_idx_panics!(Exp, 0.0f64);
unary_bad_idx_panics!(Exp2, 0.0f64);
unary_bad_idx_panics!(Expm1, 0.0f64);
unary_bad_idx_panics!(Log, 1.0f64);
unary_bad_idx_panics!(Log2, 1.0f64);
unary_bad_idx_panics!(Log10, 1.0f64);
unary_bad_idx_panics!(Log1p, 0.0f64);
unary_bad_idx_panics!(Sqrt, 1.0f64);
unary_bad_idx_panics!(Cbrt, 1.0f64);
unary_bad_idx_panics!(Abs, 1.0f64);
unary_bad_idx_panics!(Abs2, 1.0f64);
unary_bad_idx_panics!(Inv, 1.0f64);
unary_bad_idx_panics!(Sign, 1.0f64);
unary_bad_idx_panics!(Identity, 1.0f64);
unary_bad_idx_panics!(Neg, 1.0f64);

binary_bad_idx_panics!(Add, 0.0f64, 0.0f64);
binary_bad_idx_panics!(Sub, 0.0f64, 0.0f64);
binary_bad_idx_panics!(Mul, 0.0f64, 0.0f64);
binary_bad_idx_panics!(Div, 1.0f64, 1.0f64);
binary_bad_idx_panics!(Pow, 1.0f64, 2.0f64);
binary_bad_idx_panics!(Atan2, 1.0f64, 1.0f64);
binary_bad_idx_panics!(Min, 1.0f64, 2.0f64);
binary_bad_idx_panics!(Max, 1.0f64, 2.0f64);

ternary_bad_idx_panics!(Fma, 1.0f64, 1.0f64, 1.0f64);
ternary_bad_idx_panics!(Clamp, 0.0f64, -1.0f64, 1.0f64);
