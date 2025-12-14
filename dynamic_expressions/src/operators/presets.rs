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
