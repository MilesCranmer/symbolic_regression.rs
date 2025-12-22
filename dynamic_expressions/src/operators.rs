mod macros {
    macro_rules! unary_wrappers {
        ($( $fname:ident => $Op:ty ),* $(,)?) => {
            $(
                #[inline]
                #[must_use]
                pub fn $fname<T, Ops, const D: usize>(
                    x: crate::expression::PostfixExpr<T, Ops, D>,
                ) -> crate::expression::PostfixExpr<T, Ops, D>
                where
                    Ops: crate::operator_enum::scalar::HasOp<$Op, 1>,
                {
                    crate::expression_algebra::__apply_postfix::<T, Ops, D, 1>(
                        <Ops as crate::operator_enum::scalar::HasOp<$Op, 1>>::ID,
                        [x],
                    )
                }
            )*
        };
    }

    macro_rules! binary_wrappers {
        ($( $fname:ident => $Op:ty ),* $(,)?) => {
            $(
                #[inline]
                #[must_use]
                pub fn $fname<T, Ops, const D: usize>(
                    x: crate::expression::PostfixExpr<T, Ops, D>,
                    y: crate::expression::PostfixExpr<T, Ops, D>,
                ) -> crate::expression::PostfixExpr<T, Ops, D>
                where
                    Ops: crate::operator_enum::scalar::HasOp<$Op, 2>,
                {
                    crate::expression_algebra::__apply_postfix::<T, Ops, D, 2>(
                        <Ops as crate::operator_enum::scalar::HasOp<$Op, 2>>::ID,
                        [x, y],
                    )
                }
            )*
        };
    }

    macro_rules! ternary_wrappers {
        ($( $fname:ident => $Op:ty ),* $(,)?) => {
            $(
                #[inline]
                #[must_use]
                pub fn $fname<T, Ops, const D: usize>(
                    x: crate::expression::PostfixExpr<T, Ops, D>,
                    y: crate::expression::PostfixExpr<T, Ops, D>,
                    z: crate::expression::PostfixExpr<T, Ops, D>,
                ) -> crate::expression::PostfixExpr<T, Ops, D>
                where
                    Ops: crate::operator_enum::scalar::HasOp<$Op, 3>,
                {
                    crate::expression_algebra::__apply_postfix::<T, Ops, D, 3>(
                        <Ops as crate::operator_enum::scalar::HasOp<$Op, 3>>::ID,
                        [x, y, z],
                    )
                }
            )*
        };
    }

    pub(crate) use {binary_wrappers, ternary_wrappers, unary_wrappers};
}

use macros::{binary_wrappers, ternary_wrappers, unary_wrappers};

use crate::operator_enum::builtin;

unary_wrappers! {
    cos => builtin::Cos,
    sin => builtin::Sin,
    tan => builtin::Tan,
    asin => builtin::Asin,
    acos => builtin::Acos,
    atan => builtin::Atan,
    sinh => builtin::Sinh,
    cosh => builtin::Cosh,
    tanh => builtin::Tanh,
    asinh => builtin::Asinh,
    acosh => builtin::Acosh,
    atanh => builtin::Atanh,
    sec => builtin::Sec,
    csc => builtin::Csc,
    cot => builtin::Cot,
    exp => builtin::Exp,
    exp2 => builtin::Exp2,
    expm1 => builtin::Expm1,
    log => builtin::Log,
    log2 => builtin::Log2,
    log10 => builtin::Log10,
    log1p => builtin::Log1p,
    sqrt => builtin::Sqrt,
    cbrt => builtin::Cbrt,
    abs => builtin::Abs,
    abs2 => builtin::Abs2,
    inv => builtin::Inv,
    sign => builtin::Sign,
    identity => builtin::Identity,
    neg => builtin::Neg,
}

binary_wrappers! {
    div => builtin::Div,
    add => builtin::Add,
    sub => builtin::Sub,
    mul => builtin::Mul,
    pow => builtin::Pow,
    atan2 => builtin::Atan2,
    min => builtin::Min,
    max => builtin::Max,
}

ternary_wrappers! {
    fma => builtin::Fma,
    clamp => builtin::Clamp,
}
