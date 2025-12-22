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

use macros::*;

use crate::operator_enum::builtin::*;

unary_wrappers! {
    cos => Cos,
    sin => Sin,
    tan => Tan,
    asin => Asin,
    acos => Acos,
    atan => Atan,
    sinh => Sinh,
    cosh => Cosh,
    tanh => Tanh,
    asinh => Asinh,
    acosh => Acosh,
    atanh => Atanh,
    sec => Sec,
    csc => Csc,
    cot => Cot,
    exp => Exp,
    exp2 => Exp2,
    expm1 => Expm1,
    log => Log,
    log2 => Log2,
    log10 => Log10,
    log1p => Log1p,
    sqrt => Sqrt,
    cbrt => Cbrt,
    abs => Abs,
    abs2 => Abs2,
    inv => Inv,
    sign => Sign,
    identity => Identity,
    neg => Neg,
}

binary_wrappers! {
    div => Div,
    add => Add,
    sub => Sub,
    mul => Mul,
    pow => Pow,
    atan2 => Atan2,
    min => Min,
    max => Max,
}

ternary_wrappers! {
    fma => Fma,
    clamp => Clamp,
}
