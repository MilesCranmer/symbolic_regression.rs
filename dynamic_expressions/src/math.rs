use crate::algebra::__apply_postfix;
use crate::expr::PostfixExpr;
use crate::operators::builtin::{
    Abs, Abs2, Acos, Acosh, Add, Asin, Asinh, Atan, Atan2, Atanh, Cbrt, Clamp, Cos, Cosh, Cot, Csc,
    Div, Exp, Exp2, Expm1, Fma, Identity, Inv, Log, Log10, Log1p, Log2, Max, Min, Mul, Neg, Pow,
    Sec, Sign, Sin, Sinh, Sqrt, Sub, Tan, Tanh,
};
use crate::operators::scalar::HasOp;

pub fn cos<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Cos, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Cos, 1>>::ID, [x])
}

pub fn sin<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Sin, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Sin, 1>>::ID, [x])
}

pub fn tan<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Tan, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Tan, 1>>::ID, [x])
}

pub fn asin<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Asin, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Asin, 1>>::ID, [x])
}

pub fn acos<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Acos, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Acos, 1>>::ID, [x])
}

pub fn atan<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Atan, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Atan, 1>>::ID, [x])
}

pub fn sinh<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Sinh, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Sinh, 1>>::ID, [x])
}

pub fn cosh<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Cosh, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Cosh, 1>>::ID, [x])
}

pub fn tanh<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Tanh, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Tanh, 1>>::ID, [x])
}

pub fn asinh<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Asinh, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Asinh, 1>>::ID, [x])
}

pub fn acosh<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Acosh, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Acosh, 1>>::ID, [x])
}

pub fn atanh<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Atanh, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Atanh, 1>>::ID, [x])
}

pub fn sec<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Sec, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Sec, 1>>::ID, [x])
}

pub fn csc<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Csc, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Csc, 1>>::ID, [x])
}

pub fn cot<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Cot, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Cot, 1>>::ID, [x])
}

pub fn exp<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Exp, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Exp, 1>>::ID, [x])
}

pub fn exp2<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Exp2, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Exp2, 1>>::ID, [x])
}

pub fn expm1<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Expm1, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Expm1, 1>>::ID, [x])
}

pub fn log<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Log, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Log, 1>>::ID, [x])
}

pub fn log2<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Log2, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Log2, 1>>::ID, [x])
}

pub fn log10<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Log10, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Log10, 1>>::ID, [x])
}

pub fn log1p<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Log1p, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Log1p, 1>>::ID, [x])
}

pub fn sqrt<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Sqrt, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Sqrt, 1>>::ID, [x])
}

pub fn cbrt<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Cbrt, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Cbrt, 1>>::ID, [x])
}

pub fn abs<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Abs, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Abs, 1>>::ID, [x])
}

pub fn abs2<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Abs2, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Abs2, 1>>::ID, [x])
}

pub fn inv<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Inv, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Inv, 1>>::ID, [x])
}

pub fn sign<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Sign, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Sign, 1>>::ID, [x])
}

pub fn identity<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Identity, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Identity, 1>>::ID, [x])
}

pub fn neg<T, Ops, const D: usize>(x: PostfixExpr<T, Ops, D>) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Neg, 1>,
{
    __apply_postfix::<T, Ops, D, 1>(<Ops as HasOp<Neg, 1>>::ID, [x])
}

pub fn div<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Div, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Div, 2>>::ID, [a, b])
}

pub fn add<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Add, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Add, 2>>::ID, [a, b])
}

pub fn sub<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Sub, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Sub, 2>>::ID, [a, b])
}

pub fn mul<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Mul, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Mul, 2>>::ID, [a, b])
}

pub fn pow<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Pow, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Pow, 2>>::ID, [a, b])
}

pub fn atan2<T, Ops, const D: usize>(
    y: PostfixExpr<T, Ops, D>,
    x: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Atan2, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Atan2, 2>>::ID, [y, x])
}

pub fn min<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Min, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Min, 2>>::ID, [a, b])
}

pub fn max<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Max, 2>,
{
    __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<Max, 2>>::ID, [a, b])
}

pub fn fma<T, Ops, const D: usize>(
    a: PostfixExpr<T, Ops, D>,
    b: PostfixExpr<T, Ops, D>,
    c: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Fma, 3>,
{
    __apply_postfix::<T, Ops, D, 3>(<Ops as HasOp<Fma, 3>>::ID, [a, b, c])
}

pub fn clamp<T, Ops, const D: usize>(
    x: PostfixExpr<T, Ops, D>,
    lo: PostfixExpr<T, Ops, D>,
    hi: PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D>
where
    Ops: HasOp<Clamp, 3>,
{
    __apply_postfix::<T, Ops, D, 3>(<Ops as HasOp<Clamp, 3>>::ID, [x, lo, hi])
}
