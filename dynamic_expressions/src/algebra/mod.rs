use crate::expr::{Metadata, PNode, PostfixExpr};
use crate::operators::scalar::OpId;

pub trait HasAdd {
    const ADD: OpId;
}
pub trait HasSub {
    const SUB: OpId;
}
pub trait HasMul {
    const MUL: OpId;
}
pub trait HasDiv {
    const DIV: OpId;
}
pub trait HasNeg {
    const NEG: OpId;
}

#[derive(Copy, Clone, Debug)]
pub struct Lit<T>(pub T);

pub fn lit<T>(v: T) -> Lit<T> {
    Lit(v)
}

#[doc(hidden)]
pub fn __apply_postfix<T, Ops, const D: usize, const A: usize>(
    op_id: u16,
    mut args: [PostfixExpr<T, Ops, D>; A],
) -> PostfixExpr<T, Ops, D> {
    assert!(A <= D, "apply arity {} exceeds max arity D={}", A, D);

    let mut out_nodes: Vec<PNode> = Vec::new();
    let mut out_consts: Vec<T> = Vec::new();
    let mut out_meta = Metadata::default();

    for e in args.iter_mut() {
        if out_meta.variable_names.is_empty() && !e.meta.variable_names.is_empty() {
            out_meta.variable_names = core::mem::take(&mut e.meta.variable_names);
        }

        let offset: u16 = out_consts
            .len()
            .try_into()
            .unwrap_or_else(|_| panic!("too many constants to index in u16"));

        for n in e.nodes.iter_mut() {
            if let PNode::Const { idx } = n {
                *idx = idx
                    .checked_add(offset)
                    .unwrap_or_else(|| panic!("constant index overflow"));
            }
        }

        out_consts.append(&mut e.consts);
        out_nodes.append(&mut e.nodes);
    }

    out_nodes.push(PNode::Op {
        arity: A as u8,
        op: op_id,
    });
    PostfixExpr::new(out_nodes, out_consts, out_meta)
}

fn __const_expr<T, Ops, const D: usize>(value: T) -> PostfixExpr<T, Ops, D> {
    PostfixExpr::new(
        vec![PNode::Const { idx: 0 }],
        vec![value],
        Default::default(),
    )
}

impl<T, Ops, const D: usize> core::ops::Add for PostfixExpr<T, Ops, D>
where
    Ops: HasAdd,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(Ops::ADD.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::ADD.id, [self, rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Sub for PostfixExpr<T, Ops, D>
where
    Ops: HasSub,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(Ops::SUB.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::SUB.id, [self, rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Mul for PostfixExpr<T, Ops, D>
where
    Ops: HasMul,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(Ops::MUL.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::MUL.id, [self, rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Div for PostfixExpr<T, Ops, D>
where
    Ops: HasDiv,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(Ops::DIV.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::DIV.id, [self, rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Neg for PostfixExpr<T, Ops, D>
where
    Ops: HasNeg,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        debug_assert_eq!(Ops::NEG.arity, 1);
        __apply_postfix::<T, Ops, D, 1>(Ops::NEG.id, [self])
    }
}

impl<T, Ops, const D: usize> core::ops::Add<T> for PostfixExpr<T, Ops, D>
where
    Ops: HasAdd,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        debug_assert_eq!(Ops::ADD.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::ADD.id, [self, __const_expr::<T, Ops, D>(rhs)])
    }
}

impl<T, Ops, const D: usize> core::ops::Sub<T> for PostfixExpr<T, Ops, D>
where
    Ops: HasSub,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        debug_assert_eq!(Ops::SUB.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::SUB.id, [self, __const_expr::<T, Ops, D>(rhs)])
    }
}

impl<T, Ops, const D: usize> core::ops::Mul<T> for PostfixExpr<T, Ops, D>
where
    Ops: HasMul,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        debug_assert_eq!(Ops::MUL.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::MUL.id, [self, __const_expr::<T, Ops, D>(rhs)])
    }
}

impl<T, Ops, const D: usize> core::ops::Div<T> for PostfixExpr<T, Ops, D>
where
    Ops: HasDiv,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        debug_assert_eq!(Ops::DIV.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::DIV.id, [self, __const_expr::<T, Ops, D>(rhs)])
    }
}

impl<T, Ops, const D: usize> core::ops::Add<PostfixExpr<T, Ops, D>> for Lit<T>
where
    Ops: HasAdd,
{
    type Output = PostfixExpr<T, Ops, D>;

    fn add(self, rhs: PostfixExpr<T, Ops, D>) -> Self::Output {
        debug_assert_eq!(Ops::ADD.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::ADD.id, [__const_expr::<T, Ops, D>(self.0), rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Sub<PostfixExpr<T, Ops, D>> for Lit<T>
where
    Ops: HasSub,
{
    type Output = PostfixExpr<T, Ops, D>;

    fn sub(self, rhs: PostfixExpr<T, Ops, D>) -> Self::Output {
        debug_assert_eq!(Ops::SUB.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::SUB.id, [__const_expr::<T, Ops, D>(self.0), rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Mul<PostfixExpr<T, Ops, D>> for Lit<T>
where
    Ops: HasMul,
{
    type Output = PostfixExpr<T, Ops, D>;

    fn mul(self, rhs: PostfixExpr<T, Ops, D>) -> Self::Output {
        debug_assert_eq!(Ops::MUL.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::MUL.id, [__const_expr::<T, Ops, D>(self.0), rhs])
    }
}

impl<T, Ops, const D: usize> core::ops::Div<PostfixExpr<T, Ops, D>> for Lit<T>
where
    Ops: HasDiv,
{
    type Output = PostfixExpr<T, Ops, D>;

    fn div(self, rhs: PostfixExpr<T, Ops, D>) -> Self::Output {
        debug_assert_eq!(Ops::DIV.arity, 2);
        __apply_postfix::<T, Ops, D, 2>(Ops::DIV.id, [__const_expr::<T, Ops, D>(self.0), rhs])
    }
}
