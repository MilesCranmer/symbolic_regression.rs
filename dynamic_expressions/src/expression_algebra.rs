use crate::expression::{Metadata, PostfixExpr};
use crate::node::PNode;
use crate::operator_enum::builtin::{Add, Div, Mul, Neg, Sub};
use crate::operator_enum::scalar::HasOp;

#[derive(Copy, Clone, Debug)]
pub struct Lit<T>(pub T);

pub fn lit<T>(v: T) -> Lit<T> {
    Lit(v)
}

macro_rules! impl_postfix_binop_self {
    ($Trait:ident, $method:ident, $Marker:ident) => {
        impl<T, Ops, const D: usize> core::ops::$Trait for PostfixExpr<T, Ops, D>
        where
            Ops: HasOp<$Marker, 2>,
        {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                __apply_postfix::<T, Ops, D, 2>(<Ops as HasOp<$Marker, 2>>::ID, [self, rhs])
            }
        }
    };
}

macro_rules! impl_postfix_binop_scalar_rhs {
    ($Trait:ident, $method:ident, $Marker:ident) => {
        impl<T, Ops, const D: usize> core::ops::$Trait<T> for PostfixExpr<T, Ops, D>
        where
            Ops: HasOp<$Marker, 2>,
        {
            type Output = Self;

            fn $method(self, rhs: T) -> Self::Output {
                __apply_postfix::<T, Ops, D, 2>(
                    <Ops as HasOp<$Marker, 2>>::ID,
                    [self, __const_expr::<T, Ops, D>(rhs)],
                )
            }
        }
    };
}

macro_rules! impl_lit_binop_postfix_rhs {
    ($Trait:ident, $method:ident, $Marker:ident) => {
        impl<T, Ops, const D: usize> core::ops::$Trait<PostfixExpr<T, Ops, D>> for Lit<T>
        where
            Ops: HasOp<$Marker, 2>,
        {
            type Output = PostfixExpr<T, Ops, D>;

            fn $method(self, rhs: PostfixExpr<T, Ops, D>) -> Self::Output {
                __apply_postfix::<T, Ops, D, 2>(
                    <Ops as HasOp<$Marker, 2>>::ID,
                    [__const_expr::<T, Ops, D>(self.0), rhs],
                )
            }
        }
    };
}

macro_rules! impl_postfix_unop {
    ($Trait:ident, $method:ident, $Marker:ident, $arity:expr) => {
        impl<T, Ops, const D: usize> core::ops::$Trait for PostfixExpr<T, Ops, D>
        where
            Ops: HasOp<$Marker, $arity>,
        {
            type Output = Self;

            fn $method(self) -> Self::Output {
                __apply_postfix::<T, Ops, D, $arity>(<Ops as HasOp<$Marker, $arity>>::ID, [self])
            }
        }
    };
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

impl_postfix_binop_self!(Add, add, Add);
impl_postfix_binop_self!(Sub, sub, Sub);
impl_postfix_binop_self!(Mul, mul, Mul);
impl_postfix_binop_self!(Div, div, Div);

impl_postfix_unop!(Neg, neg, Neg, 1);

impl_postfix_binop_scalar_rhs!(Add, add, Add);
impl_postfix_binop_scalar_rhs!(Sub, sub, Sub);
impl_postfix_binop_scalar_rhs!(Mul, mul, Mul);
impl_postfix_binop_scalar_rhs!(Div, div, Div);

impl_lit_binop_postfix_rhs!(Add, add, Add);
impl_lit_binop_postfix_rhs!(Sub, sub, Sub);
impl_lit_binop_postfix_rhs!(Mul, mul, Mul);
impl_lit_binop_postfix_rhs!(Div, div, Div);
