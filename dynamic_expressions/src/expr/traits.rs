use crate::expr::{Metadata, PNode, PostfixExpr};

pub trait PostfixExpression<const D: usize> {
    type Scalar;
    type Ops;

    fn nodes(&self) -> &[PNode];
    fn consts(&self) -> &[Self::Scalar];
    fn meta(&self) -> &Metadata;
}

pub trait PostfixExpressionMut<const D: usize>: PostfixExpression<D> {
    fn nodes_mut(&mut self) -> &mut [PNode];
    fn consts_mut(&mut self) -> &mut [Self::Scalar];
    fn meta_mut(&mut self) -> &mut Metadata;
}

impl<T, Ops, const D: usize> PostfixExpression<D> for PostfixExpr<T, Ops, D> {
    type Scalar = T;
    type Ops = Ops;

    fn nodes(&self) -> &[PNode] {
        &self.nodes
    }

    fn consts(&self) -> &[Self::Scalar] {
        &self.consts
    }

    fn meta(&self) -> &Metadata {
        &self.meta
    }
}

impl<T, Ops, const D: usize> PostfixExpressionMut<D> for PostfixExpr<T, Ops, D> {
    fn nodes_mut(&mut self) -> &mut [PNode] {
        &mut self.nodes
    }

    fn consts_mut(&mut self) -> &mut [Self::Scalar] {
        &mut self.consts
    }

    fn meta_mut(&mut self) -> &mut Metadata {
        &mut self.meta
    }
}
