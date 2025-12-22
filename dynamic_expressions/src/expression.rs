use crate::node::PNode;
use core::marker::PhantomData;

#[derive(Clone, Debug, Default)]
pub struct Metadata {
    pub variable_names: Vec<String>,
}

#[derive(Debug)]
pub struct PostfixExpr<T, Ops, const D: usize = 2> {
    pub nodes: Vec<PNode>,
    pub consts: Vec<T>,
    pub meta: Metadata,
    _ops: PhantomData<Ops>,
}

impl<T: Clone, Ops, const D: usize> Clone for PostfixExpr<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            consts: self.consts.clone(),
            meta: self.meta.clone(),
            _ops: PhantomData,
        }
    }
}

impl<T, Ops, const D: usize> PostfixExpr<T, Ops, D> {
    pub fn new(nodes: Vec<PNode>, consts: Vec<T>, meta: Metadata) -> Self {
        Self {
            nodes,
            consts,
            meta,
            _ops: PhantomData,
        }
    }

    /// Construct a zero-valued constant expression.
    pub fn zero() -> Self
    where
        T: num_traits::Zero,
    {
        Self::new(
            vec![PNode::Const { idx: 0 }],
            vec![T::zero()],
            Metadata::default(),
        )
    }
}

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
