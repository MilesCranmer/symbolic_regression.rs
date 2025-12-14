use core::marker::PhantomData;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum PNode {
    Var { feature: u16 },
    Const { idx: u16 },
    Op { arity: u8, op: u16 },
}

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
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Src {
    Slot(u16),
    Var(u16),
    Const(u16),
}
