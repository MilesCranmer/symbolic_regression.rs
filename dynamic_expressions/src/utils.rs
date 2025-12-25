use crate::expression::PostfixExpr;
use crate::node::PNode;

/// Extension trait that provides `.zip_eq()` - like `.zip()` but debug-asserts equal lengths.
/// In release builds, this compiles to a plain `zip` with zero overhead.
pub trait ZipEq: ExactSizeIterator + Sized {
    #[inline]
    fn zip_eq<B>(self, other: B) -> std::iter::Zip<Self, B::IntoIter>
    where
        B: IntoIterator,
        B::IntoIter: ExactSizeIterator,
    {
        let other = other.into_iter();
        debug_assert_eq!(self.len(), other.len(), "zip_eq: length mismatch");
        self.zip(other)
    }
}

impl<I: ExactSizeIterator> ZipEq for I {}

#[derive(Clone, Debug)]
pub struct ConstRef {
    pub const_indices: Vec<usize>,
}

pub fn get_scalar_constants<T: Copy, Ops, const D: usize>(expr: &PostfixExpr<T, Ops, D>) -> (Vec<T>, ConstRef) {
    let cref = ConstRef {
        const_indices: (0..expr.consts.len()).collect(),
    };
    (expr.consts.clone(), cref)
}

pub fn set_scalar_constants<T, Ops, const D: usize>(
    expr: &mut PostfixExpr<T, Ops, D>,
    new_values: &[T],
    cref: &ConstRef,
) where
    T: Clone,
{
    assert_eq!(new_values.len(), cref.const_indices.len());
    for (src_i, &dst_i) in cref.const_indices.iter().enumerate() {
        expr.consts[dst_i] = new_values[src_i].clone();
    }
}

pub fn compress_constants<T: Clone, Ops, const D: usize>(expr: &mut PostfixExpr<T, Ops, D>) -> bool {
    const SENTINEL: u16 = u16::MAX;
    let mut remap: Vec<u16> = vec![SENTINEL; expr.consts.len()];
    let mut new_consts: Vec<T> = Vec::with_capacity(expr.consts.len());
    let mut changed = false;

    for node in &mut expr.nodes {
        if let PNode::Const { idx } = node {
            let slot = &mut remap[*idx as usize];
            if *slot == SENTINEL {
                let new_idx = new_consts.len();
                if new_idx >= SENTINEL as usize {
                    panic!("too many constants");
                }
                *slot = new_idx as u16;
                new_consts.push(expr.consts[*idx as usize].clone());
            }
            changed |= *idx != *slot;
            *idx = *slot;
        }
    }

    changed |= new_consts.len() != expr.consts.len();
    expr.consts = new_consts;
    changed
}
