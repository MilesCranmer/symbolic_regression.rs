use crate::expression::PostfixExpr;
use crate::node::PNode;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct ConstRef {
    pub const_indices: Vec<usize>,
}

pub fn get_scalar_constants<T: Copy, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
) -> (Vec<T>, ConstRef) {
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

pub fn compress_constants<T: Clone, Ops, const D: usize>(
    expr: &mut PostfixExpr<T, Ops, D>,
) -> bool {
    let mut remap: HashMap<u16, u16> = HashMap::new();
    let mut new_consts: Vec<T> = Vec::new();
    let mut changed = false;
    for node in &mut expr.nodes {
        let PNode::Const { idx } = node else {
            continue;
        };

        let old_idx = *idx;
        let new_idx = if let Some(&v) = remap.get(idx) {
            v
        } else {
            let old_i = *idx as usize;
            let new_i = new_consts.len();
            if new_i > u16::MAX as usize {
                // Preserve current expression; this is a hard limit in the node encoding.
                return false;
            }
            new_consts.push(expr.consts[old_i].clone());
            let v = new_i as u16;
            remap.insert(*idx, v);
            v
        };

        *idx = new_idx;
        if new_idx != old_idx {
            changed = true;
        }
    }
    if new_consts.len() != expr.consts.len() {
        changed = true;
    }
    expr.consts = new_consts;
    changed
}
