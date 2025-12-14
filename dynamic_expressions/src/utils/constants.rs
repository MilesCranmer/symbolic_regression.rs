use crate::expr::PostfixExpr;

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
