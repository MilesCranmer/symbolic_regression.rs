use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::{HasOp, OperatorSet};

use crate::operators::Operators;

pub struct OperatorLibrary;

impl OperatorLibrary {
    pub fn sr_default<Ops, const D: usize>() -> Operators<D>
    where
        Ops: HasOp<builtin::Add> + HasOp<builtin::Sub> + HasOp<builtin::Mul> + HasOp<builtin::Div> + OperatorSet,
    {
        let mut ops = Operators::<D>::new();
        if D >= 2 {
            for op in [
                <Ops as HasOp<builtin::Add>>::op_id(),
                <Ops as HasOp<builtin::Sub>>::op_id(),
                <Ops as HasOp<builtin::Mul>>::op_id(),
                <Ops as HasOp<builtin::Div>>::op_id(),
            ] {
                ops.push(op);
            }
        }
        ops
    }
}
