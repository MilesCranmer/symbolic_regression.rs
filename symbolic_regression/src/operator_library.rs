use crate::operators::{OpSpec, Operators};
use dynamic_expressions::operator_enum::builtin::{Add, Div, Mul, Sub};
use dynamic_expressions::operator_enum::scalar::{HasOp, OpId};

pub struct OperatorLibrary;

impl OperatorLibrary {
    pub fn sr_default<Ops, const D: usize>() -> Operators<D>
    where
        Ops: HasOp<Add, 2> + HasOp<Sub, 2> + HasOp<Mul, 2> + HasOp<Div, 2>,
    {
        let mut ops = Operators::<D>::new();
        if D >= 2 {
            let list = [
                OpId {
                    arity: 2,
                    id: <Ops as HasOp<Add, 2>>::ID,
                },
                OpId {
                    arity: 2,
                    id: <Ops as HasOp<Sub, 2>>::ID,
                },
                OpId {
                    arity: 2,
                    id: <Ops as HasOp<Mul, 2>>::ID,
                },
                OpId {
                    arity: 2,
                    id: <Ops as HasOp<Div, 2>>::ID,
                },
            ];
            for op in list {
                ops.push(
                    2,
                    OpSpec {
                        op,
                        commutative: op.id == <Ops as HasOp<Add, 2>>::ID
                            || op.id == <Ops as HasOp<Mul, 2>>::ID,
                        associative: op.id == <Ops as HasOp<Add, 2>>::ID
                            || op.id == <Ops as HasOp<Mul, 2>>::ID,
                        complexity: 1,
                    },
                );
            }
        }
        ops
    }
}
