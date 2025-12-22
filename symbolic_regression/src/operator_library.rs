use dynamic_expressions::operator_enum::{builtin, scalar};

use crate::operators::{OpSpec, Operators};

pub struct OperatorLibrary;

impl OperatorLibrary {
    pub fn sr_default<Ops, const D: usize>() -> Operators<D>
    where
        Ops: scalar::HasOp<builtin::Add, 2>
            + scalar::HasOp<builtin::Sub, 2>
            + scalar::HasOp<builtin::Mul, 2>
            + scalar::HasOp<builtin::Div, 2>,
    {
        let mut ops = Operators::<D>::new();
        if D >= 2 {
            let list = [
                scalar::OpId {
                    arity: 2,
                    id: <Ops as scalar::HasOp<builtin::Add, 2>>::ID,
                },
                scalar::OpId {
                    arity: 2,
                    id: <Ops as scalar::HasOp<builtin::Sub, 2>>::ID,
                },
                scalar::OpId {
                    arity: 2,
                    id: <Ops as scalar::HasOp<builtin::Mul, 2>>::ID,
                },
                scalar::OpId {
                    arity: 2,
                    id: <Ops as scalar::HasOp<builtin::Div, 2>>::ID,
                },
            ];
            for op in list {
                ops.push(
                    2,
                    OpSpec {
                        op,
                        commutative: op.id == <Ops as scalar::HasOp<builtin::Add, 2>>::ID
                            || op.id == <Ops as scalar::HasOp<builtin::Mul, 2>>::ID,
                        associative: op.id == <Ops as scalar::HasOp<builtin::Add, 2>>::ID
                            || op.id == <Ops as scalar::HasOp<builtin::Mul, 2>>::ID,
                        complexity: 1,
                    },
                );
            }
        }
        ops
    }
}
