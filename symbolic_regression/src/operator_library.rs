use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::{HasOp, OperatorSet, Operators};

/// Small helpers for building common `Operators<D>` selections.
///
/// This does *not* define an operator set; it just selects from an existing `OperatorSet`
/// (such as `BuiltinOpsF32` from `dynamic_expressions`).
pub struct OperatorLibrary;

impl OperatorLibrary {
    /// Build the SymbolicRegression.jl-style default operator set (`+`, `-`, `*`, `/`) when `D >= 2`.
    ///
    /// The returned value is an [`Operators`] selection usable as [`crate::Options::operators`].
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
