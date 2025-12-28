use dynamic_expressions::HasOp;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use fastrand::Rng;

#[test]
fn swap_operands_supports_arity_gt2() {
    type Ops = BuiltinOpsF64;
    const D: usize = 3;

    let fma = <Ops as HasOp<builtin::Fma>>::op_id();
    let mut expr = PostfixExpr::<f64, Ops, D>::new(
        vec![
            PNode::Var { feature: 0u16 },
            PNode::Var { feature: 1u16 },
            PNode::Var { feature: 2u16 },
            PNode::Op {
                arity: fma.arity,
                op: fma.id,
            },
        ],
        Vec::new(),
        Metadata::default(),
    );

    let before = expr.nodes.clone();
    let mut rng = Rng::with_seed(0);
    assert!(crate::mutation_functions::swap_operands_in_place(&mut rng, &mut expr));

    assert_eq!(expr.nodes.len(), before.len());
    assert_eq!(expr.nodes.last().unwrap(), &before[3]);
    assert_ne!(expr.nodes, before);
}
