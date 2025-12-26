use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::{builtin, scalar};
use dynamic_expressions::{node_utils, proptest_utils};
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::common::{D, T, TestOps};
use crate::mutation_functions::prepend_random_op_in_place;
use crate::operator_library::OperatorLibrary;

const N_FEATURES: usize = 5;
const N_CONSTS: usize = 3;

fn arb_postfix_nodes() -> impl Strategy<Value = Vec<PNode>> {
    let binary_ops = vec![
        <TestOps as scalar::HasOp<builtin::Add, 2>>::ID,
        <TestOps as scalar::HasOp<builtin::Sub, 2>>::ID,
        <TestOps as scalar::HasOp<builtin::Mul, 2>>::ID,
        <TestOps as scalar::HasOp<builtin::Div, 2>>::ID,
    ];
    proptest_utils::arb_postfix_nodes(N_FEATURES, N_CONSTS, Vec::new(), binary_ops, Vec::new(), 4, 32, 6)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn prepend_random_op_increases_depth_by_one(
        nodes in arb_postfix_nodes(),
        rng_seed in any::<u64>(),
    ) {
        let mut expr = PostfixExpr::<T, TestOps, D>::new(
            nodes,
            vec![0.0; N_CONSTS],
            Metadata::default(),
        );
        let before = node_utils::count_depth(&expr.nodes);

        let ops = OperatorLibrary::sr_default::<TestOps, D>();
        let mut rng = StdRng::seed_from_u64(rng_seed);
        prop_assert!(prepend_random_op_in_place(&mut rng, &mut expr, &ops, N_FEATURES));

        let after = node_utils::count_depth(&expr.nodes);
        prop_assert_eq!(after, before + 1);
    }
}
