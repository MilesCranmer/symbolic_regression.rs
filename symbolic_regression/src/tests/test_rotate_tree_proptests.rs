use std::collections::BTreeMap;

use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::proptest_utils;
use fastrand::Rng;
use proptest::prelude::*;

use crate::mutation_functions::rotate_tree_in_place;

const N_FEATURES: usize = 5;
const N_CONSTS: usize = 3;
const D_TEST: usize = 3;

fn node_multiset(nodes: &[PNode]) -> BTreeMap<(u8, u16, u16), usize> {
    // (tag, a, b) -> count, where:
    // - Var:   (0, feature, 0)
    // - Const: (1, idx, 0)
    // - Op:    (2, arity, op)
    let mut m: BTreeMap<(u8, u16, u16), usize> = BTreeMap::new();
    for n in nodes {
        let k = match *n {
            PNode::Var { feature } => (0, feature, 0),
            PNode::Const { idx } => (1, idx, 0),
            PNode::Op { arity, op } => (2, arity as u16, op),
        };
        *m.entry(k).or_insert(0) += 1;
    }
    m
}

fn arb_rotatable_tree_nodes() -> impl Strategy<Value = Vec<PNode>> {
    let op_ids: Vec<u16> = (0u16..20u16).collect();
    (1usize..=D_TEST).prop_flat_map(move |root_arity| {
        let op_ids = op_ids.clone();
        (
            Just(root_arity),
            0usize..root_arity,
            1usize..=D_TEST,
            prop::sample::select(op_ids.clone()),
            prop::sample::select(op_ids.clone()),
        )
            .prop_flat_map(move |(root_arity, pivot_pos, pivot_arity, op_root, op_pivot)| {
                let pivot_children = prop::collection::vec(
                    proptest_utils::arb_shallow_postfix_nodes(N_FEATURES, N_CONSTS, &op_ids, &op_ids, true),
                    pivot_arity,
                );
                let other_children = prop::collection::vec(
                    proptest_utils::arb_shallow_postfix_nodes(N_FEATURES, N_CONSTS, &op_ids, &op_ids, true),
                    root_arity - 1,
                );
                (
                    Just((root_arity, pivot_pos, pivot_arity, op_root, op_pivot)),
                    pivot_children,
                    other_children,
                )
            })
            .prop_map(
                |((root_arity, pivot_pos, pivot_arity, op_root, op_pivot), pivot_children, other_children)| {
                    let mut nodes = Vec::new();
                    let mut other_iter = other_children.into_iter();
                    for j in 0..root_arity {
                        if j == pivot_pos {
                            for child in &pivot_children {
                                nodes.extend_from_slice(child);
                            }
                            nodes.push(PNode::Op {
                                arity: pivot_arity as u8,
                                op: op_pivot,
                            });
                        } else {
                            let child = other_iter.next().expect("missing non-pivot child");
                            nodes.extend(child);
                        }
                    }
                    nodes.push(PNode::Op {
                        arity: root_arity as u8,
                        op: op_root,
                    });
                    nodes
                },
            )
    })
}

#[test]
fn rotate_tree_supports_non_binary_arity() {
    // Root has arity 3 and its middle child is a unary operator:
    //   root(A, pivot(B), C)
    // After rotation:
    //   pivot(root(A, B, C))
    let mut expr = PostfixExpr::<f64, (), 3>::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Var { feature: 1 },
            PNode::Op { arity: 1, op: 11 },
            PNode::Var { feature: 2 },
            PNode::Op { arity: 3, op: 7 },
        ],
        vec![0.0; N_CONSTS],
        Metadata::default(),
    );

    let mut rng = Rng::with_seed(0);
    assert!(rotate_tree_in_place(&mut rng, &mut expr));
    assert_eq!(
        expr.nodes,
        vec![
            PNode::Var { feature: 0 },
            PNode::Var { feature: 1 },
            PNode::Var { feature: 2 },
            PNode::Op { arity: 3, op: 7 },
            PNode::Op { arity: 1, op: 11 },
        ]
    );
}

proptest! {
    #[test]
    fn rotate_tree_in_place_preserves_invariants(
        nodes in arb_rotatable_tree_nodes(),
        rng_seed in any::<u64>(),
    ) {
        let mult_before = node_multiset(&nodes);

        let mut expr = PostfixExpr::<f64, (), D_TEST>::new(
            nodes.clone(),
            vec![0.0, 1.0, 2.0],
            Metadata::default(),
        );
        let before_consts = expr.consts.clone();
        let before_meta = expr.meta.clone();

        let mut rng = Rng::with_seed(rng_seed);
        prop_assert!(rotate_tree_in_place(&mut rng, &mut expr));

        let after = &expr.nodes;
        prop_assert!(dynamic_expressions::node_utils::is_valid_postfix(after));
        let _plan = dynamic_expressions::compile_plan::<D_TEST>(after, N_FEATURES, N_CONSTS);

        // Node multiset must be preserved.
        prop_assert_eq!(mult_before, node_multiset(after));
        prop_assert_eq!(nodes.len(), after.len());

        // Rotation must not touch constants or metadata.
        prop_assert_eq!(before_consts, expr.consts);
        prop_assert_eq!(before_meta.variable_names, expr.meta.variable_names);
    }
}
