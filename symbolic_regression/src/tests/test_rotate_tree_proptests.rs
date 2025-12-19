use crate::mutate::rotate_tree_in_place;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils::is_valid_postfix;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::BTreeMap;

const N_FEATURES: usize = 5;
const N_CONSTS: usize = 3;
const D_TEST: usize = 3;

#[derive(Clone, Debug)]
enum GenTree {
    Var(u16),
    Const(u16),
    Op {
        arity: u8,
        op: u16,
        children: Vec<GenTree>,
    },
}

impl GenTree {
    fn to_postfix(&self, out: &mut Vec<PNode>) {
        match self {
            GenTree::Var(feature) => out.push(PNode::Var { feature: *feature }),
            GenTree::Const(idx) => out.push(PNode::Const { idx: *idx }),
            GenTree::Op {
                arity,
                op,
                children,
            } => {
                for c in children {
                    c.to_postfix(out);
                }
                out.push(PNode::Op {
                    arity: *arity,
                    op: *op,
                });
            }
        }
    }
}

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

fn arb_subtree() -> impl Strategy<Value = GenTree> {
    let leaf = prop_oneof![
        (0u16..(N_FEATURES as u16)).prop_map(GenTree::Var),
        (0u16..(N_CONSTS as u16)).prop_map(GenTree::Const),
    ];

    leaf.prop_recursive(6, 32, 10, |inner| {
        prop_oneof![
            (
                Just(1u8),
                0u16..20u16,
                prop::collection::vec(inner.clone(), 1)
            )
                .prop_map(|(arity, op, children)| GenTree::Op {
                    arity,
                    op,
                    children
                }),
            (
                Just(2u8),
                0u16..20u16,
                prop::collection::vec(inner.clone(), 2)
            )
                .prop_map(|(arity, op, children)| GenTree::Op {
                    arity,
                    op,
                    children
                }),
            (Just(3u8), 0u16..20u16, prop::collection::vec(inner, 3)).prop_map(
                |(arity, op, children)| GenTree::Op {
                    arity,
                    op,
                    children
                }
            ),
        ]
    })
}

fn arb_rotatable_tree_nodes() -> impl Strategy<Value = Vec<PNode>> {
    // Start with a valid tree, then filter to ones our implementation can rotate.
    // (We prioritize simpler test code over generator efficiency.)
    arb_subtree()
        .prop_map(|t| {
            let mut nodes = Vec::new();
            t.to_postfix(&mut nodes);
            nodes
        })
        .prop_filter("rotatable tree", |nodes| {
            let mut expr = PostfixExpr::<f64, (), D_TEST>::new(
                nodes.clone(),
                vec![0.0; N_CONSTS],
                Metadata::default(),
            );
            let mut rng = StdRng::seed_from_u64(0);
            rotate_tree_in_place(&mut rng, &mut expr)
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

    let mut rng = StdRng::seed_from_u64(0);
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
    #![proptest_config(ProptestConfig {
        // Rotatable trees can be somewhat rare under a generic tree generator.
        // Increase the global reject budget so we don't flake.
        max_global_rejects: 100_000,
        .. ProptestConfig::default()
    })]

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

        let mut rng = StdRng::seed_from_u64(rng_seed);
        prop_assert!(rotate_tree_in_place(&mut rng, &mut expr));

        let after = &expr.nodes;
        prop_assert!(is_valid_postfix(after));
        let _plan = dynamic_expressions::compile_plan::<D_TEST>(after, N_FEATURES, N_CONSTS);

        // Node multiset must be preserved.
        prop_assert_eq!(mult_before, node_multiset(after));
        prop_assert_eq!(nodes.len(), after.len());

        // Rotation must not touch constants or metadata.
        prop_assert_eq!(before_consts, expr.consts);
        prop_assert_eq!(before_meta.variable_names, expr.meta.variable_names);
    }
}
