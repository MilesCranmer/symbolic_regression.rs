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

fn child_ranges(sizes: &[usize], root_idx: usize, arity: usize) -> Vec<(usize, usize)> {
    let mut out = vec![(0usize, 0usize); arity];
    let mut end: isize = root_idx as isize - 1;
    for k in (0..arity).rev() {
        let end_u = usize::try_from(end).expect("invalid postfix (child end underflow)");
        let sz = sizes[end_u];
        let start_u = end_u + 1 - sz;
        out[k] = (start_u, end_u);
        end = start_u as isize - 1;
    }
    out
}

fn rotate_tree_in_place_reference<T, Ops, const D: usize>(rng: &mut Rng, expr: &mut PostfixExpr<T, Ops, D>) -> bool {
    let sizes = dynamic_expressions::node_utils::subtree_sizes(&expr.nodes);
    let mut valid_roots: Vec<usize> = Vec::new();
    for (i, n) in expr.nodes.iter().enumerate() {
        let PNode::Op { arity, .. } = *n else {
            continue;
        };
        let a = arity as usize;
        if a == 0 {
            continue;
        }
        let children = child_ranges(&sizes, i, a);
        if children.iter().any(|c| matches!(expr.nodes[c.1], PNode::Op { .. })) {
            valid_roots.push(i);
        }
    }
    if valid_roots.is_empty() {
        return false;
    }

    let root_idx = valid_roots[rng.usize(0..valid_roots.len())];
    let PNode::Op {
        arity: root_arity_u8,
        op: op_root,
    } = expr.nodes[root_idx]
    else {
        return false;
    };
    let root_arity = root_arity_u8 as usize;
    if root_arity == 0 {
        return false;
    }
    let root_children = child_ranges(&sizes, root_idx, root_arity);

    let pivot_positions: Vec<usize> = root_children
        .iter()
        .enumerate()
        .filter_map(|(j, c)| matches!(expr.nodes[c.1], PNode::Op { .. }).then_some(j))
        .collect();
    if pivot_positions.is_empty() {
        return false;
    }

    let pivot_pos = pivot_positions[rng.usize(0..pivot_positions.len())];
    let pivot_root_idx = root_children[pivot_pos].1;
    let PNode::Op {
        arity: pivot_arity_u8,
        op: op_pivot,
    } = expr.nodes[pivot_root_idx]
    else {
        return false;
    };
    let pivot_arity = pivot_arity_u8 as usize;
    if pivot_arity == 0 {
        return false;
    }
    let pivot_children = child_ranges(&sizes, pivot_root_idx, pivot_arity);

    let grandchild_pos = rng.usize(0..pivot_arity);
    let grandchild = pivot_children[grandchild_pos];

    let (sub_start, sub_end) = dynamic_expressions::node_utils::subtree_range(&sizes, root_idx);

    let mut rotated_root: Vec<PNode> = Vec::with_capacity(sub_end + 1 - sub_start);
    for (j, c) in root_children.iter().enumerate() {
        if j == pivot_pos {
            rotated_root.extend_from_slice(&expr.nodes[grandchild.0..=grandchild.1]);
        } else {
            rotated_root.extend_from_slice(&expr.nodes[c.0..=c.1]);
        }
    }
    rotated_root.push(PNode::Op {
        arity: root_arity_u8,
        op: op_root,
    });

    let mut new_sub: Vec<PNode> = Vec::with_capacity(sub_end + 1 - sub_start);
    for (k, c) in pivot_children.iter().enumerate() {
        if k == grandchild_pos {
            new_sub.extend_from_slice(&rotated_root);
        } else {
            new_sub.extend_from_slice(&expr.nodes[c.0..=c.1]);
        }
    }
    new_sub.push(PNode::Op {
        arity: pivot_arity_u8,
        op: op_pivot,
    });

    expr.nodes.splice(sub_start..=sub_end, new_sub);
    true
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

    #[test]
    fn rotate_tree_in_place_matches_reference(
        nodes in arb_rotatable_tree_nodes(),
        rng_seed in any::<u64>(),
    ) {
        let mut expr_ref = PostfixExpr::<f64, (), D_TEST>::new(
            nodes.clone(),
            vec![0.0, 1.0, 2.0],
            Metadata::default(),
        );
        let mut expr_fast = expr_ref.clone();

        let mut rng_ref = Rng::with_seed(rng_seed);
        let mut rng_fast = Rng::with_seed(rng_seed);

        prop_assert!(rotate_tree_in_place_reference(&mut rng_ref, &mut expr_ref));
        prop_assert!(rotate_tree_in_place(&mut rng_fast, &mut expr_fast));
        prop_assert_eq!(expr_ref.nodes, expr_fast.nodes);
    }
}
