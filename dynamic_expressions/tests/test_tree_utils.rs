mod common;

use common::*;
use dynamic_expressions::node_utils::{
    count_operator_nodes, count_variable_nodes, has_variables, is_leaf, is_valid_postfix, max_arity,
};
use dynamic_expressions::{
    count_constant_nodes, count_depth, count_nodes, has_constants, has_operators, subtree_range,
    subtree_sizes, tree_mapreduce,
};

#[test]
fn depth_and_counts_match_expected() {
    let expr = expr_readme_like();
    let nodes = &expr.nodes;

    assert_eq!(count_nodes(nodes), nodes.len());
    assert!(has_constants(nodes));
    assert_eq!(count_constant_nodes(nodes), 1);
    assert!(has_operators(nodes));
    assert!(has_variables(nodes));
    assert_eq!(count_variable_nodes(nodes), 2);
    assert_eq!(count_operator_nodes(nodes), 3);
    assert_eq!(max_arity(nodes), 2);
    assert!(is_valid_postfix(nodes));
    assert!(!is_leaf(nodes));

    // x1 * cos(x2 - c0) has depth 4 under leaf-depth=1.
    assert_eq!(count_depth(nodes), 4);

    let depth_via_fold = tree_mapreduce(
        nodes,
        |_| 1usize,
        |_| 1usize,
        |p, ch| p + ch.iter().copied().max().unwrap_or(0),
    );
    assert_eq!(depth_via_fold, 4);
}

#[test]
fn subtree_sizes_and_ranges_are_correct() {
    let expr = expr_readme_like();
    let nodes = &expr.nodes;

    let sizes = subtree_sizes(nodes);
    assert_eq!(sizes.len(), nodes.len());

    // Root is last node.
    assert_eq!(sizes[nodes.len() - 1], nodes.len());

    // In expr_readme_like:
    // 0: x1
    // 1: x2
    // 2: c0
    // 3: Sub(x2, c0) => size 3, range [1..=3]
    // 4: Cos(Sub(...)) => size 4, range [1..=4]
    // 5: Mul(x1, Cos(...)) => size 6, range [0..=5]
    assert_eq!(sizes[3], 3);
    assert_eq!(subtree_range(&sizes, 3), (1, 3));
    assert_eq!(sizes[4], 4);
    assert_eq!(subtree_range(&sizes, 4), (1, 4));
    assert_eq!(subtree_range(&sizes, 5), (0, 5));
}

#[test]
fn leaf_has_no_operators_or_constants() {
    let leaf = var(0);
    assert!(!has_operators(&leaf.nodes));
    assert!(!has_constants(&leaf.nodes));
    assert_eq!(count_constant_nodes(&leaf.nodes), 0);
    assert_eq!(count_depth(&leaf.nodes), 1);
    assert!(has_variables(&leaf.nodes));
    assert_eq!(count_variable_nodes(&leaf.nodes), 1);
    assert_eq!(count_operator_nodes(&leaf.nodes), 0);
    assert_eq!(max_arity(&leaf.nodes), 0);
    assert!(is_valid_postfix(&leaf.nodes));
    assert!(is_leaf(&leaf.nodes));
}

#[test]
fn invalid_postfix_is_detected() {
    // Op arity needs operands.
    let bad = vec![dynamic_expressions::PNode::Op { arity: 2, op: 0 }];
    assert!(!is_valid_postfix(&bad));

    // Extra leaves left on stack.
    let bad2 = vec![
        dynamic_expressions::PNode::Var { feature: 0 },
        dynamic_expressions::PNode::Var { feature: 0 },
    ];
    assert!(!is_valid_postfix(&bad2));
}
