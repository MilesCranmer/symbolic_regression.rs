use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils;
use fastrand::Rng;
use num_traits::Float;

use crate::operators::Operators;
use crate::options::Options;
use crate::random::{standard_normal, usize_range, usize_range_excl};

fn random_leaf<T: Float>(rng: &mut Rng, n_features: usize, consts: &mut Vec<T>) -> PNode {
    if rng.bool() {
        let val_f64: f64 = standard_normal(rng);
        let val = T::from(val_f64).unwrap();
        let idx: u16 = consts
            .len()
            .try_into()
            .unwrap_or_else(|_| panic!("too many constants to index in u16"));
        consts.push(val);
        PNode::Const { idx }
    } else {
        let f: u16 = usize_range(rng, 0..n_features)
            .try_into()
            .unwrap_or_else(|_| panic!("too many features to index in u16"));
        PNode::Var { feature: f }
    }
}

pub fn random_expr<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    operators: &Operators<D>,
    n_features: usize,
    target_size: usize,
) -> PostfixExpr<T, Ops, D> {
    assert!(target_size >= 1);
    let mut nodes: Vec<PNode> = Vec::with_capacity(target_size);
    let mut consts: Vec<T> = Vec::new();
    nodes.push(random_leaf(rng, n_features, &mut consts));

    while nodes.len() < target_size && operators.total_ops_up_to(D.min(target_size - nodes.len())) > 0 {
        let rem = target_size - nodes.len();
        let max_arity = rem.min(D);
        let arity = operators.sample_arity(rng, max_arity);
        let op_id = operators.sample_op(rng, arity).id;

        let leaf_positions: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
            .collect();
        let leaf_idx = leaf_positions[usize_range(rng, 0..leaf_positions.len())];

        let mut repl: Vec<PNode> = Vec::with_capacity(arity + 1);
        for _ in 0..arity {
            repl.push(random_leaf(rng, n_features, &mut consts));
        }
        repl.push(PNode::Op {
            arity: arity as u8,
            op: op_id,
        });
        nodes.splice(leaf_idx..=leaf_idx, repl);
    }

    PostfixExpr::new(nodes, consts, Default::default())
}

/// Match SymbolicRegression.jl `gen_random_tree(nlength, ...)`:
/// start from a placeholder `init_value(T)` leaf (0 for numeric types), then
/// do `n_append_ops` rounds of `append_random_op`, which expands a random leaf
/// by replacing it with an operator node and fresh random leaf children.
///
/// Note: final node count is generally larger than `n_append_ops`.
pub fn random_expr_append_ops<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    operators: &Operators<D>,
    n_features: usize,
    n_append_ops: usize,
    max_size: usize,
) -> PostfixExpr<T, Ops, D> {
    let max_size = max_size.max(1);
    let mut expr = PostfixExpr::<T, Ops, D>::zero();

    for _ in 0..n_append_ops {
        let rem = max_size.saturating_sub(expr.nodes.len());
        if rem == 0 {
            break;
        }
        let max_arity = rem.min(D);
        if operators.total_ops_up_to(max_arity) == 0 {
            break;
        }
        let arity = operators.sample_arity(rng, max_arity);
        let op_id = operators.sample_op(rng, arity).id;

        let leaf_positions: Vec<usize> = expr
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
            .collect();
        if leaf_positions.is_empty() {
            break;
        }
        let leaf_idx = leaf_positions[usize_range(rng, 0..leaf_positions.len())];

        let mut repl: Vec<PNode> = Vec::with_capacity(arity + 1);
        for _ in 0..arity {
            repl.push(random_leaf(rng, n_features, &mut expr.consts));
        }
        repl.push(PNode::Op {
            arity: arity as u8,
            op: op_id,
        });
        expr.nodes.splice(leaf_idx..=leaf_idx, repl);
    }

    dynamic_expressions::compress_constants(&mut expr);
    expr
}

fn op_indices(nodes: &[PNode]) -> Vec<usize> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| matches!(n, PNode::Op { .. }).then_some(i))
        .collect()
}

fn const_node_indices(nodes: &[PNode]) -> Vec<usize> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| matches!(n, PNode::Const { .. }).then_some(i))
        .collect()
}

fn var_node_indices(nodes: &[PNode]) -> Vec<usize> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| matches!(n, PNode::Var { .. }).then_some(i))
        .collect()
}

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

pub(crate) fn mutate_constant_in_place<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    temperature: f64,
    options: &Options<T, D>,
) -> bool {
    let idxs = const_node_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let node_i = idxs[usize_range(rng, 0..idxs.len())];
    let PNode::Const { idx } = expr.nodes[node_i] else {
        return false;
    };
    let ci = usize::from(idx);

    // Follows SymbolicRegression.jl's `mutate_factor`.
    let pf = options.perturbation_factor * temperature.max(0.0);
    let max_change = pf + 1.1;
    let exponent: f64 = rng.f64();
    let mut mul = max_change.powf(exponent);
    let make_const_bigger: bool = rng.bool();
    mul = if make_const_bigger { mul } else { 1.0 / mul };
    if rng.f64() > options.probability_negate_constant {
        mul = -mul;
    }
    expr.consts[ci] = expr.consts[ci] * T::from(mul).unwrap();
    true
}

pub(crate) fn mutate_operator_in_place<T, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
) -> bool {
    let idxs = op_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let i = idxs[usize_range(rng, 0..idxs.len())];
    let PNode::Op { arity, .. } = expr.nodes[i] else {
        return false;
    };
    let a = arity as usize;
    if operators.nops(a) == 0 {
        return false;
    }

    // Match SymbolicRegression.jl: sample uniformly among all operators of the same arity,
    // including the current one (i.e., this mutation can be a no-op).
    let new_op = operators.sample_op(rng, a).id;
    expr.nodes[i] = PNode::Op { arity, op: new_op };
    true
}

pub(crate) fn mutate_feature_in_place<T, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    n_features: usize,
) {
    if n_features <= 1 {
        return;
    }
    let idxs = var_node_indices(&expr.nodes);
    if idxs.is_empty() {
        return;
    }
    let node_i = rng.choice(idxs).unwrap();
    let PNode::Var { ref mut feature } = &mut expr.nodes[node_i] else {
        unreachable!("expected var node");
    };
    let old = usize::from(*feature);
    assert!(old < n_features, "feature index out of bounds");
    let new_feature = usize_range_excl(rng, 0..n_features, old);
    *feature = new_feature as u16;
}

pub(crate) fn is_swappable_op(node: &PNode) -> bool {
    matches!(node, PNode::Op { arity, .. } if *arity > 1)
}

pub(crate) fn swap_operands_in_place<T, Ops, const D: usize>(rng: &mut Rng, expr: &mut PostfixExpr<T, Ops, D>) -> bool {
    let idxs: Vec<usize> = expr
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| is_swappable_op(n).then_some(i))
        .collect();
    if idxs.is_empty() {
        return false;
    }
    let sizes = node_utils::subtree_sizes(&expr.nodes);
    let root_idx = rng.choice(idxs).unwrap();
    let PNode::Op { op, arity: arity_u8 } = expr.nodes[root_idx] else {
        unreachable!("expected swappable op");
    };
    let arity = arity_u8 as usize;
    let (sub_start, sub_end) = node_utils::subtree_range(&sizes, root_idx);
    let child = child_ranges(&sizes, root_idx, arity);

    // Choose two distinct child slots uniformly and swap them.
    let i = usize_range(rng, 0..arity);
    let j = usize_range_excl(rng, 0..arity, i);

    let mut positions: Vec<usize> = (0..arity).collect();
    positions.swap(i, j);
    let mut new_sub: Vec<PNode> = Vec::with_capacity(sub_end + 1 - sub_start);
    for pos in positions {
        new_sub.extend_from_slice(&expr.nodes[child[pos].0..=child[pos].1]);
    }
    new_sub.push(PNode::Op { arity: arity as u8, op });
    expr.nodes.splice(sub_start..=sub_end, new_sub);
    true
}

pub fn rotate_tree_in_place<T, Ops, const D: usize>(rng: &mut Rng, expr: &mut PostfixExpr<T, Ops, D>) -> bool {
    // Match SymbolicRegression.jl's `randomly_rotate_tree!`:
    // pick a random rotation root where some child is an operator, then
    // rotate along a random internal edge (root -> pivot) using a random grandchild.
    let sizes = node_utils::subtree_sizes(&expr.nodes);
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

    let root_idx = valid_roots[usize_range(rng, 0..valid_roots.len())];
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

    let pivot_pos = pivot_positions[usize_range(rng, 0..pivot_positions.len())];
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

    let grandchild_pos = usize_range(rng, 0..pivot_arity);
    let grandchild = pivot_children[grandchild_pos];

    let (sub_start, sub_end) = node_utils::subtree_range(&sizes, root_idx);

    // Build the rotated version of the old root, with its `pivot_pos` child replaced by `grandchild`.
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

    // Build the new subtree rooted at `pivot`, replacing its `grandchild_pos` with `rotated_root`.
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

pub fn insert_random_op_in_place<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
    n_features: usize,
) -> bool {
    if expr.nodes.is_empty() {
        return false;
    }
    if operators.total_ops_up_to(D) == 0 {
        return false;
    }
    let root_idx = usize_range(rng, 0..expr.nodes.len());
    let sizes = node_utils::subtree_sizes(&expr.nodes);
    let (start, end) = node_utils::subtree_range(&sizes, root_idx);
    let old_sub: Vec<PNode> = expr.nodes[start..=end].to_vec();

    let arity = operators.sample_arity(rng, D);
    let op_id = operators.sample_op(rng, arity).id;
    let carry_pos = usize_range(rng, 0..arity);

    let mut new_sub: Vec<PNode> = Vec::new();
    for j in 0..arity {
        if j == carry_pos {
            new_sub.extend_from_slice(&old_sub);
        } else {
            new_sub.push(random_leaf(rng, n_features, &mut expr.consts));
        }
    }
    new_sub.push(PNode::Op {
        arity: arity as u8,
        op: op_id,
    });
    expr.nodes.splice(start..=end, new_sub);
    dynamic_expressions::compress_constants(expr);
    true
}

pub(crate) fn prepend_random_op_in_place<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
    n_features: usize,
) -> bool {
    if expr.nodes.is_empty() {
        return false;
    }
    if operators.total_ops_up_to(D) == 0 {
        return false;
    }
    let arity = operators.sample_arity(rng, D);
    let op_id = operators.sample_op(rng, arity).id;
    let carry_pos = usize_range(rng, 0..arity);

    let old = expr.nodes.clone();
    let mut new_nodes: Vec<PNode> = Vec::new();
    for j in 0..arity {
        if j == carry_pos {
            new_nodes.extend_from_slice(&old);
        } else {
            new_nodes.push(random_leaf(rng, n_features, &mut expr.consts));
        }
    }
    new_nodes.push(PNode::Op {
        arity: arity as u8,
        op: op_id,
    });
    expr.nodes = new_nodes;
    dynamic_expressions::compress_constants(expr);
    true
}

pub(crate) fn append_random_op_in_place<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
    n_features: usize,
) -> bool {
    if expr.nodes.is_empty() {
        return false;
    }
    if operators.total_ops_up_to(D) == 0 {
        return false;
    }

    let leaf_positions: Vec<usize> = expr
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
        .collect();
    if leaf_positions.is_empty() {
        return false;
    }
    let leaf_idx = leaf_positions[usize_range(rng, 0..leaf_positions.len())];

    let arity = operators.sample_arity(rng, D);
    let op_id = operators.sample_op(rng, arity).id;

    let mut replace_with: Vec<PNode> = Vec::with_capacity(arity + 1);
    for _ in 0..arity {
        replace_with.push(random_leaf(rng, n_features, &mut expr.consts));
    }
    replace_with.push(PNode::Op {
        arity: arity as u8,
        op: op_id,
    });

    expr.nodes.splice(leaf_idx..=leaf_idx, replace_with);
    dynamic_expressions::compress_constants(expr);
    true
}

pub(crate) fn add_node_in_place<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
    n_features: usize,
) -> bool {
    if rng.bool() {
        append_random_op_in_place(rng, expr, operators, n_features)
    } else {
        prepend_random_op_in_place(rng, expr, operators, n_features)
    }
}

pub(crate) fn delete_random_op_in_place<T: Clone, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut PostfixExpr<T, Ops, D>,
) -> bool {
    let idxs = op_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let root_idx = idxs[usize_range(rng, 0..idxs.len())];
    let PNode::Op { arity, .. } = expr.nodes[root_idx] else {
        return false;
    };
    let a = arity as usize;
    if a == 0 {
        return false;
    }
    let sizes = node_utils::subtree_sizes(&expr.nodes);
    let (sub_start, sub_end) = node_utils::subtree_range(&sizes, root_idx);
    if sub_start == sub_end {
        return false;
    }
    let child = child_ranges(&sizes, root_idx, a);
    let keep = child[usize_range(rng, 0..a)];
    let kept_nodes: Vec<PNode> = expr.nodes[keep.0..=keep.1].to_vec();
    expr.nodes.splice(sub_start..=sub_end, kept_nodes);
    dynamic_expressions::compress_constants(expr);
    true
}

pub(crate) fn crossover_trees<T: Clone, Ops, const D: usize>(
    rng: &mut Rng,
    a: &PostfixExpr<T, Ops, D>,
    b: &PostfixExpr<T, Ops, D>,
) -> (PostfixExpr<T, Ops, D>, PostfixExpr<T, Ops, D>) {
    fn remap_subtree_consts<T: Clone>(
        donor_nodes: &[PNode],
        donor_consts: &[T],
        dst_consts: &mut Vec<T>,
    ) -> Vec<PNode> {
        let mut map: Vec<Option<u16>> = vec![None; donor_consts.len()];
        let mut out: Vec<PNode> = Vec::with_capacity(donor_nodes.len());
        for n in donor_nodes {
            match *n {
                PNode::Const { idx } => {
                    let old = usize::from(idx);
                    let new_idx = match map[old] {
                        Some(v) => v,
                        None => {
                            let v: u16 = dst_consts
                                .len()
                                .try_into()
                                .unwrap_or_else(|_| panic!("too many constants to index in u16"));
                            dst_consts.push(donor_consts[old].clone());
                            map[old] = Some(v);
                            v
                        }
                    };
                    out.push(PNode::Const { idx: new_idx });
                }
                PNode::Var { feature } => out.push(PNode::Var { feature }),
                PNode::Op { arity, op } => out.push(PNode::Op { arity, op }),
            }
        }
        out
    }

    let a_sizes = node_utils::subtree_sizes(&a.nodes);
    let b_sizes = node_utils::subtree_sizes(&b.nodes);
    let a_root = usize_range(rng, 0..a.nodes.len());
    let b_root = usize_range(rng, 0..b.nodes.len());
    let (a_start, a_end) = node_utils::subtree_range(&a_sizes, a_root);
    let (b_start, b_end) = node_utils::subtree_range(&b_sizes, b_root);

    let a_sub = &a.nodes[a_start..=a_end];
    let b_sub = &b.nodes[b_start..=b_end];

    let mut child_a_nodes: Vec<PNode> = Vec::with_capacity(a.nodes.len() - a_sub.len() + b_sub.len());
    child_a_nodes.extend_from_slice(&a.nodes[..a_start]);
    let mut child_a_consts = a.consts.clone();
    let b_sub_remap = remap_subtree_consts(b_sub, &b.consts, &mut child_a_consts);
    child_a_nodes.extend_from_slice(&b_sub_remap);
    child_a_nodes.extend_from_slice(&a.nodes[a_end + 1..]);

    let mut child_b_nodes: Vec<PNode> = Vec::with_capacity(b.nodes.len() - b_sub.len() + a_sub.len());
    child_b_nodes.extend_from_slice(&b.nodes[..b_start]);
    let mut child_b_consts = b.consts.clone();
    let a_sub_remap = remap_subtree_consts(a_sub, &a.consts, &mut child_b_consts);
    child_b_nodes.extend_from_slice(&a_sub_remap);
    child_b_nodes.extend_from_slice(&b.nodes[b_end + 1..]);

    let mut child_a = PostfixExpr::new(child_a_nodes, child_a_consts, a.meta.clone());
    let mut child_b = PostfixExpr::new(child_b_nodes, child_b_consts, b.meta.clone());
    dynamic_expressions::compress_constants(&mut child_a);
    dynamic_expressions::compress_constants(&mut child_b);
    (child_a, child_b)
}
