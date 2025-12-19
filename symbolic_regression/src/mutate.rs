use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_constraints;
use crate::complexity::compute_complexity;
use crate::dataset::TaggedDataset;
use crate::loss_functions::loss_to_cost;
use crate::operators::Operators;
use crate::options::{MutationWeights, Options};
use crate::pop_member::{Evaluator, MemberId, PopMember};
pub use dynamic_expressions::compress_constants;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils::{subtree_range, subtree_sizes};
use dynamic_expressions::operator_enum::scalar::ScalarOpSet;
use dynamic_expressions::operator_registry::OpRegistry;
use num_traits::Float;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::Rng;
use rand_distr::StandardNormal;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MutationChoice {
    MutateConstant,
    MutateOperator,
    MutateFeature,
    SwapOperands,
    RotateTree,
    AddNode,
    InsertNode,
    DeleteNode,
    Simplify,
    Randomize,
    DoNothing,
    Optimize,
}

pub struct NextGenerationCtx<'a, T: Float, Ops, const D: usize, R: Rng> {
    pub rng: &'a mut R,
    pub dataset: TaggedDataset<'a, T>,
    pub temperature: f64,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub next_id: &'a mut u64,
    pub next_birth: &'a mut u64,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub struct CrossoverCtx<'a, T: Float, Ops, const D: usize, R: Rng> {
    pub rng: &'a mut R,
    pub dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub next_id: &'a mut u64,
    pub next_birth: &'a mut u64,
    pub _ops: core::marker::PhantomData<Ops>,
}

fn count_constants(nodes: &[PNode]) -> usize {
    nodes
        .iter()
        .filter(|n| matches!(n, PNode::Const { .. }))
        .count()
}

fn has_binary_op(nodes: &[PNode]) -> bool {
    nodes
        .iter()
        .any(|n| matches!(n, PNode::Op { arity: 2, .. }))
}

pub fn condition_mutation_weights<T: Float, Ops, const D: usize>(
    weights: &mut MutationWeights,
    member: &PopMember<T, Ops, D>,
    options: &Options<T, D>,
    curmaxsize: usize,
    nfeatures: usize,
) {
    let tree_is_leaf = member
        .expr
        .nodes
        .iter()
        .all(|n| matches!(n, PNode::Var { .. } | PNode::Const { .. }));
    if tree_is_leaf {
        weights.mutate_operator = 0.0;
        weights.swap_operands = 0.0;
        weights.delete_node = 0.0;
        weights.simplify = 0.0;
        if member.expr.consts.is_empty() {
            weights.optimize = 0.0;
            weights.mutate_constant = 0.0;
        } else {
            weights.mutate_feature = 0.0;
        }
        return;
    }

    if !has_binary_op(&member.expr.nodes) {
        weights.swap_operands = 0.0;
    }

    let nconst = count_constants(&member.expr.nodes);
    weights.mutate_constant *= (nconst.min(8) as f64) / 8.0;

    if nfeatures <= 1 {
        weights.mutate_feature = 0.0;
    }

    let complexity = member.complexity;
    if complexity >= curmaxsize {
        weights.add_node = 0.0;
        weights.insert_node = 0.0;
    }

    if !options.should_simplify {
        weights.simplify = 0.0;
    }

    if !options.should_optimize_constants
        || options.optimizer_probability == 0.0
        || member.expr.consts.is_empty()
    {
        weights.optimize = 0.0;
    }
}

pub fn sample_mutation<R: Rng>(rng: &mut R, weights: &MutationWeights) -> MutationChoice {
    let choices = [
        (MutationChoice::MutateConstant, weights.mutate_constant),
        (MutationChoice::MutateOperator, weights.mutate_operator),
        (MutationChoice::MutateFeature, weights.mutate_feature),
        (MutationChoice::SwapOperands, weights.swap_operands),
        (MutationChoice::RotateTree, weights.rotate_tree),
        (MutationChoice::AddNode, weights.add_node),
        (MutationChoice::InsertNode, weights.insert_node),
        (MutationChoice::DeleteNode, weights.delete_node),
        (MutationChoice::Simplify, weights.simplify),
        (MutationChoice::Randomize, weights.randomize),
        (MutationChoice::DoNothing, weights.do_nothing),
        (MutationChoice::Optimize, weights.optimize),
    ];
    let w: Vec<f64> = choices.iter().map(|(_, v)| *v).collect();
    let dist = WeightedIndex::new(w).expect("at least one mutation weight must be > 0");
    choices[dist.sample(rng)].0
}

fn random_leaf<T: Float, R: Rng>(rng: &mut R, n_features: usize, consts: &mut Vec<T>) -> PNode {
    if rng.random::<bool>() {
        let val_f64: f64 = StandardNormal.sample(rng);
        let val = T::from(val_f64).unwrap();
        let idx: u16 = consts
            .len()
            .try_into()
            .unwrap_or_else(|_| panic!("too many constants to index in u16"));
        consts.push(val);
        PNode::Const { idx }
    } else {
        let f: u16 = rng
            .random_range(0..n_features)
            .try_into()
            .unwrap_or_else(|_| panic!("too many features to index in u16"));
        PNode::Var { feature: f }
    }
}

pub fn random_expr<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    operators: &Operators<D>,
    n_features: usize,
    target_size: usize,
) -> PostfixExpr<T, Ops, D> {
    assert!(target_size >= 1);
    let mut nodes: Vec<PNode> = Vec::with_capacity(target_size);
    let mut consts: Vec<T> = Vec::new();
    nodes.push(random_leaf(rng, n_features, &mut consts));

    while nodes.len() < target_size
        && operators.total_ops_up_to(D.min(target_size - nodes.len())) > 0
    {
        let rem = target_size - nodes.len();
        let max_arity = rem.min(D);
        let arity = operators.sample_arity(rng, max_arity);
        let op_id = operators.sample_op(rng, arity).op.id;

        let leaf_positions: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
            .collect();
        let leaf_idx = leaf_positions[rng.random_range(0..leaf_positions.len())];

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

pub(crate) fn mutate_constant_in_place<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
    temperature: f64,
    options: &Options<T, D>,
) -> bool {
    let idxs = const_node_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let node_i = idxs[rng.random_range(0..idxs.len())];
    let PNode::Const { idx } = expr.nodes[node_i] else {
        return false;
    };
    let ci = usize::from(idx);

    // Follows SymbolicRegression.jl's `mutate_factor`.
    let pf = options.perturbation_factor * temperature.max(0.0);
    let max_change = pf + 1.1;
    let exponent: f64 = rng.random::<f64>();
    let mut mul = max_change.powf(exponent);
    let make_const_bigger: bool = rng.random();
    mul = if make_const_bigger { mul } else { 1.0 / mul };
    if rng.random::<f64>() > options.probability_negate_constant {
        mul = -mul;
    }
    expr.consts[ci] = expr.consts[ci] * T::from(mul).unwrap();
    true
}

fn mutate_operator_in_place<T, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
) -> bool {
    let idxs = op_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let i = idxs[rng.random_range(0..idxs.len())];
    let PNode::Op { arity, .. } = expr.nodes[i] else {
        return false;
    };
    let a = arity as usize;
    if operators.nops(a) == 0 {
        return false;
    }

    // Match SymbolicRegression.jl: sample uniformly among all operators of the same arity,
    // including the current one (i.e., this mutation can be a no-op).
    let new_op = operators.sample_op(rng, a).op.id;
    expr.nodes[i] = PNode::Op { arity, op: new_op };
    true
}

fn mutate_feature_in_place<T, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
    n_features: usize,
) -> bool {
    if n_features <= 1 {
        return false;
    }
    let idxs = var_node_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let node_i = idxs[rng.random_range(0..idxs.len())];
    let PNode::Var { feature } = expr.nodes[node_i] else {
        return false;
    };
    let old = usize::from(feature);
    if old >= n_features {
        return false;
    }

    for _ in 0..8 {
        let new_feature = rng.random_range(0..n_features);
        if new_feature != old {
            let new_u16: u16 = new_feature
                .try_into()
                .unwrap_or_else(|_| panic!("too many features to index in u16"));
            expr.nodes[node_i] = PNode::Var { feature: new_u16 };
            return true;
        }
    }
    false
}

fn swap_operands_in_place<T, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
) -> bool {
    let idxs: Vec<usize> = expr
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| matches!(n, PNode::Op { arity: 2, .. }).then_some(i))
        .collect();
    if idxs.is_empty() {
        return false;
    }
    let sizes = subtree_sizes(&expr.nodes);
    let root_idx = idxs[rng.random_range(0..idxs.len())];
    let PNode::Op { op, .. } = expr.nodes[root_idx] else {
        return false;
    };
    let (sub_start, sub_end) = subtree_range(&sizes, root_idx);
    let child = child_ranges(&sizes, root_idx, 2);
    let mut new_sub: Vec<PNode> = Vec::with_capacity(sub_end + 1 - sub_start);
    new_sub.extend_from_slice(&expr.nodes[child[1].0..=child[1].1]);
    new_sub.extend_from_slice(&expr.nodes[child[0].0..=child[0].1]);
    new_sub.push(PNode::Op { arity: 2, op });
    expr.nodes.splice(sub_start..=sub_end, new_sub);
    true
}

pub(crate) fn rotate_tree_in_place<T, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
) -> bool {
    // Match SymbolicRegression.jl's `randomly_rotate_tree!`:
    // pick a random rotation root where some child is an operator, then
    // rotate along a random internal edge (root -> pivot) using a random grandchild.
    let sizes = subtree_sizes(&expr.nodes);
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
        if children
            .iter()
            .any(|c| matches!(expr.nodes[c.1], PNode::Op { .. }))
        {
            valid_roots.push(i);
        }
    }
    if valid_roots.is_empty() {
        return false;
    }

    let root_idx = valid_roots[rng.random_range(0..valid_roots.len())];
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

    let pivot_pos = pivot_positions[rng.random_range(0..pivot_positions.len())];
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

    let grandchild_pos = rng.random_range(0..pivot_arity);
    let grandchild = pivot_children[grandchild_pos];

    let (sub_start, sub_end) = subtree_range(&sizes, root_idx);

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

fn insert_random_op_in_place<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
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
    let root_idx = rng.random_range(0..expr.nodes.len());
    let sizes = subtree_sizes(&expr.nodes);
    let (start, end) = subtree_range(&sizes, root_idx);
    let old_sub: Vec<PNode> = expr.nodes[start..=end].to_vec();

    let arity = operators.sample_arity(rng, D);
    let op_id = operators.sample_op(rng, arity).op.id;
    let carry_pos = rng.random_range(0..arity);

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
    compress_constants(expr);
    true
}

pub(crate) fn prepend_random_op_in_place<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
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
    let op_id = operators.sample_op(rng, arity).op.id;
    let carry_pos = rng.random_range(0..arity);

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
    compress_constants(expr);
    true
}

pub(crate) fn append_random_op_in_place<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
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
    let leaf_idx = leaf_positions[rng.random_range(0..leaf_positions.len())];

    let arity = operators.sample_arity(rng, D);
    let op_id = operators.sample_op(rng, arity).op.id;

    let mut replace_with: Vec<PNode> = Vec::with_capacity(arity + 1);
    for _ in 0..arity {
        replace_with.push(random_leaf(rng, n_features, &mut expr.consts));
    }
    replace_with.push(PNode::Op {
        arity: arity as u8,
        op: op_id,
    });

    expr.nodes.splice(leaf_idx..=leaf_idx, replace_with);
    compress_constants(expr);
    true
}

fn add_node_in_place<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
    operators: &Operators<D>,
    n_features: usize,
) -> bool {
    if rng.random::<bool>() {
        append_random_op_in_place(rng, expr, operators, n_features)
    } else {
        prepend_random_op_in_place(rng, expr, operators, n_features)
    }
}

fn delete_random_op_in_place<T: Clone, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    expr: &mut PostfixExpr<T, Ops, D>,
) -> bool {
    let idxs = op_indices(&expr.nodes);
    if idxs.is_empty() {
        return false;
    }
    let root_idx = idxs[rng.random_range(0..idxs.len())];
    let PNode::Op { arity, .. } = expr.nodes[root_idx] else {
        return false;
    };
    let a = arity as usize;
    if a == 0 {
        return false;
    }
    let sizes = subtree_sizes(&expr.nodes);
    let (sub_start, sub_end) = subtree_range(&sizes, root_idx);
    if sub_start == sub_end {
        return false;
    }
    let child = child_ranges(&sizes, root_idx, a);
    let keep = child[rng.random_range(0..a)];
    let kept_nodes: Vec<PNode> = expr.nodes[keep.0..=keep.1].to_vec();
    expr.nodes.splice(sub_start..=sub_end, kept_nodes);
    compress_constants(expr);
    true
}

fn crossover_trees<T: Clone, Ops, const D: usize, R: Rng>(
    rng: &mut R,
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

    let a_sizes = subtree_sizes(&a.nodes);
    let b_sizes = subtree_sizes(&b.nodes);
    let a_root = rng.random_range(0..a.nodes.len());
    let b_root = rng.random_range(0..b.nodes.len());
    let (a_start, a_end) = subtree_range(&a_sizes, a_root);
    let (b_start, b_end) = subtree_range(&b_sizes, b_root);

    let a_sub = &a.nodes[a_start..=a_end];
    let b_sub = &b.nodes[b_start..=b_end];

    let mut child_a_nodes: Vec<PNode> =
        Vec::with_capacity(a.nodes.len() - a_sub.len() + b_sub.len());
    child_a_nodes.extend_from_slice(&a.nodes[..a_start]);
    let mut child_a_consts = a.consts.clone();
    let b_sub_remap = remap_subtree_consts(b_sub, &b.consts, &mut child_a_consts);
    child_a_nodes.extend_from_slice(&b_sub_remap);
    child_a_nodes.extend_from_slice(&a.nodes[a_end + 1..]);

    let mut child_b_nodes: Vec<PNode> =
        Vec::with_capacity(b.nodes.len() - b_sub.len() + a_sub.len());
    child_b_nodes.extend_from_slice(&b.nodes[..b_start]);
    let mut child_b_consts = b.consts.clone();
    let a_sub_remap = remap_subtree_consts(a_sub, &a.consts, &mut child_b_consts);
    child_b_nodes.extend_from_slice(&a_sub_remap);
    child_b_nodes.extend_from_slice(&b.nodes[b_end + 1..]);

    let mut child_a = PostfixExpr::new(child_a_nodes, child_a_consts, a.meta.clone());
    let mut child_b = PostfixExpr::new(child_b_nodes, child_b_consts, b.meta.clone());
    compress_constants(&mut child_a);
    compress_constants(&mut child_b);
    (child_a, child_b)
}

pub fn next_generation<T: Float, Ops, const D: usize, R: Rng>(
    member: &PopMember<T, Ops, D>,
    ctx: NextGenerationCtx<'_, T, Ops, D, R>,
) -> (PopMember<T, Ops, D>, bool, f64)
where
    Ops: ScalarOpSet<T> + OpRegistry,
{
    let NextGenerationCtx {
        rng,
        dataset,
        temperature,
        curmaxsize,
        stats,
        options,
        evaluator,
        next_id,
        next_birth,
        ..
    } = ctx;

    let before_cost = member.cost.to_f64().unwrap_or(f64::INFINITY);
    let _before_loss = member.loss.to_f64().unwrap_or(f64::INFINITY);
    let n_features = dataset.n_features;

    let mut weights = options.mutation_weights.clone();
    condition_mutation_weights(&mut weights, member, options, curmaxsize, n_features);
    let choice = sample_mutation(rng, &weights);

    let max_attempts = 10;
    let mut successful = false;
    let mut return_immediately = false;
    let mut tree = member.expr.clone();

    for _ in 0..max_attempts {
        tree = member.expr.clone();
        let mutated = match choice {
            MutationChoice::MutateConstant => {
                mutate_constant_in_place(rng, &mut tree, temperature, options)
            }
            MutationChoice::MutateOperator => {
                mutate_operator_in_place(rng, &mut tree, &options.operators)
            }
            MutationChoice::MutateFeature => mutate_feature_in_place(rng, &mut tree, n_features),
            MutationChoice::SwapOperands => swap_operands_in_place(rng, &mut tree),
            MutationChoice::RotateTree => rotate_tree_in_place(rng, &mut tree),
            MutationChoice::AddNode => {
                add_node_in_place(rng, &mut tree, &options.operators, n_features)
            }
            MutationChoice::InsertNode => {
                insert_random_op_in_place(rng, &mut tree, &options.operators, n_features)
            }
            MutationChoice::DeleteNode => delete_random_op_in_place(rng, &mut tree),
            MutationChoice::Simplify => {
                let _ = dynamic_expressions::simplify_in_place(&mut tree, &evaluator.eval_opts);
                return_immediately = true;
                true
            }
            MutationChoice::Randomize => {
                // Match SymbolicRegression.jl: sample a *uniform* random size in 1:curmaxsize.
                let max_size = curmaxsize.max(1).min(options.maxsize.max(1));
                let target_size = rng.random_range(1..=max_size);
                tree = random_expr(rng, &options.operators, n_features, target_size);
                true
            }
            MutationChoice::DoNothing => true,
            MutationChoice::Optimize => true, // handled outside in tuning pass; as mutation it is a no-op here
        };
        if !mutated {
            continue;
        }
        compress_constants(&mut tree);
        if check_constraints(&tree, options, curmaxsize) {
            successful = true;
            break;
        }
    }

    let id = MemberId(*next_id);
    *next_id += 1;
    let birth = *next_birth;
    *next_birth += 1;

    if !successful {
        let mut baby =
            PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        baby.complexity = member.complexity;
        baby.loss = member.loss;
        baby.cost = member.cost;
        return (baby, false, 0.0);
    }

    if return_immediately {
        let mut baby = PopMember::from_expr(id, Some(member.id), birth, tree, n_features);
        baby.rebuild_plan(n_features);
        baby.loss = member.loss;
        baby.complexity = compute_complexity(&baby.expr.nodes, options);
        baby.cost = loss_to_cost(
            baby.loss,
            baby.complexity,
            options.parsimony,
            options.use_baseline,
            dataset.baseline_loss,
        );
        return (baby, true, 0.0);
    }

    let mut baby = PopMember::from_expr(id, Some(member.id), birth, tree, n_features);
    let ok = baby.evaluate(&dataset, options, evaluator);
    let after_cost = baby.cost.to_f64().unwrap_or(f64::INFINITY);
    let after_loss = baby.loss.to_f64().unwrap_or(f64::INFINITY);
    let _ = after_loss;
    if !ok || !after_cost.is_finite() {
        let mut reject =
            PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, 0.0);
    }

    let mut prob = 1.0f64;
    if options.annealing {
        let delta = after_cost - before_cost;
        prob *= (-delta / (temperature * options.alpha)).exp();
    }
    if options.use_frequency {
        let old_size = member.complexity;
        let new_size = baby.complexity;
        let old_f = if old_size > 0 && old_size <= options.maxsize {
            stats.freq(old_size)
        } else {
            1e-6
        };
        let new_f = if new_size > 0 && new_size <= options.maxsize {
            stats.freq(new_size)
        } else {
            1e-6
        };
        prob *= old_f / new_f;
    }

    if prob < rng.random::<f64>() {
        let mut reject =
            PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, 0.0);
    }

    (baby, true, 1.0)
}

pub fn crossover_generation<T: Float, Ops, const D: usize, R: Rng>(
    member1: &PopMember<T, Ops, D>,
    member2: &PopMember<T, Ops, D>,
    ctx: CrossoverCtx<'_, T, Ops, D, R>,
) -> (PopMember<T, Ops, D>, PopMember<T, Ops, D>, bool, f64)
where
    Ops: ScalarOpSet<T>,
{
    let CrossoverCtx {
        rng,
        dataset,
        curmaxsize,
        options,
        evaluator,
        next_id,
        next_birth,
        ..
    } = ctx;

    let max_tries = 10;
    let mut tries = 0;
    loop {
        let (c1_expr, c2_expr) = crossover_trees(rng, &member1.expr, &member2.expr);
        tries += 1;
        if check_constraints(&c1_expr, options, curmaxsize)
            && check_constraints(&c2_expr, options, curmaxsize)
        {
            let id1 = MemberId(*next_id);
            *next_id += 1;
            let b1 = *next_birth;
            *next_birth += 1;
            let id2 = MemberId(*next_id);
            *next_id += 1;
            let b2 = *next_birth;
            *next_birth += 1;

            let mut baby1 =
                PopMember::from_expr(id1, Some(member1.id), b1, c1_expr, dataset.n_features);
            let mut baby2 =
                PopMember::from_expr(id2, Some(member2.id), b2, c2_expr, dataset.n_features);
            let _ = baby1.evaluate(&dataset, options, evaluator);
            let _ = baby2.evaluate(&dataset, options, evaluator);
            return (baby1, baby2, true, 2.0);
        }
        if tries >= max_tries {
            let baby1 = member1.clone();
            let baby2 = member2.clone();
            return (baby1, baby2, false, 0.0);
        }
    }
}
