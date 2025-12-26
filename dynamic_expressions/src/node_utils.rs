use std::cell::RefCell;

use crate::node::PNode;

pub fn tree_mapreduce<R>(
    nodes: &[PNode],
    f_leaf: impl FnMut(&PNode) -> R,
    f_branch: impl FnMut(&PNode) -> R,
    op: impl FnMut(R, &[R]) -> R,
) -> R {
    tree_mapreduce_with_stack(nodes, f_leaf, f_branch, op, None)
}

pub fn tree_mapreduce_with_stack<R>(
    nodes: &[PNode],
    mut f_leaf: impl FnMut(&PNode) -> R,
    mut f_branch: impl FnMut(&PNode) -> R,
    mut op: impl FnMut(R, &[R]) -> R,
    reusable_stack: Option<&mut Vec<R>>,
) -> R {
    match reusable_stack {
        Some(stack) => {
            stack.clear();
            tree_mapreduce_impl(nodes, stack, &mut f_leaf, &mut f_branch, &mut op)
        }
        None => {
            let mut stack = Vec::with_capacity(nodes.len());
            tree_mapreduce_impl(nodes, &mut stack, &mut f_leaf, &mut f_branch, &mut op)
        }
    }
}

fn tree_mapreduce_impl<R>(
    nodes: &[PNode],
    stack: &mut Vec<R>,
    f_leaf: &mut impl FnMut(&PNode) -> R,
    f_branch: &mut impl FnMut(&PNode) -> R,
    op: &mut impl FnMut(R, &[R]) -> R,
) -> R {
    for n in nodes {
        match *n {
            PNode::Var { .. } | PNode::Const { .. } => stack.push(f_leaf(n)),
            PNode::Op { arity, .. } => {
                let a = arity as usize;
                let start = stack.len().checked_sub(a).expect("invalid postfix (stack underflow)");
                let parent = f_branch(n);
                let out = op(parent, &stack[start..]);
                stack.truncate(start);
                stack.push(out);
            }
        }
    }
    assert_eq!(stack.len(), 1, "invalid postfix (did not reduce to one root)");
    stack.pop().expect("non-empty stack")
}

thread_local! {
    static COUNT_DEPTH_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

pub fn count_depth(nodes: &[PNode]) -> usize {
    COUNT_DEPTH_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        tree_mapreduce_with_stack(
            nodes,
            |_| 1usize,
            |_| 0usize,
            |_, children| children.iter().copied().max().unwrap_or(0) + 1,
            Some(&mut stack),
        )
    })
}

pub fn count_nodes(nodes: &[PNode]) -> usize {
    nodes.len()
}

pub fn has_constants(nodes: &[PNode]) -> bool {
    nodes.iter().any(|n| matches!(n, PNode::Const { .. }))
}

pub fn count_constant_nodes(nodes: &[PNode]) -> usize {
    nodes.iter().filter(|n| matches!(n, PNode::Const { .. })).count()
}

pub fn has_operators(nodes: &[PNode]) -> bool {
    nodes.iter().any(|n| matches!(n, PNode::Op { .. }))
}

pub fn has_variables(nodes: &[PNode]) -> bool {
    nodes.iter().any(|n| matches!(n, PNode::Var { .. }))
}

pub fn count_variable_nodes(nodes: &[PNode]) -> usize {
    nodes.iter().filter(|n| matches!(n, PNode::Var { .. })).count()
}

pub fn count_operator_nodes(nodes: &[PNode]) -> usize {
    nodes.iter().filter(|n| matches!(n, PNode::Op { .. })).count()
}

pub fn max_arity(nodes: &[PNode]) -> u8 {
    nodes
        .iter()
        .filter_map(|n| match n {
            PNode::Op { arity, .. } => Some(*arity),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

pub fn is_leaf(nodes: &[PNode]) -> bool {
    matches!(nodes, [PNode::Var { .. }] | [PNode::Const { .. }])
}

pub fn is_valid_postfix(nodes: &[PNode]) -> bool {
    let mut stack: isize = 0;
    for n in nodes {
        match *n {
            PNode::Var { .. } | PNode::Const { .. } => stack += 1,
            PNode::Op { arity, .. } => {
                let a = arity as isize;
                if a <= 0 {
                    return false;
                }
                if stack < a {
                    return false;
                }
                stack = stack - a + 1;
            }
        }
    }
    stack == 1
}

pub fn subtree_sizes(nodes: &[PNode]) -> Vec<usize> {
    let mut sizes = vec![0usize; nodes.len()];
    let mut stack: Vec<usize> = Vec::with_capacity(nodes.len());

    for (i, n) in nodes.iter().enumerate() {
        match *n {
            PNode::Var { .. } | PNode::Const { .. } => {
                sizes[i] = 1;
                stack.push(1);
            }
            PNode::Op { arity, .. } => {
                let a = arity as usize;
                let mut sum = 1usize;
                for _ in 0..a {
                    sum += stack.pop().expect("invalid postfix (stack underflow)");
                }
                sizes[i] = sum;
                stack.push(sum);
            }
        }
    }

    assert_eq!(stack.len(), 1, "invalid postfix (did not reduce to one root)");
    sizes
}

pub fn subtree_range(subtree_sizes: &[usize], root_idx: usize) -> (usize, usize) {
    let sz = subtree_sizes[root_idx];
    (root_idx + 1 - sz, root_idx)
}
