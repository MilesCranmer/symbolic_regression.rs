use crate::complexity;
use crate::options::Options;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils::{count_depth, subtree_sizes};
use dynamic_expressions::operator_enum::scalar::OpId;
use num_traits::Float;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct OpConstraints<const D: usize> {
    /// Per-operator, per-argument complexity limits.
    /// `None` means no constraint for that argument.
    pub limits: HashMap<OpId, [Option<u16>; D]>,
}

impl<const D: usize> Default for OpConstraints<D> {
    fn default() -> Self {
        Self {
            limits: HashMap::new(),
        }
    }
}

impl<const D: usize> OpConstraints<D> {
    pub fn set_op_arg_constraint(&mut self, op: OpId, arg_idx: usize, max_complexity: u16) {
        assert!(arg_idx < D);
        let entry = self.limits.entry(op).or_insert([None; D]);
        entry[arg_idx] = Some(max_complexity);
    }
}

#[derive(Clone, Debug, Default)]
pub struct NestedConstraints {
    /// Root operator -> list of (nested operator, max nestedness).
    pub limits: HashMap<OpId, Vec<(OpId, u8)>>,
}

impl NestedConstraints {
    pub fn add_nested_constraint(&mut self, root: OpId, nested: OpId, max_nestedness: u8) {
        self.limits
            .entry(root)
            .or_default()
            .push((nested, max_nestedness));
    }
}

pub fn check_constraints<T: Float, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    options: &Options<T, D>,
    curmaxsize: usize,
) -> bool {
    if count_depth(&expr.nodes) > options.maxdepth {
        return false;
    }
    if options.uses_default_complexity() {
        if expr.nodes.len() > curmaxsize {
            return false;
        }
        if !check_default_op_arg_constraints::<T, D>(&expr.nodes, options) {
            return false;
        }
    } else {
        let Some(total) = complexity::compute_custom_complexity_checked(
            &expr.nodes,
            options,
            Some(&options.op_constraints.limits),
        ) else {
            return false;
        };
        if total > curmaxsize {
            return false;
        }
    }

    check_nested_constraints::<D>(&expr.nodes, &options.nested_constraints)
}

fn check_default_op_arg_constraints<T: Float, const D: usize>(
    nodes: &[PNode],
    options: &Options<T, D>,
) -> bool {
    if options.op_constraints.limits.is_empty() {
        return true;
    }
    let sizes = subtree_sizes(nodes);
    for (i, n) in nodes.iter().enumerate() {
        let PNode::Op { arity, op } = *n else {
            continue;
        };
        let oid = OpId { arity, id: op };
        let Some(lims) = options.op_constraints.limits.get(&oid) else {
            continue;
        };
        let a = arity as usize;
        let ranges = child_ranges(&sizes, i, a);
        for j in 0..a {
            let Some(lim) = lims[j] else {
                continue;
            };
            {
                let (_start, end) = ranges[j];
                let child_sz = sizes[end];
                if child_sz > (lim as usize) {
                    return false;
                }
            }
        }
    }
    true
}

fn check_nested_constraints<const D: usize>(nodes: &[PNode], nested: &NestedConstraints) -> bool {
    if nested.limits.is_empty() {
        return true;
    }

    fn nestedness_vec(nodes: &[PNode], target: OpId) -> Option<Vec<u16>> {
        let mut st: Vec<u16> = Vec::with_capacity(nodes.len().min(256));
        let mut out: Vec<u16> = Vec::with_capacity(nodes.len());
        for n in nodes {
            match *n {
                PNode::Var { .. } | PNode::Const { .. } => {
                    st.push(0);
                    out.push(0);
                }
                PNode::Op { arity, op } => {
                    let a = arity as usize;
                    if st.len() < a {
                        return None;
                    }
                    let mut m = 0u16;
                    for _ in 0..a {
                        m = m.max(st.pop().expect("checked"));
                    }
                    let self_is = (arity == target.arity && op == target.id) as u16;
                    let v = m.saturating_add(self_is);
                    st.push(v);
                    out.push(v);
                }
            }
        }
        if st.len() != 1 {
            return None;
        }
        Some(out)
    }

    let mut nested_cache: HashMap<OpId, Vec<u16>> = HashMap::new();
    for rules in nested.limits.values() {
        for (nested_op, _max) in rules {
            if !nested_cache.contains_key(nested_op) {
                let Some(v) = nestedness_vec(nodes, *nested_op) else {
                    return false;
                };
                nested_cache.insert(*nested_op, v);
            }
        }
    }

    for (i, n) in nodes.iter().enumerate() {
        let PNode::Op { arity, op } = *n else {
            continue;
        };
        let root = OpId { arity, id: op };
        let Some(rules) = nested.limits.get(&root) else {
            continue;
        };
        for (nested_op, max_n) in rules {
            let mut v = nested_cache
                .get(nested_op)
                .and_then(|vv| vv.get(i).copied())
                .unwrap_or(0);
            if root == *nested_op && v > 0 {
                v -= 1;
            }
            if v > (*max_n as u16) {
                return false;
            }
        }
    }

    true
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
