use crate::options::Options;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::scalar::OpId;
use num_traits::Float;
use std::collections::HashMap;

pub(crate) fn compute_custom_complexity_checked<T: Float, const D: usize>(
    nodes: &[PNode],
    options: &Options<T, D>,
    op_arg_limits: Option<&HashMap<OpId, [Option<u16>; D]>>,
) -> Option<usize> {
    let mut st: Vec<usize> = Vec::with_capacity(nodes.len().min(256));

    for n in nodes {
        match *n {
            PNode::Var { feature } => {
                let idx = feature as usize;
                let c = options
                    .variable_complexities
                    .as_ref()
                    .and_then(|v| v.get(idx))
                    .copied()
                    .unwrap_or(options.complexity_of_variables);
                st.push(c as usize);
            }
            PNode::Const { .. } => st.push(options.complexity_of_constants as usize),
            PNode::Op { arity, op } => {
                let a = arity as usize;
                let mut child = [0usize; D];
                for j in (0..a).rev() {
                    child[j] = st.pop().unwrap_or(0);
                }

                let oid = OpId { arity, id: op };
                if let Some(limits) = op_arg_limits.and_then(|m| m.get(&oid)) {
                    for j in 0..a {
                        let Some(lim) = limits[j] else {
                            continue;
                        };
                        if child[j] > (lim as usize) {
                            return None;
                        }
                    }
                }

                let mut sum: usize = 0;
                for c in child {
                    sum = sum.saturating_add(c);
                }
                let base = options
                    .operator_complexity_overrides
                    .get(&oid)
                    .copied()
                    .unwrap_or(1);
                st.push((base as usize).saturating_add(sum));
            }
        }
    }

    if st.len() != 1 {
        return None;
    }
    Some(st[0])
}

pub fn compute_complexity<T: Float, const D: usize>(
    nodes: &[PNode],
    options: &Options<T, D>,
) -> usize {
    if options.uses_default_complexity() {
        return nodes.len();
    }

    compute_custom_complexity_checked::<T, D>(nodes, options, None).unwrap_or(0)
}
