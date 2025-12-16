use crate::options::Options;
use dynamic_expressions::expr::PNode;
use dynamic_expressions::operators::scalar::OpId;
use num_traits::Float;

pub fn compute_complexity<T: Float, Ops, const D: usize>(
    nodes: &[PNode],
    options: &Options<T, D>,
) -> usize {
    if options.uses_default_complexity() {
        return nodes.len();
    }

    let mut st: Vec<i32> = Vec::with_capacity(nodes.len().min(256));

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
                st.push(c.max(0));
            }
            PNode::Const { .. } => st.push(options.complexity_of_constants.max(0)),
            PNode::Op { arity, op } => {
                let a = arity as usize;
                let mut sum: i32 = 0;
                for _ in 0..a {
                    sum = sum.saturating_add(st.pop().unwrap_or(0));
                }
                let oid = OpId { arity, id: op };
                let base = options
                    .operator_complexity_overrides
                    .get(&oid)
                    .copied()
                    .unwrap_or(1);
                st.push(base.max(0).saturating_add(sum));
            }
        }
    }

    if st.len() != 1 {
        return 0;
    }
    usize::try_from(st[0].max(0)).unwrap_or(usize::MAX)
}
