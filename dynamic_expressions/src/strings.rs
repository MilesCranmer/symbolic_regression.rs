use core::fmt;

use crate::expression::PostfixExpr;
use crate::node::PNode;
use crate::traits::{OpId, OperatorSet};

#[derive(Clone, Debug, Default)]
pub struct StringTreeOptions<'a> {
    pub variable_names: Option<&'a [String]>,
    pub pretty: bool,
}

pub fn default_string_variable(feature: u16, names: Option<&[String]>) -> String {
    if let Some(names) = names {
        if let Some(name) = names.get(usize::from(feature)) {
            return name.clone();
        }
    }
    format!("x{}", u32::from(feature))
}

pub fn default_string_constant<T: fmt::Display>(v: &T) -> String {
    v.to_string()
}

fn strip_outer_parens(mut s: &str) -> &str {
    loop {
        let bytes = s.as_bytes();
        if bytes.len() < 2 || bytes[0] != b'(' || bytes[bytes.len() - 1] != b')' {
            return s;
        }

        let mut depth = 0i32;
        let mut encloses_all = false;
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        encloses_all = i == bytes.len() - 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if encloses_all {
            s = &s[1..s.len() - 1];
            continue;
        }

        return s;
    }
}

pub fn string_tree<T, Ops, const D: usize>(expr: &PostfixExpr<T, Ops, D>, opts: StringTreeOptions<'_>) -> String
where
    T: fmt::Display,
    Ops: OperatorSet,
{
    let names = opts.variable_names.or({
        if expr.meta.variable_names.is_empty() {
            None
        } else {
            Some(expr.meta.variable_names.as_slice())
        }
    });

    let out = crate::node_utils::tree_mapreduce(
        &expr.nodes,
        |n| match *n {
            PNode::Var { feature } => default_string_variable(feature, names),
            PNode::Const { idx } => default_string_constant(&expr.consts[usize::from(idx)]),
            _ => unreachable!("branch node in leaf mapper"),
        },
        |n| match *n {
            PNode::Op { arity, op } => OpId { arity, id: op },
            _ => unreachable!("leaf node in branch mapper"),
        },
        |op, children| {
            debug_assert_eq!(children.len(), op.arity as usize);
            let has_infix = Ops::infix(op).is_some();
            let op_display = if opts.pretty {
                Ops::display(op)
            } else {
                Ops::infix(op).unwrap_or(Ops::name(op))
            };
            let children_strings = children.iter().map(|s| strip_outer_parens(s)).collect::<Vec<_>>();
            if has_infix && op.arity > 1 {
                // Infix form, like {c1} {op} {c2} {op} {c3} ...
                let joined = children_strings.join(format!(" {op_display} ").as_str());
                format!("({joined})")
            } else {
                let joined = children_strings.join(", ");
                format!("{op_display}({joined})")
            }
        },
    );
    strip_outer_parens(&out).to_string()
}

pub fn print_tree<T, Ops, const D: usize>(expr: &PostfixExpr<T, Ops, D>)
where
    T: fmt::Display,
    Ops: OperatorSet,
{
    println!("{}", string_tree(expr, StringTreeOptions::default()));
}

impl<T, Ops, const D: usize> fmt::Display for PostfixExpr<T, Ops, D>
where
    T: fmt::Display,
    Ops: OperatorSet,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&string_tree(self, StringTreeOptions::default()))
    }
}
