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

enum OpStyle<'a> {
    Prefix(&'a str),
    Infix(&'a str),
    Call(&'a str),
}

fn combine(style: OpStyle<'_>, args: &[String]) -> String {
    match style {
        OpStyle::Prefix(tok) => {
            debug_assert_eq!(args.len(), 1);
            let a = strip_outer_parens(&args[0]);
            if a.contains(' ') {
                format!("{tok}({a})")
            } else {
                format!("{tok}{a}")
            }
        }
        OpStyle::Infix(tok) => {
            debug_assert_eq!(args.len(), 2);
            format!("({} {} {})", args[0], tok, args[1])
        }
        OpStyle::Call(opname) => {
            let mut out = String::new();
            out.push_str(opname);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(strip_outer_parens(a));
            }
            out.push(')');
            out
        }
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

    let mut stack: Vec<String> = Vec::with_capacity(expr.nodes.len());

    for n in &expr.nodes {
        match *n {
            PNode::Var { feature } => stack.push(default_string_variable(feature, names)),
            PNode::Const { idx } => {
                let idx = usize::from(idx);
                stack.push(default_string_constant(&expr.consts[idx]));
            }
            PNode::Op { arity, op } => {
                let a = arity as usize;
                let start = stack.len() - a;
                let op = OpId { arity, id: op };

                let infix = Ops::infix(op);
                let style = match (arity, infix) {
                    (1, Some(tok)) => {
                        let tok = if opts.pretty { Ops::display(op) } else { tok };
                        OpStyle::Prefix(tok)
                    }
                    (2, Some(tok)) => {
                        let tok = if opts.pretty { Ops::display(op) } else { tok };
                        OpStyle::Infix(tok)
                    }
                    _ => {
                        let name = if opts.pretty { Ops::display(op) } else { Ops::name(op) };
                        OpStyle::Call(name)
                    }
                };

                let out = combine(style, &stack[start..]);
                stack.truncate(start);
                stack.push(out);
            }
        }
    }

    assert_eq!(stack.len(), 1);
    strip_outer_parens(&stack[0]).to_string()
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
