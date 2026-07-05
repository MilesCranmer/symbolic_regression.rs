use core::fmt;

use num_traits::ToPrimitive;

use crate::expression::PostfixExpr;
use crate::node::PNode;
use crate::traits::{OpId, OperatorSet};

#[derive(Clone, Debug, Default)]
pub struct StringTreeOptions<'a> {
    pub variable_names: Option<&'a [String]>,
    pub pretty: bool,
    /// Number of significant digits to use when formatting numeric constants. `None` reverts to
    /// each value's `Display` impl (full precision). `Some(n)` formats via a `%.<n>g`-style
    /// emulation that matches SymbolicRegression.jl's `print_precision` behavior.
    pub print_precision: Option<usize>,
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

/// Format a float with `sig_digits` significant digits, mimicking C `%g` / Julia `%.<n>g`:
/// - returns `"0"` for zero,
/// - uses scientific notation when the decimal exponent is `< -4` or `>= sig_digits`,
/// - otherwise uses fixed notation,
/// - strips trailing zeros (and a trailing decimal point) from the mantissa.
pub fn format_g(v: f64, sig_digits: usize) -> String {
    let sig_digits = sig_digits.max(1);
    if !v.is_finite() {
        return v.to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let p = (sig_digits as i32) - 1;

    if exp < -4 || exp >= sig_digits as i32 {
        let raw = format!("{:.*e}", p as usize, v);
        strip_zeros_scientific(&raw)
    } else {
        let decimals = (p - exp).max(0) as usize;
        let raw = format!("{:.*}", decimals, v);
        strip_zeros_fixed(&raw)
    }
}

fn strip_zeros_fixed(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn strip_zeros_scientific(s: &str) -> String {
    let Some(e_pos) = s.find('e') else {
        return s.to_string();
    };
    let (mantissa, exp_part) = s.split_at(e_pos);
    let mantissa = if mantissa.contains('.') {
        mantissa.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        mantissa.to_string()
    };
    // Rust's `{:e}` formatter emits `e5`, `e-5`, etc. Reformat to match C / Julia `%g`:
    // explicit sign and minimum two-digit exponent (e.g. `e+05`, `e-08`).
    let exp_str = &exp_part[1..];
    let (sign, mag) = if let Some(rest) = exp_str.strip_prefix('-') {
        ("-", rest)
    } else if let Some(rest) = exp_str.strip_prefix('+') {
        ("+", rest)
    } else {
        ("+", exp_str)
    };
    let padded = if mag.len() < 2 {
        format!("0{mag}")
    } else {
        mag.to_string()
    };
    format!("{mantissa}e{sign}{padded}")
}

fn format_constant<T: fmt::Display + ToPrimitive>(v: &T, precision: Option<usize>) -> String {
    match precision {
        Some(n) => match v.to_f64() {
            Some(f) => format_g(f, n),
            None => default_string_constant(v),
        },
        None => default_string_constant(v),
    }
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
    T: fmt::Display + ToPrimitive,
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
            PNode::Delay { feature, offset } => {
                format!("delay({}, {})", default_string_variable(feature, names), offset)
            }
            PNode::Const { idx } => format_constant(&expr.consts[usize::from(idx)], opts.print_precision),
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
            let args = children.iter().map(|s| s.as_str());
            if has_infix && op.arity > 1 {
                // Infix form, like {c1} {op} {c2} {op} {c3} ...
                let joined = args.collect::<Vec<_>>().join(format!(" {op_display} ").as_str());
                format!("({joined})")
            } else {
                let joined = args.map(strip_outer_parens).collect::<Vec<_>>().join(", ");
                format!("{op_display}({joined})")
            }
        },
    );
    strip_outer_parens(&out).to_string()
}

pub fn print_tree<T, Ops, const D: usize>(expr: &PostfixExpr<T, Ops, D>)
where
    T: fmt::Display + ToPrimitive,
    Ops: OperatorSet,
{
    println!("{}", string_tree(expr, StringTreeOptions::default()));
}

impl<T, Ops, const D: usize> fmt::Display for PostfixExpr<T, Ops, D>
where
    T: fmt::Display + ToPrimitive,
    Ops: OperatorSet,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&string_tree(self, StringTreeOptions::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::format_g;

    // Reference values pinned against Julia's `@sprintf("%.5g", x)` semantics.
    #[test]
    fn format_g_matches_julia_percent_g() {
        // Zero.
        assert_eq!(format_g(0.0, 5), "0");
        assert_eq!(format_g(-0.0, 5), "0");

        // Fixed-point range with trailing-zero stripping.
        assert_eq!(format_g(1.0, 5), "1");
        assert_eq!(format_g(2.0, 5), "2");
        assert_eq!(format_g(-2.0, 5), "-2");
        assert_eq!(format_g(1.5, 5), "1.5");
        assert_eq!(format_g(1.999999998, 5), "2");
        assert_eq!(format_g(1.7994438074463217, 5), "1.7994");
        assert_eq!(format_g(0.5, 5), "0.5");
        assert_eq!(format_g(-0.5001000001, 5), "-0.5001");

        // Scientific range (exp < -4 or exp >= precision). Exponent format mirrors C / Julia
        // `%g`: explicit sign, minimum two-digit magnitude.
        assert_eq!(format_g(1.350382597828686e-8, 5), "1.3504e-08");
        assert_eq!(format_g(1.234e10, 5), "1.234e+10");
        assert_eq!(format_g(1.6941562429091146, 5), "1.6942");

        // Different precisions.
        assert_eq!(format_g(1.23456789, 3), "1.23");
        assert_eq!(format_g(1.23456789, 7), "1.234568");
        assert_eq!(format_g(0.0001234, 5), "0.0001234");
        assert_eq!(format_g(0.00001234, 5), "1.234e-05");

        // Non-finite passthrough.
        assert_eq!(format_g(f64::NAN, 5), f64::NAN.to_string());
        assert_eq!(format_g(f64::INFINITY, 5), f64::INFINITY.to_string());
        assert_eq!(format_g(f64::NEG_INFINITY, 5), f64::NEG_INFINITY.to_string());
    }
}
