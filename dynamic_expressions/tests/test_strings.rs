mod common;

use common::expr_readme_like;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{HasOp, Metadata, PNode, PostfixExpr, custom_opset};

custom_opset! {
    struct Infix3Ops<f64> {
        2 {
            add {
                display: "+",
                infix: "+",
                commutative: true,
                associative: true,
                eval(args) { args[0] + args[1] },
                partial(_args, _idx) { 1.0 },
            }
        }
        3 {
            oplus {
                display: "⊕",
                infix: "⊕",
                eval(args) { args[0] + args[1] + args[2] },
                partial(_args, _idx) { 1.0 },
            }
        }
    }
}

fn var_infix3(feature: u16) -> PostfixExpr<f64, Infix3Ops, 3> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn add_infix3(x: PostfixExpr<f64, Infix3Ops, 3>, y: PostfixExpr<f64, Infix3Ops, 3>) -> PostfixExpr<f64, Infix3Ops, 3> {
    dynamic_expressions::expression_algebra::__apply_postfix::<f64, Infix3Ops, 3, 2>(
        <Infix3Ops as HasOp<Infix3OpsAdd>>::ID,
        [x, y],
    )
}

fn oplus3(
    x: PostfixExpr<f64, Infix3Ops, 3>,
    y: PostfixExpr<f64, Infix3Ops, 3>,
    z: PostfixExpr<f64, Infix3Ops, 3>,
) -> dynamic_expressions::PostfixExpr<f64, Infix3Ops, 3> {
    dynamic_expressions::expression_algebra::__apply_postfix::<f64, Infix3Ops, 3, 3>(
        <Infix3Ops as HasOp<Infix3OpsOplus>>::ID,
        [x, y, z],
    )
}

#[test]
fn string_tree_matches_expected() {
    let ex = expr_readme_like();
    let s = string_tree(&ex, StringTreeOptions::default());
    assert_eq!(s, "x0 * cos(x1 - 3.2)");
}

#[test]
fn string_tree_uses_variable_names() {
    let mut ex = expr_readme_like();
    ex.meta.variable_names = vec!["x".into(), "y".into()];
    let s = string_tree(&ex, StringTreeOptions::default());
    assert_eq!(s, "x * cos(y - 3.2)");
}

#[test]
fn infix_preserves_child_parens_for_precedence() {
    let mut a = common::var(0);
    a.meta.variable_names = vec!["a".into(), "b".into(), "c".into()];
    let b = common::var(1);
    let c = common::var(2);

    let expr = (a + b) * c;
    let s = string_tree(&expr, StringTreeOptions::default());
    assert_eq!(s, "(a + b) * c");
}

#[test]
fn ternary_infix_custom_op_preserves_child_grouping() {
    let mut a = var_infix3(0);
    a.meta.variable_names = vec!["a".into(), "b".into(), "c".into()];
    let b = var_infix3(1);
    let c = var_infix3(2);

    // (a + b) ⊕ (b + c) ⊕ a
    //
    // If the "infix join" path strips child outer parens, this can degrade to:
    // a + b ⊕ b + c ⊕ a  (ambiguous / wrong-looking).
    let expr = oplus3(add_infix3(a.clone(), b.clone()), add_infix3(b, c), a);

    let s = string_tree(&expr, StringTreeOptions::default());
    assert_eq!(s, "(a + b) ⊕ (b + c) ⊕ a");
}
