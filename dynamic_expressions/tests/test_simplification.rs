use approx::assert_relative_eq;
use dynamic_expressions::math::cos;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::{
    combine_operators_in_place, compress_constants, simplify_tree_in_place, EvalOptions, PNode,
    PostfixExpr,
};

type Ops = BuiltinOpsF64;
const D: usize = 3;

fn var(feature: u16) -> PostfixExpr<f64, Ops, D> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Default::default())
}

fn cst(v: f64) -> PostfixExpr<f64, Ops, D> {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![v], Default::default())
}

fn opts() -> EvalOptions {
    EvalOptions {
        check_finite: true,
        early_exit: true,
    }
}

#[test]
fn simplify_tree_folds_unary_constant() {
    let mut expr: PostfixExpr<f64, Ops, D> = cos(cst(0.0));
    let changed = simplify_tree_in_place::<f64, Ops, D>(&mut expr, &opts());
    assert!(changed);
    assert_eq!(expr.nodes.len(), 1);
    let PNode::Const { idx } = expr.nodes[0] else {
        panic!("expected constant leaf after folding");
    };
    assert_relative_eq!(
        expr.consts[idx as usize],
        1.0,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
}

#[test]
fn simplify_tree_does_not_fold_nan() {
    let mut expr: PostfixExpr<f64, Ops, D> = cos(cst(f64::NAN));
    let changed = simplify_tree_in_place::<f64, Ops, D>(&mut expr, &opts());
    assert!(!changed);
    assert!(expr.nodes.len() > 1);
}

#[test]
fn combine_operators_commutative_add_combines_constants() {
    let mut expr: PostfixExpr<f64, Ops, D> = cst(0.5) + (cst(0.2) + var(0));
    let changed = combine_operators_in_place::<f64, Ops, D>(&mut expr);
    assert!(changed);
    let _ = compress_constants(&mut expr);

    assert_eq!(expr.consts.len(), 1);
    assert_relative_eq!(expr.consts[0], 0.7, epsilon = 1e-12, max_relative = 1e-12);
    assert_eq!(expr.nodes.len(), 3);
    assert!(matches!(
        expr.nodes[0],
        PNode::Var { .. } | PNode::Const { .. }
    ));
    assert!(matches!(
        expr.nodes[1],
        PNode::Var { .. } | PNode::Const { .. }
    ));
    assert!(matches!(expr.nodes[2], PNode::Op { arity: 2, .. }));
}

#[test]
fn combine_operators_sub_pattern_1() {
    // (cA - (cB - x)) => (x - (cB - cA))
    let mut expr: PostfixExpr<f64, Ops, D> = cst(0.5) - (cst(0.2) - var(0));
    let _ = combine_operators_in_place::<f64, Ops, D>(&mut expr);
    let _ = compress_constants(&mut expr);

    assert_eq!(expr.nodes.len(), 3);
    assert!(matches!(expr.nodes[0], PNode::Var { .. }));
    assert!(matches!(expr.nodes[1], PNode::Const { .. }));
    assert!(matches!(expr.nodes[2], PNode::Op { arity: 2, .. }));
    assert_relative_eq!(expr.consts[0], -0.3, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn combine_operators_sub_pattern_2() {
    // (cA - (x - cB)) => ((cA + cB) - x)
    let mut expr: PostfixExpr<f64, Ops, D> = cst(0.5) - (var(0) - cst(0.2));
    let _ = combine_operators_in_place::<f64, Ops, D>(&mut expr);
    let _ = compress_constants(&mut expr);

    assert_eq!(expr.nodes.len(), 3);
    assert!(matches!(expr.nodes[0], PNode::Const { .. }));
    assert!(matches!(expr.nodes[1], PNode::Var { .. }));
    assert!(matches!(expr.nodes[2], PNode::Op { arity: 2, .. }));
    assert_relative_eq!(expr.consts[0], 0.7, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn combine_operators_sub_pattern_3() {
    // ((cA - x) - cB) => ((cA - cB) - x)
    let mut expr: PostfixExpr<f64, Ops, D> = (cst(0.5) - var(0)) - cst(0.2);
    let _ = combine_operators_in_place::<f64, Ops, D>(&mut expr);
    let _ = compress_constants(&mut expr);

    assert_eq!(expr.nodes.len(), 3);
    assert!(matches!(expr.nodes[0], PNode::Const { .. }));
    assert!(matches!(expr.nodes[1], PNode::Var { .. }));
    assert!(matches!(expr.nodes[2], PNode::Op { arity: 2, .. }));
    assert_relative_eq!(expr.consts[0], 0.3, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn combine_operators_sub_pattern_4() {
    // ((x - cA) - cB) => (x - (cA + cB))
    let mut expr: PostfixExpr<f64, Ops, D> = (var(0) - cst(0.2)) - cst(0.6);
    let _ = combine_operators_in_place::<f64, Ops, D>(&mut expr);
    let _ = compress_constants(&mut expr);

    assert_eq!(expr.nodes.len(), 3);
    assert!(matches!(expr.nodes[0], PNode::Var { .. }));
    assert!(matches!(expr.nodes[1], PNode::Const { .. }));
    assert!(matches!(expr.nodes[2], PNode::Op { arity: 2, .. }));
    assert_relative_eq!(expr.consts[0], 0.8, epsilon = 1e-12, max_relative = 1e-12);
}

#[test]
fn combine_operators_nested_add_inside_cos_matches_julia_like() {
    // cos((0.1 + 0.2) + 0.2) + 2.0  => combine constants to 0.4 and keep 0.1
    let mut expr: PostfixExpr<f64, Ops, D> = cos((cst(0.1) + cst(0.2)) + cst(0.2)) + cst(2.0);
    let _ = combine_operators_in_place::<f64, Ops, D>(&mut expr);
    let _ = compress_constants(&mut expr);

    assert!(expr.consts.iter().copied().all(f64::is_finite));
    let mut cs = expr.consts.clone();
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_relative_eq!(cs[0], 0.1, epsilon = 1e-12, max_relative = 1e-12);
    assert_relative_eq!(cs[1], 0.4, epsilon = 1e-12, max_relative = 1e-12);
    assert_relative_eq!(cs[2], 2.0, epsilon = 1e-12, max_relative = 1e-12);
}
