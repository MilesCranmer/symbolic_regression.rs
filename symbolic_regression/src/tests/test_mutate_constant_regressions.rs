use crate::mutate::mutate_constant_in_place;
use crate::Options;
use dynamic_expressions::expression::Metadata;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn constant_mutation_is_bounded_with_floor_at_zero_temperature() {
    let mut expr = PostfixExpr::<f64, (), 2>::new(
        vec![PNode::Const { idx: 0 }],
        vec![1.0],
        Metadata::default(),
    );
    let options: Options<f64, 2> = Options {
        probability_negate_constant: 1.0, // `rand() > 1.0` never, so no sign flip.
        perturbation_factor: 123.0,       // irrelevant at temperature=0.
        ..Default::default()
    };

    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..256 {
        let before = expr.consts[0];
        assert!(mutate_constant_in_place(&mut rng, &mut expr, 0.0, &options));
        let after = expr.consts[0];
        let ratio = after / before;
        assert!(
            ((1.0 / 1.1) - 1e-12..=1.1 + 1e-12).contains(&ratio),
            "ratio={ratio} out of bounds at temperature=0"
        );
    }
}

#[test]
fn constant_mutation_uses_inverted_sign_flip_probability() {
    let mut expr = PostfixExpr::<f64, (), 2>::new(
        vec![PNode::Const { idx: 0 }],
        vec![1.0],
        Metadata::default(),
    );
    let options: Options<f64, 2> = Options {
        probability_negate_constant: 0.0, // `rand() > 0.0` should negate almost always.
        ..Default::default()
    };

    let mut rng = StdRng::seed_from_u64(1);
    let mut saw_negative = false;
    for _ in 0..32 {
        assert!(mutate_constant_in_place(&mut rng, &mut expr, 0.0, &options));
        if expr.consts[0].is_sign_negative() {
            saw_negative = true;
            break;
        }
    }
    assert!(saw_negative, "expected at least one sign flip with p=0.0");
}
