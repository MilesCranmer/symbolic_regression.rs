use super::common::{TestOps, D, T};
use crate::constant_optimization::optimize_constants;
use crate::member::{Evaluator, MemberId, PopMember};
use crate::operator_library::OperatorLibrary;
use crate::Options;
use dynamic_expressions::expr::{Metadata, PNode, PostfixExpr};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn var(feature: u16) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(
        vec![PNode::Var { feature }],
        Vec::new(),
        Metadata::default(),
    )
}

fn constant(value: T) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(
        vec![PNode::Const { idx: 0 }],
        vec![value],
        Metadata::default(),
    )
}

#[test]
fn optimize_constants_resets_birth_on_improvement() {
    let n_rows = 128;
    let n_features = 1;
    let mut x = vec![0.0; n_rows];
    let mut y = vec![0.0; n_rows];
    for i in 0..n_rows {
        let xi = (i as T) / (n_rows as T);
        x[i] = xi;
        y[i] = 2.0 * xi + 3.0;
    }
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((n_rows, n_features), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        should_optimize_constants: true,
        optimizer_iterations: 200,
        optimizer_nrestarts: 1,
        ..Default::default()
    };

    let expr = (constant(0.0) * var(0)) + constant(0.0);
    let mut member = PopMember::from_expr(MemberId(0), None, 0, expr, dataset.n_features);
    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let mut grad_ctx = dynamic_expressions::GradContext::<T, D>::new(dataset.n_rows);
    let _ = member.evaluate(&dataset, &options, &mut evaluator);

    let mut rng = StdRng::seed_from_u64(0);
    let mut next_birth = 1000u64;
    let birth_before = member.birth;
    let (improved, _) = optimize_constants::<T, TestOps, D, _>(
        &mut rng,
        &mut member,
        &dataset,
        &options,
        &mut evaluator,
        &mut grad_ctx,
        &mut next_birth,
    );
    assert!(improved);
    assert_ne!(member.birth, birth_before);
}
