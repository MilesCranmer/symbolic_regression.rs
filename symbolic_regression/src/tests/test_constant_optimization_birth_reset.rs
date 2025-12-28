use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use fastrand::Rng;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::Options;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::operator_library::OperatorLibrary;
use crate::pop_member::{Evaluator, MemberId, PopMember, reset_pseudo_time_for_tests};

fn var(feature: u16) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(vec![PNode::Var { feature }], Vec::new(), Metadata::default())
}

#[test]
fn optimize_constants_resets_birth_on_improvement() {
    reset_pseudo_time_for_tests();
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
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        should_optimize_constants: true,
        deterministic: true,
        optimizer_iterations: 200,
        optimizer_nrestarts: 1,
        ..Default::default()
    };

    let zero = PostfixExpr::<T, TestOps, D>::zero();
    let expr = (zero.clone() * var(0)) + zero;
    let mut member = PopMember::from_expr(MemberId(0), None, expr, dataset.n_features, &options);
    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let mut grad_ctx = dynamic_expressions::GradContext::<T, D>::new(dataset.n_rows);
    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
    let _ = member.evaluate(&full_dataset, &options, &mut evaluator);

    let mut rng = Rng::with_seed(0);
    let birth_before = member.birth;
    let (improved, _) = optimize_constants::<T, TestOps, D>(
        &mut rng,
        &mut member,
        OptimizeConstantsCtx {
            dataset: full_dataset,
            options: &options,
            evaluator: &mut evaluator,
            grad_ctx: &mut grad_ctx,
        },
    );
    assert!(improved);
    assert_ne!(member.birth, birth_before);
}
