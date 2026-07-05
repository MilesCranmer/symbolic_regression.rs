use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::dataset::TaggedDataset;
use crate::pop_member::{Evaluator, PopMember};
use crate::{Dataset, Options};

#[test]
fn delayed_expression_loss_ignores_unavailable_warmup_rows() {
    let x = Array2::from_shape_vec((1, 5), vec![10.0 as T, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let y = Array1::from_vec(vec![999.0 as T, 999.0, 10.0, 20.0, 30.0]);
    let dataset = Dataset::with_weights_and_names(x, y, None, vec!["x".into()]);
    let tagged = TaggedDataset::new(&dataset, None);
    let expr = PostfixExpr::<T, TestOps, D>::new(
        vec![PNode::Delay { feature: 0, offset: 2 }],
        Vec::new(),
        Metadata {
            variable_names: vec!["x".into()],
        },
    );
    let options = Options::<T, D> {
        max_delay: 2,
        progress: false,
        ..Default::default()
    };
    let mut member = PopMember::from_expr(expr, dataset.n_features, &options);
    let mut evaluator = Evaluator::new(dataset.n_rows);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    assert_eq!(member.loss, 0.0);
}

#[test]
fn delayed_expression_loss_respects_sequence_boundaries() {
    let x = Array2::from_shape_vec((1, 6), vec![10.0 as T, 20.0, 30.0, 100.0, 200.0, 300.0]).unwrap();
    let y = Array1::from_vec(vec![999.0 as T, 10.0, 20.0, 999.0, 100.0, 200.0]);
    let dataset = Dataset::with_weights_names_and_sequence_ids(x, y, None, vec!["x".into()], vec![0, 0, 0, 1, 1, 1]);
    let tagged = TaggedDataset::new(&dataset, None);
    let expr = PostfixExpr::<T, TestOps, D>::new(
        vec![PNode::Delay { feature: 0, offset: 1 }],
        Vec::new(),
        Metadata {
            variable_names: vec!["x".into()],
        },
    );
    let options = Options::<T, D> {
        max_delay: 1,
        progress: false,
        ..Default::default()
    };
    let mut member = PopMember::from_expr(expr, dataset.n_features, &options);
    let mut evaluator = Evaluator::new(dataset.n_rows);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    assert_eq!(member.loss, 0.0);
}

#[test]
#[should_panic(expected = "sequence_ids must be nondecreasing")]
fn sequence_ids_must_be_contiguous() {
    let x = Array2::from_shape_vec((1, 3), vec![1.0 as T, 2.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![0.0 as T, 0.0, 0.0]);
    let _ = Dataset::with_weights_names_and_sequence_ids(x, y, None, vec!["x".into()], vec![0, 1, 0]);
}
