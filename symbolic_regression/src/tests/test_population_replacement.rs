use super::common::{TestOps, D, T};
use crate::member::{Evaluator, MemberId, PopMember};
use crate::operator_library::OperatorLibrary;
use crate::population::Population;
use crate::Options;
use dynamic_expressions::expr::{Metadata, PNode, PostfixExpr};
use ndarray::{Array1, Array2};

fn leaf_expr(feature: u16) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(
        vec![PNode::Var { feature }],
        Vec::new(),
        Metadata::default(),
    )
}

#[test]
fn population_replaces_by_oldest_birth() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(),
        Array1::from_vec(vec![0.0]),
    );
    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        ..Default::default()
    };
    let mut evaluator = Evaluator::<T, D>::new(1);

    let mut m1 = PopMember::from_expr(MemberId(1), None, 10, leaf_expr(0), 1);
    let mut m2 = PopMember::from_expr(MemberId(2), None, 20, leaf_expr(0), 1);
    let mut m3 = PopMember::from_expr(MemberId(3), None, 30, leaf_expr(0), 1);
    let _ = m1.evaluate(&dataset, &options, &mut evaluator);
    let _ = m2.evaluate(&dataset, &options, &mut evaluator);
    let _ = m3.evaluate(&dataset, &options, &mut evaluator);

    let mut pop = Population::new(vec![m1, m2, m3]);
    assert_eq!(pop.oldest_index(), 0);

    pop.members[2].birth = 5;
    assert_eq!(pop.oldest_index(), 2);

    let child = PopMember::from_expr(MemberId(99), None, 100, leaf_expr(0), 1);
    pop.replace_oldest(child);
    assert_eq!(pop.members[2].id, MemberId(99));
}
