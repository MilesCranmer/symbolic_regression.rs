use super::common::{TestOps, D, T};
use crate::mutate::compress_constants;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;

#[test]
fn compress_constants_remaps_and_shrinks_pool() {
    let mut expr = PostfixExpr::<T, TestOps, D>::new(
        vec![PNode::Const { idx: 1 }],
        vec![1.0, 2.0, 3.0],
        Metadata::default(),
    );
    compress_constants(&mut expr);
    assert_eq!(expr.consts, vec![2.0]);
    assert_eq!(expr.nodes, vec![PNode::Const { idx: 0 }]);
}
