use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_registry::OpRegistry;
use dynamic_expressions::{
    compile_plan, eval_plan_array_into, eval_plan_array_into_cached, EvalOptions, SubtreeCache,
};
use ndarray::Array2;

const D: usize = 2;
type T = f64;
type Ops = BuiltinOpsF64;

fn op_id(token: &str, arity: u8) -> u16 {
    <Ops as OpRegistry>::lookup_with_arity(token, arity)
        .unwrap_or_else(|_| panic!("missing op {token}/{arity}"))
        .op
        .id
}

fn build_expr() -> PostfixExpr<T, Ops, D> {
    let add = op_id("+", 2);
    let mul = op_id("*", 2);
    let nodes = vec![
        PNode::Var { feature: 0 },
        PNode::Var { feature: 1 },
        PNode::Op { arity: 2, op: add },
        PNode::Var { feature: 0 },
        PNode::Const { idx: 0 },
        PNode::Op { arity: 2, op: mul },
        PNode::Op { arity: 2, op: add },
    ];
    PostfixExpr::new(nodes, vec![1.0], Default::default())
}

fn make_x(n_rows: usize) -> Array2<T> {
    let n_features = 2usize;
    let mut data = vec![0.0; n_rows * n_features];
    for row in 0..n_rows {
        let base = row * n_features;
        data[base] = row as f64 * 0.25;
        data[base + 1] = 1.0 + row as f64 * 0.1;
    }
    Array2::from_shape_vec((n_features, n_rows), data).unwrap()
}

#[test]
fn subtree_cache_disabled_matches_uncached() {
    let expr = build_expr();
    let plan = compile_plan::<D>(&expr.nodes, 2, expr.consts.len());
    let x = make_x(4);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };

    let mut out_uncached = vec![0.0; x.ncols()];
    let mut out_cached = vec![0.0; x.ncols()];
    let mut scratch = Array2::<T>::zeros((0, 0));

    let ok_uncached = eval_plan_array_into(
        &mut out_uncached,
        &plan,
        &expr,
        x.view(),
        &mut scratch,
        &opts,
    );

    let mut cache = SubtreeCache::new(x.ncols(), 0);
    let ok_cached = eval_plan_array_into_cached(
        &mut out_cached,
        &plan,
        &expr,
        x.view(),
        &mut scratch,
        &opts,
        &mut cache,
        1,
    );

    assert_eq!(ok_uncached, ok_cached);
    assert_eq!(out_uncached, out_cached);
}

#[test]
fn subtree_cache_reuse_with_const_updates_matches_uncached() {
    let mut expr = build_expr();
    let plan = compile_plan::<D>(&expr.nodes, 2, expr.consts.len());
    let x = make_x(8);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };

    let mut scratch = Array2::<T>::zeros((0, 0));
    let mut cache = SubtreeCache::new(x.ncols(), 1 << 20);

    for tick in 0..5u64 {
        expr.consts[0] = (tick as f64 * 0.3).sin();

        let mut out_uncached = vec![0.0; x.ncols()];
        let mut out_cached = vec![0.0; x.ncols()];

        let ok_uncached = eval_plan_array_into(
            &mut out_uncached,
            &plan,
            &expr,
            x.view(),
            &mut scratch,
            &opts,
        );

        let ok_cached = eval_plan_array_into_cached(
            &mut out_cached,
            &plan,
            &expr,
            x.view(),
            &mut scratch,
            &opts,
            &mut cache,
            42,
        );

        assert_eq!(ok_uncached, ok_cached);
        assert_eq!(out_uncached, out_cached);
    }
}

#[test]
fn subtree_cache_handles_row_count_change() {
    let expr = build_expr();
    let plan = compile_plan::<D>(&expr.nodes, 2, expr.consts.len());
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };

    let mut cache = SubtreeCache::new(4, 1 << 20);
    let mut scratch = Array2::<T>::zeros((0, 0));

    let x1 = make_x(4);
    let mut out1_uncached = vec![0.0; x1.ncols()];
    let mut out1_cached = vec![0.0; x1.ncols()];
    let ok1_uncached = eval_plan_array_into(
        &mut out1_uncached,
        &plan,
        &expr,
        x1.view(),
        &mut scratch,
        &opts,
    );
    let ok1_cached = eval_plan_array_into_cached(
        &mut out1_cached,
        &plan,
        &expr,
        x1.view(),
        &mut scratch,
        &opts,
        &mut cache,
        7,
    );
    assert_eq!(ok1_uncached, ok1_cached);
    assert_eq!(out1_uncached, out1_cached);

    let x2 = make_x(3);
    let mut out2_uncached = vec![0.0; x2.ncols()];
    let mut out2_cached = vec![0.0; x2.ncols()];
    let ok2_uncached = eval_plan_array_into(
        &mut out2_uncached,
        &plan,
        &expr,
        x2.view(),
        &mut scratch,
        &opts,
    );
    let ok2_cached = eval_plan_array_into_cached(
        &mut out2_cached,
        &plan,
        &expr,
        x2.view(),
        &mut scratch,
        &opts,
        &mut cache,
        7,
    );
    assert_eq!(ok2_uncached, ok2_cached);
    assert_eq!(out2_uncached, out2_cached);
}
