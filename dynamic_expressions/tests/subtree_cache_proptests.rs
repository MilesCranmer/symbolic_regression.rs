use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_registry::OpRegistry;
use dynamic_expressions::{
    compile_plan, eval_plan_array_into, eval_plan_array_into_cached, EvalOptions, SubtreeCache,
};
use ndarray::Array2;
use proptest::prelude::*;

const N_FEATURES: usize = 4;
const N_CONSTS: usize = 3;
const N_ROWS: usize = 32;
const D: usize = 2;

type T = f64;
type Ops = BuiltinOpsF64;

#[derive(Clone, Debug)]
enum GenTree {
    Var(u16),
    Const(u16),
    Op {
        op: u16,
        children: Vec<GenTree>,
    },
}

impl GenTree {
    fn to_postfix(&self, out: &mut Vec<PNode>) {
        match self {
            GenTree::Var(feature) => out.push(PNode::Var { feature: *feature }),
            GenTree::Const(idx) => out.push(PNode::Const { idx: *idx }),
            GenTree::Op { op, children } => {
                for child in children {
                    child.to_postfix(out);
                }
                out.push(PNode::Op { arity: 2, op: *op });
            }
        }
    }
}

fn safe_binary_op_ids() -> Vec<u16> {
    let add = <Ops as OpRegistry>::lookup_with_arity("+", 2)
        .expect("missing add")
        .op
        .id;
    let sub = <Ops as OpRegistry>::lookup_with_arity("-", 2)
        .expect("missing sub")
        .op
        .id;
    let mul = <Ops as OpRegistry>::lookup_with_arity("*", 2)
        .expect("missing mul")
        .op
        .id;
    vec![add, sub, mul]
}

fn arb_tree_nodes() -> impl Strategy<Value = Vec<PNode>> {
    let leaf = prop_oneof![
        (0u16..(N_FEATURES as u16)).prop_map(GenTree::Var),
        (0u16..(N_CONSTS as u16)).prop_map(GenTree::Const),
    ];

    let ops = safe_binary_op_ids();
    leaf.prop_recursive(5, 64, 8, move |inner| {
        (
            prop::sample::select(ops.clone()),
            prop::collection::vec(inner, 2),
        )
            .prop_map(|(op, children)| GenTree::Op { op, children })
    })
    .prop_map(|tree| {
        let mut nodes = Vec::new();
        tree.to_postfix(&mut nodes);
        nodes
    })
}

fn arb_consts() -> impl Strategy<Value = Vec<T>> {
    prop::collection::vec(prop::num::f64::NORMAL, N_CONSTS)
        .prop_map(|vals| vals.into_iter().map(|v| v.tanh()).collect())
}

fn arb_x_data() -> impl Strategy<Value = Vec<T>> {
    prop::collection::vec(prop::num::f64::NORMAL, N_ROWS * N_FEATURES)
        .prop_map(|vals| vals.into_iter().map(|v| v.tanh()).collect())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        .. ProptestConfig::default()
    })]

    #[test]
    fn cached_eval_matches_uncached(
        nodes in arb_tree_nodes(),
        consts in arb_consts(),
        x_data in arb_x_data(),
        dataset_key in any::<u64>(),
    ) {
        let expr = PostfixExpr::<T, Ops, D>::new(nodes, consts, Metadata::default());
        let plan = compile_plan::<D>(&expr.nodes, N_FEATURES, expr.consts.len());
        let x = Array2::from_shape_vec((N_FEATURES, N_ROWS), x_data).expect("shape");

        let mut out_regular = vec![0.0; N_ROWS];
        let mut out_cached = vec![0.0; N_ROWS];
        let mut scratch = Array2::<T>::zeros((0, 0));
        let opts = EvalOptions {
            check_finite: true,
            early_exit: false,
        };

        let ok_regular = eval_plan_array_into(
            &mut out_regular,
            &plan,
            &expr,
            x.view(),
            &mut scratch,
            &opts,
        );

        let mut cache = SubtreeCache::new(N_ROWS, 1 << 20);
        let ok_cached = eval_plan_array_into_cached(
            &mut out_cached,
            &plan,
            &expr,
            x.view(),
            &mut scratch,
            &opts,
            &mut cache,
            dataset_key,
        );

        prop_assert_eq!(ok_regular, ok_cached);
        prop_assert_eq!(out_regular, out_cached);
    }
}
