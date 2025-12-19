use crate::check_constraints::check_constraints;
use crate::check_constraints::{NestedConstraints, OpConstraints};
use crate::dataset::Dataset;
use crate::{compute_complexity, Options};
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin::{Add, Cos, Sin};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_enum::scalar::{HasOp, OpId};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;

#[test]
fn batch_resample_copies_rows_and_weights() {
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let y = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0]);
    let w = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let full_dataset =
        Dataset::with_weights_and_names(x, y, Some(w), vec!["x0".into(), "x1".into()]);

    let mut batch = Dataset::make_batch_buffer(&full_dataset, 3);
    let mut rng = StdRng::seed_from_u64(123);
    batch.resample_from(&full_dataset, &mut rng);

    assert_eq!(batch.n_rows, 3);
    assert_eq!(batch.n_features, 2);
    assert!(batch.weights.is_some());

    for i in 0..batch.n_rows {
        let row = batch.x.row(i).to_owned();
        let yi = batch.y[i];
        let wi = batch.weights.as_ref().unwrap()[i];

        let mut found = false;
        for j in 0..full_dataset.n_rows {
            if row == full_dataset.x.row(j).to_owned()
                && yi == full_dataset.y[j]
                && wi == full_dataset.weights.as_ref().unwrap()[j]
            {
                found = true;
                break;
            }
        }
        assert!(found, "batch row {i} was not copied from any full row");
    }
}

#[test]
fn batch_size_can_exceed_full_rows_with_replacement() {
    let n_rows = 10usize;
    let n_features = 2usize;

    let mut x = Vec::with_capacity(n_rows * n_features);
    let mut y = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        x.push(i as f64);
        x.push((i as f64) + 1000.0);
        y.push((i as f64) + 10.0);
    }
    let x = Array2::from_shape_vec((n_rows, n_features), x).unwrap();
    let y = Array1::from_vec(y);
    let full_dataset = Dataset::with_weights_and_names(x, y, None, vec!["x0".into(), "x1".into()]);

    let mut batch = Dataset::make_batch_buffer(&full_dataset, 50);
    assert_eq!(batch.n_rows, 50);

    let mut rng = StdRng::seed_from_u64(0);
    batch.resample_from(&full_dataset, &mut rng);

    let mut uniq: HashSet<(u64, u64, u64)> = HashSet::new();
    for i in 0..batch.n_rows {
        let a = batch.x[(i, 0)].to_bits();
        let b = batch.x[(i, 1)].to_bits();
        let c = batch.y[i].to_bits();
        uniq.insert((a, b, c));
    }
    assert!(uniq.len() < batch.n_rows);
}

#[test]
fn constraints_op_arg_and_nested_constraints_work() {
    type Ops = BuiltinOpsF64;
    const D: usize = 3;

    let add = OpId {
        arity: 2,
        id: <Ops as HasOp<Add, 2>>::ID,
    };
    let cos = OpId {
        arity: 1,
        id: <Ops as HasOp<Cos, 1>>::ID,
    };

    // expr = (x0 + x1) + x2
    let expr_add = PostfixExpr::<f64, Ops, D>::new(
        vec![
            PNode::Var { feature: 0u16 },
            PNode::Var { feature: 1u16 },
            PNode::Op {
                arity: 2,
                op: add.id,
            },
            PNode::Var { feature: 2u16 },
            PNode::Op {
                arity: 2,
                op: add.id,
            },
        ],
        vec![],
        Default::default(),
    );

    let mut options: Options<f64, D> = Options {
        complexity_of_variables: 1,
        complexity_of_constants: 1,
        ..Default::default()
    };
    options.operator_complexity_overrides.insert(add, 0);

    // Require arg0 of Add to have complexity <= 1 (so (x0+x1) fails).
    let mut op_constraints = OpConstraints::<D>::default();
    op_constraints.set_op_arg_constraint(add, 0, 1);
    options.op_constraints = op_constraints;

    assert!(!check_constraints(&expr_add, &options, 10));

    // expr = cos(cos(x0)), nested cos inside cos.
    let expr_cos = PostfixExpr::<f64, Ops, D>::new(
        vec![
            PNode::Var { feature: 0u16 },
            PNode::Op {
                arity: 1,
                op: cos.id,
            },
            PNode::Op {
                arity: 1,
                op: cos.id,
            },
        ],
        vec![],
        Default::default(),
    );

    let mut nested = NestedConstraints::default();
    nested.add_nested_constraint(cos, cos, 0);
    options.nested_constraints = nested;
    options.op_constraints = Default::default();

    assert!(!check_constraints(&expr_cos, &options, 10));
}

#[test]
fn compute_complexity_respects_custom_weights_and_rounding() {
    type Ops = BuiltinOpsF64;
    const D: usize = 3;

    let add = OpId {
        arity: 2,
        id: <Ops as HasOp<Add, 2>>::ID,
    };
    let sin = OpId {
        arity: 1,
        id: <Ops as HasOp<Sin, 1>>::ID,
    };

    // expr = sin(x0) + c0
    let expr = PostfixExpr::<f64, Ops, D>::new(
        vec![
            PNode::Var { feature: 0u16 },
            PNode::Op {
                arity: 1,
                op: sin.id,
            },
            PNode::Const { idx: 0 },
            PNode::Op {
                arity: 2,
                op: add.id,
            },
        ],
        vec![1.23],
        Default::default(),
    );

    let mut options: Options<f64, D> = Options {
        complexity_of_variables: 2,
        complexity_of_constants: 1,
        ..Default::default()
    };
    options.operator_complexity_overrides.insert(add, 1);
    options.operator_complexity_overrides.insert(sin, 3);

    // sin(x0): 3 + 2 = 5
    // add: 1 + 5 + 1 = 7
    assert_eq!(compute_complexity(&expr.nodes, &options), 7);
}
