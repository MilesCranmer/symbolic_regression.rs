use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use fastrand::Rng;
use ndarray::{Array1, Array2};
use num_traits::Zero;

use super::common::{D, T, TestOps};
use crate::Options;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::expression::{ExprExt, ExpressionSpec};
use crate::operator_library::OperatorLibrary;
use crate::pop_member::{Evaluator, MemberId, PopMember};
use crate::template::{TemplateExpression, TemplateSpec, TemplateStructure, string_template_pretty};

fn max_feature(nodes: &[PNode]) -> Option<u16> {
    nodes
        .iter()
        .filter_map(|n| match *n {
            PNode::Var { feature } => Some(feature),
            _ => None,
        })
        .max()
}

#[test]
fn template_fixed_inputs_allows_destructure_and_multiple_calls() {
    let n_rows = 32;
    let n_features = 3;

    // y = 2*(x0 + x1) + a0*x2, with a0 = 3
    let mut x = vec![0.0; n_features * n_rows];
    let mut y = vec![0.0; n_rows];
    for (r, yr) in y.iter_mut().enumerate() {
        let x0 = r as T;
        let x1 = (r as T) * 0.5;
        let x2 = 1.0 + (r as T) * 0.25;
        let o0 = r;
        let o1 = n_rows + r;
        let o2 = 2 * n_rows + r;
        x[o0] = x0;
        x[o1] = x1;
        x[o2] = x2;

        *yr = 2.0 * (x0 + x1) + 3.0 * x2;
    }

    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        use_baseline: false,
        ..Default::default()
    };

    let add = <TestOps as dynamic_expressions::HasOp<dynamic_expressions::operator_enum::builtin::Add>>::op_id();
    let f_tree = PostfixExpr::<T, TestOps, D>::new(
        vec![
            PNode::Var { feature: 0u16 },
            PNode::Var { feature: 1u16 },
            PNode::Op { arity: 2, op: add.id },
        ],
        vec![],
        Metadata::default(),
    );
    let g_tree = PostfixExpr::<T, TestOps, D>::new(vec![PNode::Var { feature: 0u16 }], vec![], Metadata::default());

    let structure = TemplateStructure::<T, TestOps, D>::new_fixed_inputs::<3, _>(
        vec![("f", 2), ("g", 1)],
        vec![("a", 1)],
        |ctx, [x0, x1, x2]| {
            let f1 = ctx.call("f", &[x0, x1]);
            let f2 = ctx.call("f", &[x0, x1]);
            let g = ctx.call("g", &[x2]);

            let a0 = ctx.param("a").unwrap()[0];
            let mut out = vec![T::zero(); ctx.n_rows()];
            for (dst, ((&v1, &v2), &gv)) in out.iter_mut().zip(f1.iter().zip(f2.iter()).zip(g.iter())) {
                *dst = (v1 + v2) + a0 * gv;
            }
            out
        },
    );

    let template = TemplateExpression::<T, TestOps, D> {
        structure: std::sync::Arc::new(structure),
        trees: vec![f_tree, g_tree],
        params: vec![vec![3.0]],
    };

    let mut member = PopMember::from_expr(MemberId(0), None, 0, template, dataset.n_features);
    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let tagged = TaggedDataset::new(&dataset, None);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    assert!(member.loss < 1e-12);
}

#[test]
fn template_constraints_reject_out_of_range_feature_for_subexpression() {
    // f has arity 1, but its tree references feature 1.
    let structure = TemplateStructure::<T, TestOps, D>::new(vec![("f", 1)], vec![], |_ctx, x| x[0].to_vec());

    let bad_tree = PostfixExpr::<T, TestOps, D>::new(vec![PNode::Var { feature: 1u16 }], vec![], Metadata::default());

    let expr = TemplateExpression::<T, TestOps, D> {
        structure: std::sync::Arc::new(structure),
        trees: vec![bad_tree],
        params: vec![],
    };

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        maxsize: 10,
        ..Default::default()
    };

    assert!(!expr.check_constraints(&options, options.maxsize));
}

#[test]
fn template_spec_random_expr_respects_per_tree_feature_limits() {
    let spec = TemplateSpec::<T, TestOps, D>::new_with_combine_fixed_inputs::<3, _>(
        vec![("f", 2), ("g", 1)],
        vec![],
        |_ctx, [x0, _x1, _x2]| x0.to_vec(),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        maxsize: 20,
        ..Default::default()
    };
    let operators = &options.operators;
    let mut rng = Rng::with_seed(0);

    let expr = spec.random_expr(&mut rng, operators, 3, 12, &options);
    assert_eq!(expr.trees.len(), 2);

    // f sees 2 features, g sees 1 feature.
    let mf_f = max_feature(&expr.trees[0].nodes);
    let mf_g = max_feature(&expr.trees[1].nodes);
    assert!(mf_f.map(|m| m < 2).unwrap_or(true));
    assert!(mf_g.map(|m| m < 1).unwrap_or(true));

    assert!(expr.check_constraints(&options, options.maxsize));
}

#[test]
fn template_parameter_is_optimizable_via_constant_optimizer() {
    let n_rows = 64;
    let n_features = 1;

    // y = a0 * x0, with a0 = 2.0
    let mut x = vec![0.0; n_features * n_rows];
    let mut y = vec![0.0; n_rows];
    for (r, yr) in y.iter_mut().enumerate() {
        let xr = (r as T) / (n_rows as T) - 0.5;
        x[r] = xr;
        *yr = 2.0 * xr;
    }
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        should_optimize_constants: true,
        optimizer_iterations: 60,
        optimizer_nrestarts: 0,
        use_baseline: false,
        ..Default::default()
    };

    let structure =
        TemplateStructure::<T, TestOps, D>::new_fixed_inputs::<1, _>(vec![("g", 1)], vec![("a", 1)], |ctx, [x0]| {
            let g = ctx.call("g", &[x0]);
            let a0 = ctx.param("a").unwrap()[0];
            let mut out = vec![T::zero(); ctx.n_rows()];
            for (dst, &gv) in out.iter_mut().zip(g.iter()) {
                *dst = a0 * gv;
            }
            out
        });

    let g_tree = PostfixExpr::<T, TestOps, D>::new(vec![PNode::Var { feature: 0u16 }], vec![], Metadata::default());

    let expr = TemplateExpression::<T, TestOps, D> {
        structure: std::sync::Arc::new(structure),
        trees: vec![g_tree],
        params: vec![vec![0.0]],
    };

    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let tagged = TaggedDataset::new(&dataset, baseline_loss);

    let mut member = PopMember::from_expr(MemberId(0), None, 0, expr, dataset.n_features);
    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let mut grad_ctx = dynamic_expressions::GradContext::<T, D>::new(dataset.n_rows);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    let loss_before = member.loss;

    let mut rng = Rng::with_seed(0);
    let mut next_birth = 100u64;
    let (improved, _evals) = optimize_constants(
        &mut rng,
        &mut member,
        OptimizeConstantsCtx {
            dataset: tagged,
            options: &options,
            evaluator: &mut evaluator,
            grad_ctx: &mut grad_ctx,
            next_birth: &mut next_birth,
        },
    );
    assert!(improved);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    let loss_after = member.loss;
    assert!(loss_after < loss_before);

    let a0 = member.expr.params[0][0];
    assert!((a0 - 2.0).abs() < 1e-2, "a0={a0}");
}

#[test]
fn template_pretty_string_includes_components_and_params() {
    let structure = TemplateStructure::<T, TestOps, D>::new(vec![("f", 1)], vec![("a", 2)], |_ctx, x| x[0].to_vec());

    let expr = TemplateExpression::<T, TestOps, D> {
        structure: std::sync::Arc::new(structure),
        trees: vec![PostfixExpr::<T, TestOps, D>::new(
            vec![PNode::Var { feature: 0u16 }],
            vec![],
            Metadata::default(),
        )],
        params: vec![vec![1.0, 2.0]],
    };

    let s = string_template_pretty(&expr);
    assert!(s.contains("TemplateExpression"));
    assert!(s.contains("f = "));
    assert!(s.contains("a = "));
}
