use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::{HasOp, Operators};
use fastrand::Rng;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::TaggedDataset;
use crate::operator_library::OperatorLibrary;
use crate::options::MutationWeights;
use crate::pop_member::{Evaluator, PopMember};
use crate::{Options, mutate};

fn leaf_expr(feature: u16) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(vec![PNode::Var { feature }], Vec::new(), Metadata::default())
}

#[test]
fn randomize_mutation_can_succeed_below_size_3() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(),
        Array1::from_vec(vec![0.0]),
    );
    let mut options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        maxsize: 2,
        ..Default::default()
    };
    options.annealing = false;
    options.use_frequency = false;
    options.mutation_weights = MutationWeights {
        mutate_constant: 0.0,
        mutate_operator: 0.0,
        mutate_feature: 0.0,
        swap_operands: 0.0,
        rotate_tree: 0.0,
        add_node: 0.0,
        insert_node: 0.0,
        delete_node: 0.0,
        simplify: 0.0,
        randomize: 1.0,
        do_nothing: 0.0,
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
    let stats = RunningSearchStatistics::new(options.maxsize, 10_000);

    let mut parent = PopMember::from_expr_with_birth(0, leaf_expr(0), dataset.n_features);
    assert!(parent.evaluate(&full_dataset, &options, &mut evaluator));

    let mut rng = Rng::with_seed(0);
    let (child, ok, _) = mutate::next_generation::<T, TestOps, D>(
        &parent,
        mutate::NextGenerationCtx {
            rng: &mut rng,
            dataset: full_dataset,
            temperature: 0.0,
            curmaxsize: 2,
            stats: &stats,
            options: &options,
            evaluator: &mut evaluator,
            _ops: core::marker::PhantomData,
        },
    );

    assert!(ok, "randomize mutation should succeed with curmaxsize=2");
    assert!(
        (1..=2).contains(&child.expr.nodes.len()),
        "expected a tree size in 1..=2, got {}",
        child.expr.nodes.len()
    );
}

#[test]
fn rotate_tree_is_not_disabled_on_non_binary_trees() {
    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        ..Default::default()
    };
    let member = PopMember::from_expr(
        PostfixExpr::<T, TestOps, D>::new(
            vec![PNode::Var { feature: 0 }, PNode::Op { arity: 1, op: 0 }],
            Vec::new(),
            Metadata::default(),
        ),
        1,
        &options,
    );

    let mut weights = MutationWeights {
        rotate_tree: 1.0,
        swap_operands: 1.0,
        ..MutationWeights::default()
    };
    mutate::condition_mutation_weights(&mut weights, &member, &options, 10, 1);

    assert_eq!(weights.swap_operands, 0.0);
    assert_eq!(weights.rotate_tree, 1.0);
}

#[test]
fn default_mutation_weights_match_symbolicregressionjl() {
    let w = Options::<T, D>::default().mutation_weights;
    let eps = 1e-12;

    // Match SymbolicRegression.jl's `defaults` tuple in `src/Options.jl`.
    assert!((w.mutate_constant - 0.0346).abs() < eps);
    assert!((w.mutate_operator - 0.293).abs() < eps);
    assert!((w.mutate_feature - 0.1).abs() < eps);
    assert!((w.swap_operands - 0.198).abs() < eps);
    assert!((w.rotate_tree - 4.26).abs() < eps);
    assert!((w.add_node - 2.47).abs() < eps);
    assert!((w.insert_node - 0.0112).abs() < eps);
    assert!((w.delete_node - 0.870).abs() < eps);
    assert!((w.simplify - 0.00209).abs() < eps);
    assert!((w.randomize - 0.000502).abs() < eps);
    assert!((w.do_nothing - 0.273).abs() < eps);
    assert!((w.optimize - 0.0).abs() < eps);
    assert!((w.form_connection - 0.5).abs() < eps);
    assert!((w.break_connection - 0.1).abs() < eps);
}

fn contains_contiguous_slice<T: PartialEq>(haystack: &[T], needle: &[T]) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack.windows(needle.len()).any(|window| window == needle)
}

#[test]
fn add_node_includes_append_at_leaf_move() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap(),
        Array1::from_vec(vec![0.0]),
    );

    let add = <TestOps as HasOp<builtin::Add>>::op_id();
    let mut ops = Operators::<D>::new();
    ops.push(add);
    let mut options = Options::<T, D> {
        operators: ops,
        ..Default::default()
    };
    options.annealing = false;
    options.use_frequency = false;
    options.mutation_weights = MutationWeights {
        mutate_constant: 0.0,
        mutate_operator: 0.0,
        mutate_feature: 0.0,
        swap_operands: 0.0,
        rotate_tree: 0.0,
        add_node: 1.0,
        insert_node: 0.0,
        delete_node: 0.0,
        simplify: 0.0,
        randomize: 0.0,
        do_nothing: 0.0,
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };

    let expr = PostfixExpr::<T, TestOps, D>::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Var { feature: 1 },
            PNode::Op {
                arity: add.arity,
                op: add.id,
            },
        ],
        Vec::new(),
        Metadata::default(),
    );

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
    let stats = RunningSearchStatistics::new(options.maxsize, 10_000);

    let mut parent = PopMember::from_expr_with_birth(0, expr, dataset.n_features);
    assert!(parent.evaluate(&full_dataset, &options, &mut evaluator));

    let mut rng = Rng::with_seed(0);
    let parent_nodes = parent.expr.nodes.clone();
    let mut saw_prepend = false;
    let mut saw_append = false;

    for _ in 0..64 {
        let (child, ok, _) = mutate::next_generation::<T, TestOps, D>(
            &parent,
            mutate::NextGenerationCtx {
                rng: &mut rng,
                dataset: full_dataset,
                temperature: 0.0,
                curmaxsize: 100,
                stats: &stats,
                options: &options,
                evaluator: &mut evaluator,
                _ops: core::marker::PhantomData,
            },
        );
        assert!(ok, "add_node mutation should succeed");

        if contains_contiguous_slice(&child.expr.nodes, &parent_nodes) {
            saw_prepend = true;
        } else {
            saw_append = true;
        }

        if saw_prepend && saw_append {
            break;
        }
    }

    assert!(saw_prepend, "expected add_node to sometimes prepend at root");
    assert!(saw_append, "expected add_node to sometimes append at leaf");
}

#[test]
fn mutate_operator_can_be_a_noop_and_still_succeeds() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap(),
        Array1::from_vec(vec![0.0]),
    );

    let add = <TestOps as HasOp<builtin::Add>>::op_id();
    let mut ops = Operators::<D>::new();
    ops.push(add);

    let mut options = Options::<T, D> {
        operators: ops,
        ..Default::default()
    };
    options.annealing = false;
    options.use_frequency = false;
    options.mutation_weights = MutationWeights {
        mutate_constant: 0.0,
        mutate_operator: 1.0,
        mutate_feature: 0.0,
        swap_operands: 0.0,
        rotate_tree: 0.0,
        add_node: 0.0,
        insert_node: 0.0,
        delete_node: 0.0,
        simplify: 0.0,
        randomize: 0.0,
        do_nothing: 0.0,
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };

    let expr = PostfixExpr::<T, TestOps, D>::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Var { feature: 1 },
            PNode::Op {
                arity: add.arity,
                op: add.id,
            },
        ],
        Vec::new(),
        Metadata::default(),
    );

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
    let stats = RunningSearchStatistics::new(options.maxsize, 10_000);

    let mut parent = PopMember::from_expr_with_birth(0, expr, dataset.n_features);
    assert!(parent.evaluate(&full_dataset, &options, &mut evaluator));

    let mut rng = Rng::with_seed(0);
    let (child, ok, _) = mutate::next_generation::<T, TestOps, D>(
        &parent,
        mutate::NextGenerationCtx {
            rng: &mut rng,
            dataset: full_dataset,
            temperature: 0.0,
            curmaxsize: 100,
            stats: &stats,
            options: &options,
            evaluator: &mut evaluator,
            _ops: core::marker::PhantomData,
        },
    );

    assert!(ok, "mutate_operator should succeed even when it is a no-op");
    assert_eq!(
        child.expr.nodes, parent.expr.nodes,
        "with only one operator available, mutate_operator should be a no-op"
    );
}
