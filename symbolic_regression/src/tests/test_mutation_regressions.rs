use super::common::{TestOps, D, T};
use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::TaggedDataset;
use crate::mutate::{condition_mutation_weights, next_generation, NextGenerationCtx};
use crate::operator_library::OperatorLibrary;
use crate::operators::{OpSpec, Operators};
use crate::options::MutationWeights;
use crate::pop_member::{Evaluator, MemberId, PopMember};
use crate::Options;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin::Add;
use dynamic_expressions::operator_enum::scalar::{HasOp, OpId};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn leaf_expr(feature: u16) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(
        vec![PNode::Var { feature }],
        Vec::new(),
        Metadata::default(),
    )
}

#[test]
fn randomize_mutation_can_succeed_below_size_3() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap(),
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
    let full_dataset = TaggedDataset::new(&dataset, options.loss.as_ref(), options.use_baseline);
    let stats = RunningSearchStatistics::new(options.maxsize, 10_000);

    let mut parent = PopMember::from_expr(MemberId(0), None, 0, leaf_expr(0), dataset.n_features);
    assert!(parent.evaluate(&full_dataset, &options, &mut evaluator));

    let mut rng = StdRng::seed_from_u64(0);
    let mut next_id = 1u64;
    let mut next_birth = 0u64;

    let (child, ok, _) = next_generation::<T, TestOps, D, _>(
        &parent,
        NextGenerationCtx {
            rng: &mut rng,
            dataset: full_dataset,
            temperature: 0.0,
            curmaxsize: 2,
            stats: &stats,
            options: &options,
            evaluator: &mut evaluator,
            next_id: &mut next_id,
            next_birth: &mut next_birth,
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
        MemberId(0),
        None,
        0,
        PostfixExpr::<T, TestOps, D>::new(
            vec![PNode::Var { feature: 0 }, PNode::Op { arity: 1, op: 0 }],
            Vec::new(),
            Metadata::default(),
        ),
        1,
    );

    let mut weights = MutationWeights {
        rotate_tree: 1.0,
        swap_operands: 1.0,
        ..MutationWeights::default()
    };
    condition_mutation_weights(&mut weights, &member, &options, 10, 1);

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
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[test]
fn add_node_includes_append_at_leaf_move() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap(),
        Array1::from_vec(vec![0.0]),
    );

    let mut ops = Operators::<D>::new();
    ops.push(
        2,
        OpSpec {
            op: OpId {
                arity: 2,
                id: <TestOps as HasOp<Add, 2>>::ID,
            },
            commutative: true,
            associative: true,
            complexity: 1,
        },
    );
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
                arity: 2,
                op: <TestOps as HasOp<Add, 2>>::ID,
            },
        ],
        Vec::new(),
        Metadata::default(),
    );

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let full_dataset = TaggedDataset::new(&dataset, options.loss.as_ref(), options.use_baseline);
    let stats = RunningSearchStatistics::new(options.maxsize, 10_000);

    let mut parent = PopMember::from_expr(MemberId(0), None, 0, expr, dataset.n_features);
    assert!(parent.evaluate(&full_dataset, &options, &mut evaluator));

    let mut rng = StdRng::seed_from_u64(0);
    let mut next_id = 1u64;
    let mut next_birth = 0u64;

    let parent_nodes = parent.expr.nodes.clone();
    let mut saw_prepend = false;
    let mut saw_append = false;

    for _ in 0..64 {
        let (child, ok, _) = next_generation::<T, TestOps, D, _>(
            &parent,
            NextGenerationCtx {
                rng: &mut rng,
                dataset: full_dataset,
                temperature: 0.0,
                curmaxsize: 100,
                stats: &stats,
                options: &options,
                evaluator: &mut evaluator,
                next_id: &mut next_id,
                next_birth: &mut next_birth,
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

    assert!(
        saw_prepend,
        "expected add_node to sometimes prepend at root"
    );
    assert!(saw_append, "expected add_node to sometimes append at leaf");
}

#[test]
fn mutate_operator_can_be_a_noop_and_still_succeeds() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap(),
        Array1::from_vec(vec![0.0]),
    );

    let mut ops = Operators::<D>::new();
    ops.push(
        2,
        OpSpec {
            op: OpId {
                arity: 2,
                id: <TestOps as HasOp<Add, 2>>::ID,
            },
            commutative: true,
            associative: true,
            complexity: 1,
        },
    );

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
                arity: 2,
                op: <TestOps as HasOp<Add, 2>>::ID,
            },
        ],
        Vec::new(),
        Metadata::default(),
    );

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let full_dataset = TaggedDataset::new(&dataset, options.loss.as_ref(), options.use_baseline);
    let stats = RunningSearchStatistics::new(options.maxsize, 10_000);

    let mut parent = PopMember::from_expr(MemberId(0), None, 0, expr, dataset.n_features);
    assert!(parent.evaluate(&full_dataset, &options, &mut evaluator));

    let mut rng = StdRng::seed_from_u64(0);
    let mut next_id = 1u64;
    let mut next_birth = 0u64;

    let (child, ok, _) = next_generation::<T, TestOps, D, _>(
        &parent,
        NextGenerationCtx {
            rng: &mut rng,
            dataset: full_dataset,
            temperature: 0.0,
            curmaxsize: 100,
            stats: &stats,
            options: &options,
            evaluator: &mut evaluator,
            next_id: &mut next_id,
            next_birth: &mut next_birth,
            _ops: core::marker::PhantomData,
        },
    );

    assert!(ok, "mutate_operator should succeed even when it is a no-op");
    assert_eq!(
        child.expr.nodes, parent.expr.nodes,
        "with only one operator available, mutate_operator should be a no-op"
    );
}
