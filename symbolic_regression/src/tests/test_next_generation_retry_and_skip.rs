use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use fastrand::Rng;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::TaggedDataset;
use crate::operator_library::OperatorLibrary;
use crate::pop_member::{Evaluator, PopMember};
use crate::population::Population;
use crate::{MutationWeights, Options, mutate, regularized_evolution};

fn leaf_expr() -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(vec![PNode::Var { feature: 0 }], Vec::new(), Metadata::default())
}

#[test]
fn next_generation_fails_constraints_after_retries() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 8), vec![0.0; 8]).unwrap(),
        Array1::from_vec(vec![0.0; 8]),
    );

    let weights = MutationWeights {
        mutate_constant: 0.0,
        mutate_operator: 0.0,
        mutate_feature: 0.0,
        swap_operands: 0.0,
        rotate_tree: 0.0,
        add_node: 0.0,
        insert_node: 1.0,
        delete_node: 0.0,
        simplify: 0.0,
        randomize: 0.0,
        do_nothing: 0.0,
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        mutation_weights: weights,
        maxsize: 1,
        maxdepth: 1,
        annealing: false,
        skip_mutation_failures: true,
        tournament_selection_n: 1,
        crossover_probability: 0.0,
        ..Default::default()
    };

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let mut member = PopMember::from_expr_with_birth(0, leaf_expr(), dataset.n_features);
    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
    let _ = member.evaluate(&full_dataset, &options, &mut evaluator);

    let mut rng = Rng::with_seed(0);
    let mut stats = RunningSearchStatistics::new(options.maxsize, 1000);
    stats.normalize();

    let (_baby, accepted, _evals) = mutate::next_generation::<T, TestOps, D>(
        &member,
        mutate::NextGenerationCtx {
            rng: &mut rng,
            dataset: full_dataset,
            temperature: 1.0,
            curmaxsize: 1,
            stats: &stats,
            options: &options,
            evaluator: &mut evaluator,
            _ops: core::marker::PhantomData,
        },
    );
    assert!(!accepted);
}

#[test]
fn reg_evol_cycle_skips_replacement_when_configured() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 8), vec![0.0; 8]).unwrap(),
        Array1::from_vec(vec![0.0; 8]),
    );
    let weights = MutationWeights {
        mutate_constant: 0.0,
        mutate_operator: 0.0,
        mutate_feature: 0.0,
        swap_operands: 0.0,
        rotate_tree: 0.0,
        add_node: 0.0,
        insert_node: 1.0,
        delete_node: 0.0,
        simplify: 0.0,
        randomize: 0.0,
        do_nothing: 0.0,
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        mutation_weights: weights,
        maxsize: 1,
        maxdepth: 1,
        annealing: false,
        skip_mutation_failures: true,
        tournament_selection_n: 1,
        tournament_selection_p: 1.0,
        crossover_probability: 0.0,
        ..Default::default()
    };

    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let baseline_loss = if options.use_baseline {
        crate::loss_functions::baseline_loss_from_zero_expression::<T, TestOps, D>(&dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
    let member = PopMember::from_expr_with_birth(0, leaf_expr(), dataset.n_features);
    let mut pop = Population::new(vec![member]);

    let mut rng = Rng::with_seed(0);
    let mut stats = RunningSearchStatistics::new(options.maxsize, 1000);
    stats.normalize();

    let controller = crate::stop_controller::StopController::from_options(&options);
    let ctx = regularized_evolution::RegEvolCtx::<T, TestOps, D> {
        rng: &mut rng,
        dataset: full_dataset,
        stats: &stats,
        options: &options,
        evaluator: &mut evaluator,
        controller: &controller,
        temperature: 1.0,
        curmaxsize: 1,
        _ops: core::marker::PhantomData,
    };

    regularized_evolution::reg_evol_cycle::<T, TestOps, D>(&mut pop, ctx);
    assert_eq!(pop.members.len(), 1);
}
