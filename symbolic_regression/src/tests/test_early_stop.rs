use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::operator_library::OperatorLibrary;
use crate::search_utils::SearchEngine;
use crate::{EarlyStop, Options};

#[test]
fn search_terminates_when_early_stop_condition_satisfied() {
    // y == 0 everywhere, so the trivial zero-constant member already in the initial population
    // produces loss == 0. Any non-trivial early-stop threshold should fire immediately.
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        use_baseline: false,
        niterations: 100,
        populations: 2,
        population_size: 4,
        ncycles_per_iteration: 5,
        early_stop_condition: Some(EarlyStop::below::<T>(1e-6)),
        ..Default::default()
    };

    let mut engine = SearchEngine::<T, TestOps, D>::new(dataset, options);
    engine.step(usize::MAX);

    // The search should bail out long before the full niterations budget.
    assert!(engine.is_finished());
    assert!(engine.cycles_completed() < 100 * 2);
}
