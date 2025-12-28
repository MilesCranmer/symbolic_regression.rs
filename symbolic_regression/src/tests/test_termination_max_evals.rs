use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::Options;
use crate::operator_library::OperatorLibrary;
use crate::search_utils::SearchEngine;

#[test]
fn search_engine_honors_max_evals_budget() {
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        use_baseline: false,
        niterations: 10,
        populations: 2,
        population_size: 2,
        max_evals: 4, // exactly the initial population evaluation budget
        ..Default::default()
    };

    let mut engine = SearchEngine::<T, TestOps, D>::new(dataset, options);
    let completed = engine.step(1000);

    assert_eq!(completed, 0);
    assert!(engine.is_finished());
    assert_eq!(engine.cycles_completed(), 0);
    assert_eq!(engine.total_evals(), 4);
}
