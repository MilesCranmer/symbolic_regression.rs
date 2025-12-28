use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use symbolic_regression::{Operators, Options, SearchEngine};

symbolic_regression::custom_opset! {
    struct SlowOps<T = f64>;

    1 => {
        slow_id {
            eval: |[x]| x,
            partial: |[_x]| 1.0
        },
    },
}

#[test]
fn test_timeout_under_max_iterations() {
    const D: usize = 1;
    let dataset = symbolic_regression::Dataset::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::from_vec(vec![1.0]),
    );

    let operators = Operators::<D>::from_names::<SlowOps>(&["slow_id"]).unwrap();

    let options = Options::<f64, D> {
        timeout_in_seconds: 0.05,
        niterations: 1_000_000_000,
        operators,
        ncycles_per_iteration: 1,
        ..Default::default()
    };

    // Start the wall-clock timer before constructing the engine, since the timeout clock starts in
    // `StopController::from_options(...)` during `SearchEngine::new(...)`.
    let start = Instant::now();
    let mut engine = SearchEngine::<f64, SlowOps, D>::new(dataset, options);
    let total_cycles = engine.total_cycles();
    while engine.step(1) > 0 {}
    let elapsed = start.elapsed();

    assert!(engine.is_finished());
    assert!(engine.cycles_completed() < total_cycles);
    assert!(elapsed < Duration::from_secs(2), "timeout test took {:?}", elapsed);
}
