use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::operator_library::OperatorLibrary;
use crate::{Options, equation_search};

fn make_dataset() -> crate::Dataset<T> {
    let n_rows = 64;
    let n_features = 1;
    let mut x = vec![0.0; n_rows];
    let mut y = vec![0.0; n_rows];
    for i in 0..n_rows {
        let xi = (i as T) / (n_rows as T);
        x[i] = xi;
        y[i] = xi * xi + xi;
    }
    crate::Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    )
}

fn best_signature(res: &crate::SearchResult<T, TestOps, D>) -> (Vec<dynamic_expressions::node::PNode>, Vec<u64>, u64) {
    let nodes = res.best.expr.nodes.clone();
    let const_bits = res.best.expr.consts.iter().map(|c| c.to_bits()).collect();
    let loss_bits = res.best.loss.to_bits();
    (nodes, const_bits, loss_bits)
}

#[test]
fn deterministic_parallel_mode_is_deterministic_even_with_jittered_scheduling() {
    let _guard = crate::search_utils::test_hooks::exclusive_guard();

    let dataset = make_dataset();
    let options = Options::<T, D> {
        seed: 123,
        deterministic: true,
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        populations: 2,
        population_size: 40,
        niterations: 2,
        ncycles_per_iteration: 10,
        maxsize: 10,
        maxdepth: 8,
        migration: false,
        hof_migration: false,
        optimizer_probability: 0.0,
        ..Default::default()
    };

    crate::pop_member::reset_pseudo_time_for_tests();
    crate::search_utils::test_hooks::set_pop_jitter_ms(vec![25, 0]);
    let res1 = equation_search::<T, TestOps, D>(&dataset, &options);

    crate::pop_member::reset_pseudo_time_for_tests();
    crate::search_utils::test_hooks::set_pop_jitter_ms(vec![0, 25]);
    let res2 = equation_search::<T, TestOps, D>(&dataset, &options);

    assert_eq!(best_signature(&res1), best_signature(&res2));
}

#[test]
fn deterministic_parallel_mode_uses_multiple_worker_threads_when_available() {
    let _guard = crate::search_utils::test_hooks::exclusive_guard();

    if rayon::current_num_threads() < 2 {
        return;
    }

    let dataset = make_dataset();
    let options = Options::<T, D> {
        seed: 123,
        deterministic: true,
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        populations: 2,
        population_size: 40,
        niterations: 1,
        ncycles_per_iteration: 10,
        maxsize: 10,
        maxdepth: 8,
        migration: false,
        hof_migration: false,
        optimizer_probability: 0.0,
        ..Default::default()
    };

    crate::pop_member::reset_pseudo_time_for_tests();
    crate::search_utils::test_hooks::set_pop_jitter_ms(vec![50, 50]);
    let res = equation_search::<T, TestOps, D>(&dataset, &options);
    assert!(res.best.loss.is_finite());
    assert!(
        crate::search_utils::test_hooks::max_active_tasks() >= 2,
        "expected >=2 active tasks, got {}",
        crate::search_utils::test_hooks::max_active_tasks()
    );
}
