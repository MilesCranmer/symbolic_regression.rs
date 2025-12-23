use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::operator_library::OperatorLibrary;
use crate::{Options, equation_search};

#[test]
fn equation_search_runs() {
    let n_rows = 64;
    let n_features = 1;
    let mut x = vec![0.0; n_rows];
    let mut y = vec![0.0; n_rows];
    for i in 0..n_rows {
        let xi = (i as T) / (n_rows as T);
        x[i] = xi;
        y[i] = xi * xi + xi;
    }
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        seed: 123,
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

    let result = equation_search::<T, TestOps, D>(&dataset, &options);
    assert!(result.best.loss.is_finite());
}
