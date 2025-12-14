use ndarray::{Array1, Array2};
use num_traits::Float;

#[derive(Clone, Debug)]
pub struct Dataset<T: Float> {
    /// Row-major contiguous data with shape `(n_rows, n_features)` (ndarray default).
    pub x: Array2<T>,
    /// Target vector with length `n_rows`.
    pub y: Array1<T>,
    pub n_features: usize,
    pub n_rows: usize,
    pub weights: Option<Array1<T>>,
    pub variable_names: Vec<String>,
    pub baseline_loss: T,
}

impl<T: Float> Dataset<T> {
    pub fn new(x: Array2<T>, y: Array1<T>) -> Self {
        let x = x.as_standard_layout().to_owned();
        let (n_rows, n_features) = x.dim();
        assert_eq!(y.len(), n_rows);

        let baseline_loss = Self::compute_baseline_mse(y.as_slice().unwrap(), None);

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights: None,
            variable_names: Vec::new(),
            baseline_loss,
        }
    }

    pub fn with_weights_and_names(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
    ) -> Self {
        let x = x.as_standard_layout().to_owned();
        let (n_rows, n_features) = x.dim();
        assert_eq!(y.len(), n_rows);
        if let Some(w) = &weights {
            assert_eq!(w.len(), n_rows);
        }

        let baseline_loss = Self::compute_baseline_mse(
            y.as_slice().unwrap(),
            weights.as_ref().and_then(|w| w.as_slice()),
        );

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights,
            variable_names,
            baseline_loss,
        }
    }

    pub fn compute_baseline_mse(y: &[T], weights: Option<&[T]>) -> T {
        if y.is_empty() {
            return T::zero();
        }
        match weights {
            None => {
                let n = T::from(y.len()).unwrap();
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / n;
                y.iter()
                    .copied()
                    .map(|v| {
                        let r = v - mean;
                        r * r
                    })
                    .fold(T::zero(), |a, b| a + b)
                    / n
            }
            Some(w) => {
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    return T::zero();
                }
                let mean = y
                    .iter()
                    .copied()
                    .zip(w.iter().copied())
                    .map(|(yi, wi)| yi * wi)
                    .fold(T::zero(), |a, b| a + b)
                    / sum_w;
                y.iter()
                    .copied()
                    .zip(w.iter().copied())
                    .map(|(yi, wi)| {
                        let r = yi - mean;
                        wi * r * r
                    })
                    .fold(T::zero(), |a, b| a + b)
                    / sum_w
            }
        }
    }
}
