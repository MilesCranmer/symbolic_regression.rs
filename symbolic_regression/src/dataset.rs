use dynamic_expressions::utils::ZipEq;
use fastrand::Rng;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::random::usize_range;

#[derive(Copy, Clone, Debug)]
/// A [`Dataset`] paired with an optional baseline loss.
///
/// This is internal search-engine state when `Options::use_baseline` is enabled.
pub(crate) struct TaggedDataset<'a, T: Float> {
    pub data: &'a Dataset<T>,
    pub baseline_loss: Option<T>,
}

impl<'a, T: Float> TaggedDataset<'a, T> {
    /// Create a tagged dataset.
    pub(crate) fn new(data: &'a Dataset<T>, baseline_loss: Option<T>) -> Self {
        Self { data, baseline_loss }
    }
}

impl<'a, T: Float> std::ops::Deref for TaggedDataset<'a, T> {
    type Target = Dataset<T>;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

#[derive(Clone, Debug)]
/// A supervised regression dataset used by the symbolic regression engine.
///
/// - `x` is expected to have shape `(n_features, n_rows)` (i.e. column-major with each column a row / sample).
/// - `y` has length `n_rows`.
///
/// Some internals assume `x` and `y` are contiguous; constructors will copy as needed.
pub struct Dataset<T: Float> {
    /// Column-major contiguous data with shape `(n_features, n_rows)` for vectorization over rows.
    pub x: Array2<T>,
    /// Target vector with length `n_rows`.
    pub y: Array1<T>,
    pub n_features: usize,
    pub n_rows: usize,
    pub weights: Option<Array1<T>>,
    pub variable_names: Vec<String>,
    /// Weighted mean of `y` (or unweighted mean when no weights).
    pub avg_y: T,
}

impl<T: Float> Dataset<T> {
    fn build_dataset(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
        avg_y: Option<T>,
    ) -> Self {
        let x = x.as_standard_layout().to_owned();
        let (n_features, n_rows) = x.dim();
        assert_eq!(y.len(), n_rows);
        if let Some(ref w) = weights {
            assert_eq!(w.len(), n_rows);
        }

        let avg_y = avg_y
            .unwrap_or_else(|| Self::compute_avg_y(y.as_slice().unwrap(), weights.as_ref().and_then(|w| w.as_slice())));

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights,
            variable_names,
            avg_y,
        }
    }

    /// Create a dataset without weights or variable names.
    pub fn new(x: Array2<T>, y: Array1<T>) -> Self {
        Self::build_dataset(x, y, None, Vec::new(), None)
    }

    /// Create a dataset with optional per-row weights and variable names.
    ///
    /// When provided, `weights` must have length `n_rows`.
    pub fn with_weights_and_names(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
    ) -> Self {
        Self::build_dataset(x, y, weights, variable_names, None)
    }

    pub(crate) fn y_slice(&self) -> &[T] {
        self.y.as_slice().expect("y is contiguous")
    }

    pub(crate) fn weights_slice(&self) -> Option<&[T]> {
        self.weights.as_ref().and_then(|w| w.as_slice())
    }

    pub(crate) fn compute_avg_y(y: &[T], weights: Option<&[T]>) -> T {
        if y.is_empty() {
            return T::zero();
        }
        match weights {
            None => {
                let n = T::from(y.len()).unwrap();
                y.iter().copied().fold(T::zero(), |a, b| a + b) / n
            }
            Some(w) => {
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                y.iter()
                    .copied()
                    .zip_eq(w.iter().copied())
                    .map(|(yi, wi)| yi * wi)
                    .fold(T::zero(), |a, b| a + b)
                    / sum_w
            }
        }
    }

    /// Create a dataset-shaped buffer used for batching.
    ///
    /// This preserves the feature count and (optionally) the presence of weights.
    pub(crate) fn make_batch_buffer(full: &Dataset<T>, batch_size: usize) -> Dataset<T> {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        let batch_size = batch_size.max(1);
        let x = Array2::<T>::zeros((full.n_features, batch_size));
        let y = Array1::<T>::zeros(batch_size);
        let weights = full.weights.as_ref().map(|_| Array1::<T>::zeros(batch_size));
        Self::build_dataset(x, y, weights, full.variable_names.clone(), Some(full.avg_y))
    }

    /// Resample rows (with replacement) from `full` into `self`.
    ///
    /// `self` must be a buffer created via [`Dataset::make_batch_buffer`] (same feature count,
    /// batch size, and weight presence).
    pub(crate) fn resample_from(&mut self, full: &Dataset<T>, rng: &mut Rng) {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        assert_eq!(self.n_features, full.n_features);
        assert_eq!(self.x.dim().0, self.n_features);
        assert_eq!(self.x.dim().1, self.n_rows);
        assert_eq!(self.y.len(), self.n_rows);
        if let Some(w) = &self.weights {
            assert_eq!(w.len(), self.n_rows);
            assert!(full.weights.is_some());
        } else {
            assert!(full.weights.is_none());
        }

        for (dst_col, src_idx) in (0..self.n_rows).map(|i| (i, usize_range(rng, 0..full.n_rows))) {
            self.x.column_mut(dst_col).assign(&full.x.column(src_idx));
            self.y[dst_col] = full.y[src_idx];
            if let (Some(dst), Some(src)) = (self.weights.as_mut(), full.weights.as_ref()) {
                dst[dst_col] = src[src_idx];
            }
        }
    }
}
