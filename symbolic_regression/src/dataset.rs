use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::mem;

use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::Rng;

#[derive(Copy, Clone, Debug)]
pub struct TaggedDataset<'a, T: Float> {
    pub data: &'a Dataset<T>,
    pub baseline_loss: Option<T>,
}

impl<'a, T: Float> TaggedDataset<'a, T> {
    pub fn new(data: &'a Dataset<T>, baseline_loss: Option<T>) -> Self {
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
    /// Hash of dataset values (x/y/weights), used to invalidate cached evaluations.
    pub x_key: u64,
}

impl<T: Float> Dataset<T> {
    fn hash_slice(hasher: &mut DefaultHasher, slice: &[T]) {
        let len_bytes = slice.len() * mem::size_of_val(slice);
        // SAFETY: The slice is contiguous and we only hash raw bytes.
        let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len_bytes) };
        hasher.write(bytes);
    }

    fn compute_x_key(x: &Array2<T>, y: &Array1<T>, weights: Option<&Array1<T>>) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write_usize(x.nrows());
        hasher.write_usize(x.ncols());
        let x_data = x.as_slice().expect("x is contiguous");
        let y_data = y.as_slice().expect("y is contiguous");
        Self::hash_slice(&mut hasher, x_data);
        Self::hash_slice(&mut hasher, y_data);
        if let Some(w) = weights {
            hasher.write_u8(1);
            let w_data = w.as_slice().expect("weights is contiguous");
            Self::hash_slice(&mut hasher, w_data);
        } else {
            hasher.write_u8(0);
        }
        hasher.finish()
    }

    pub fn new(x: Array2<T>, y: Array1<T>) -> Self {
        let x = x.as_standard_layout().to_owned();
        let (n_features, n_rows) = x.dim();
        assert_eq!(y.len(), n_rows);

        let avg_y = Self::compute_avg_y(y.as_slice().unwrap(), None);
        let x_key = Self::compute_x_key(&x, &y, None);

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights: None,
            variable_names: Vec::new(),
            avg_y,
            x_key,
        }
    }

    pub fn with_weights_and_names(
        x: Array2<T>,
        y: Array1<T>,
        weights: Option<Array1<T>>,
        variable_names: Vec<String>,
    ) -> Self {
        let x = x.as_standard_layout().to_owned();
        let (n_features, n_rows) = x.dim();
        assert_eq!(y.len(), n_rows);
        if let Some(w) = &weights {
            assert_eq!(w.len(), n_rows);
        }

        let avg_y = Self::compute_avg_y(y.as_slice().unwrap(), weights.as_ref().and_then(|w| w.as_slice()));
        let x_key = Self::compute_x_key(&x, &y, weights.as_ref());

        Self {
            x,
            y,
            n_features,
            n_rows,
            weights,
            variable_names,
            avg_y,
            x_key,
        }
    }

    pub fn y_slice(&self) -> &[T] {
        self.y.as_slice().expect("y is contiguous")
    }

    pub fn weights_slice(&self) -> Option<&[T]> {
        self.weights.as_ref().and_then(|w| w.as_slice())
    }

    pub fn compute_avg_y(y: &[T], weights: Option<&[T]>) -> T {
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
                    .zip(w.iter().copied())
                    .map(|(yi, wi)| yi * wi)
                    .fold(T::zero(), |a, b| a + b)
                    / sum_w
            }
        }
    }

    pub fn make_batch_buffer(full: &Dataset<T>, batch_size: usize) -> Dataset<T> {
        if full.n_rows == 0 {
            panic!("Cannot batch from an empty dataset (n_rows = 0).");
        }
        let batch_size = batch_size.max(1);
        let x = Array2::<T>::zeros((full.n_features, batch_size));
        let y = Array1::<T>::zeros(batch_size);
        let weights = full.weights.as_ref().map(|_| Array1::<T>::zeros(batch_size));
        let x_key = Self::compute_x_key(&x, &y, weights.as_ref());
        Dataset {
            x,
            y,
            n_features: full.n_features,
            n_rows: batch_size,
            weights,
            variable_names: full.variable_names.clone(),
            avg_y: full.avg_y,
            x_key,
        }
    }

    pub fn resample_from(&mut self, full: &Dataset<T>, rng: &mut impl Rng) {
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

        for i in 0..self.n_rows {
            let idx = rng.random_range(0..full.n_rows);
            for f in 0..self.n_features {
                self.x[(f, i)] = full.x[(f, idx)];
            }
            self.y[i] = full.y[idx];
            if let (Some(dst), Some(src)) = (self.weights.as_mut(), full.weights.as_ref()) {
                dst[i] = src[idx];
            }
        }
        self.x_key = Self::compute_x_key(&self.x, &self.y, self.weights.as_ref());
    }
}
