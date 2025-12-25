use std::cmp::Ordering;

use dynamic_expressions::utils::ZipEq;

#[derive(Clone, Debug)]
pub struct RunningSearchStatistics {
    pub window_size: f64,
    pub frequencies: Vec<f64>,
    pub normalized_frequencies: Vec<f64>,
}

impl RunningSearchStatistics {
    pub fn new(maxsize: usize, window_size: usize) -> Self {
        let init = vec![1.0; maxsize];
        let sum = init.iter().sum::<f64>();
        Self {
            window_size: window_size as f64,
            frequencies: init.clone(),
            normalized_frequencies: init.into_iter().map(|v| v / sum).collect(),
        }
    }

    pub fn update_frequencies(&mut self, size: usize) {
        if size > 0 && size <= self.frequencies.len() {
            self.frequencies[size - 1] += 1.0;
        }
    }

    pub fn update_from_population(&mut self, sizes: impl IntoIterator<Item = usize>) {
        for s in sizes {
            self.update_frequencies(s);
        }
    }

    pub fn move_window(&mut self) {
        let smallest_allowed = 1.0;
        let max_loops = 1000;

        let cur_sum: f64 = self.frequencies.iter().sum();
        if cur_sum <= self.window_size {
            return;
        }

        let mut remaining = cur_sum - self.window_size;
        let mut loops = 0;
        while remaining > 0.0 {
            let indices: Vec<usize> = self
                .frequencies
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v > smallest_allowed).then_some(i))
                .collect();
            if indices.is_empty() {
                break;
            }
            let num_remaining = indices.len() as f64;
            let min_above = indices
                .iter()
                .map(|&i| self.frequencies[i])
                .fold(f64::INFINITY, f64::min);
            let amount = (remaining / num_remaining).min(min_above - smallest_allowed);
            if amount.partial_cmp(&0.0) != Some(Ordering::Greater) {
                break;
            }
            for &i in &indices {
                self.frequencies[i] -= amount;
            }
            let total = amount * num_remaining;
            remaining -= total;
            loops += 1;
            if loops > max_loops || total < 1e-6 {
                break;
            }
        }
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.frequencies.iter().sum();
        if sum == 0.0 {
            self.normalized_frequencies.fill(0.0);
            return;
        }
        if self.normalized_frequencies.len() != self.frequencies.len() {
            self.normalized_frequencies.resize(self.frequencies.len(), 0.0);
        }
        for (o, &v) in self.normalized_frequencies.iter_mut().zip_eq(&self.frequencies) {
            *o = v / sum;
        }
    }

    pub fn freq(&self, size: usize) -> f64 {
        if size == 0 || size > self.normalized_frequencies.len() {
            1e-6
        } else {
            self.normalized_frequencies[size - 1].max(1e-6)
        }
    }
}
