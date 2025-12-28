use std::cmp::Ordering;

use fastrand::Rng;
use num_traits::Float;

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::options::Options;
use crate::pop_member::PopMember;
use crate::population::Population;

fn sample_tournament_place(rng: &mut Rng, n: usize, p: f32) -> usize {
    if n <= 1 || p >= 1.0 {
        return 1;
    }
    if p <= 0.0 {
        panic!("tournament_selection_p must be > 0");
    }
    let q = 1.0 - (p as f64);
    if q <= 0.0 {
        return 1;
    }

    // Tournament sampling is typically called with small `n` (e.g. 10-30).
    // Avoid `ln`/`pow` and just construct the (truncated) geometric weights.
    let mut weights = Vec::with_capacity(n);
    let mut cur = p as f64;
    for _ in 0..n {
        weights.push(cur);
        cur *= q;
    }
    weighted_index(rng, &weights) + 1
}

pub fn best_of_sample<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    pop: &Population<T, Ops, D>,
    stats: &RunningSearchStatistics,
    options: &Options<T, D>,
) -> PopMember<T, Ops, D> {
    let n = options.tournament_selection_n.min(pop.len());
    let indices = sample_indices(rng, pop.len(), n);

    let mut scored: Vec<(f64, usize)> = Vec::with_capacity(n);
    for idx in indices {
        let m = &pop.members[idx];
        let mut adjusted = m.cost.to_f64().unwrap_or(f64::INFINITY);
        if options.use_frequency_in_tournament {
            let size = m.complexity;
            let freq = if size > 0 && size <= options.maxsize {
                stats.normalized_frequencies[size - 1]
            } else {
                0.0
            };
            adjusted *= (options.adaptive_parsimony_scaling * freq).exp();
        }
        scored.push((adjusted, idx));
    }

    let place = sample_tournament_place(rng, scored.len(), options.tournament_selection_p);
    let place_index = (place - 1).min(scored.len() - 1);
    scored.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let chosen = scored[place_index].1;
    pop.members[chosen].clone()
}

pub(crate) fn weighted_index(rng: &mut Rng, weights: &[f64]) -> usize {
    if weights.is_empty() {
        panic!("weights must be non-empty");
    }
    let mut total = 0.0f64;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            panic!("weights must be finite and non-negative");
        }
        total += w;
    }
    if !total.is_finite() || total <= 0.0 {
        panic!("at least one weight must be > 0");
    }

    let mut target = rng.f64() * total;
    for (idx, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        if target < w {
            return idx;
        }
        target -= w;
    }
    weights.len() - 1
}

fn sample_indices(rng: &mut Rng, len: usize, n: usize) -> Vec<usize> {
    let take = n.min(len);
    if len == 0 || n == 0 {
        Vec::new()
    } else if take == len {
        (0..len).collect()
    } else if take * take <= len {
        // Rejection sampling is ~O(take^2) due to linear duplicate checks; partial shuffle is ~O(len).
        // Choose based on which term dominates, avoiding overflow.
        let mut out = Vec::with_capacity(take);
        while out.len() < take {
            let idx = rng.usize(0..len);
            if !out.contains(&idx) {
                out.push(idx);
            }
        }
        out
    } else {
        let mut v: Vec<usize> = (0..len).collect();
        for i in 0..take {
            let j = rng.usize(i..len);
            v.swap(i, j);
        }
        v.truncate(take);
        v
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use fastrand::Rng;

    use super::{sample_indices, weighted_index};

    #[test]
    fn weighted_index_panics_like_weightedindex_new() {
        let run = |weights: &'static [f64]| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rng = Rng::with_seed(0);
                let _ = weighted_index(&mut rng, weights);
            }))
            .is_err()
        };

        assert!(run(&[]));
        assert!(run(&[0.0, 0.0]));
        assert!(run(&[1.0, -1.0]));
        assert!(run(&[f64::NAN, 1.0]));
        assert!(run(&[f64::INFINITY, 1.0]));
    }

    #[test]
    fn sample_indices_are_unique_and_in_range() {
        let mut rng = Rng::with_seed(1);
        for len in [1usize, 2, 3, 10, 100, 10_000] {
            for n in [0usize, 1, 2, 5, 50, len / 2, len] {
                let idx = sample_indices(&mut rng, len, n.min(len));
                assert_eq!(idx.len(), n.min(len));
                for &i in &idx {
                    assert!(i < len);
                }
                let uniq: BTreeSet<usize> = idx.iter().copied().collect();
                assert_eq!(uniq.len(), idx.len());
            }
        }
    }
}
