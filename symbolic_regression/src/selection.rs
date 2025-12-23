use std::cmp::Ordering;

use num_traits::Float;
use rand::Rng;
use rand::distr::{self, Distribution};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::options::Options;
use crate::pop_member::PopMember;
use crate::population::Population;

fn sample_tournament_place<R: Rng>(rng: &mut R, n: usize, p: f32) -> usize {
    if n <= 1 || p >= 1.0 {
        return 1;
    }
    let mut weights = Vec::with_capacity(n);
    let mut cur = p as f64;
    let q = 1.0 - (p as f64);
    for _ in 0..n {
        weights.push(cur);
        cur *= q;
    }
    let dist = distr::weighted::WeightedIndex::new(weights).expect("tournament weights must be positive");
    dist.sample(rng) + 1
}

pub fn best_of_sample<T: Float, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    pop: &Population<T, Ops, D>,
    stats: &RunningSearchStatistics,
    options: &Options<T, D>,
) -> PopMember<T, Ops, D> {
    let n = options.tournament_selection_n.min(pop.len());
    let indices = rand::seq::index::sample(rng, pop.len(), n).into_vec();

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
    scored.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let chosen = scored[(place - 1).min(scored.len() - 1)].1;
    pop.members[chosen].clone()
}
