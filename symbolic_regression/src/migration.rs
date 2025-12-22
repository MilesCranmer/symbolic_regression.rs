use crate::pop_member::{MemberId, PopMember};
use crate::population::Population;
use num_traits::Float;
use rand::seq::IndexedRandom;
use rand::Rng;
use rand_distr::{Distribution, Poisson};
use std::cmp::Ordering;

pub fn best_sub_pop<T: Float, Ops, const D: usize>(
    pop: &Population<T, Ops, D>,
    topn: usize,
) -> Vec<PopMember<T, Ops, D>> {
    let mut idxs: Vec<usize> = (0..pop.len()).collect();
    idxs.sort_by(|&i, &j| {
        pop.members[i]
            .cost
            .partial_cmp(&pop.members[j].cost)
            .unwrap_or(Ordering::Greater)
    });
    idxs.truncate(topn.min(idxs.len()));
    idxs.into_iter().map(|i| pop.members[i].clone()).collect()
}

fn poisson_sample<R: Rng + ?Sized>(rng: &mut R, mean: f64) -> usize {
    if !mean.is_finite() || mean <= 0.0 {
        return 0;
    }
    let dist = Poisson::new(mean).expect("invalid poisson mean");
    dist.sample(rng) as usize
}

pub fn migrate_into<T: Float, Ops, const D: usize, R: Rng + ?Sized>(
    dst: &mut Population<T, Ops, D>,
    migrants: &[PopMember<T, Ops, D>],
    frac: f64,
    rng: &mut R,
    next_id: &mut u64,
    next_birth: &mut u64,
) {
    if migrants.is_empty() {
        return;
    }
    if frac <= 0.0 {
        return;
    }
    if dst.is_empty() {
        return;
    }

    let mean = (dst.len() as f64) * frac;
    let mut n_replace = poisson_sample(rng, mean);
    n_replace = n_replace.min(dst.len()).min(migrants.len());
    if n_replace == 0 {
        return;
    }

    for _ in 0..n_replace {
        let loc = rng.random_range(0..dst.len());
        let src = migrants.choose(rng).expect("migrants is non-empty");
        let mut m = src.clone();
        m.parent = Some(src.id);
        m.id = MemberId(*next_id);
        *next_id += 1;
        m.birth = *next_birth;
        *next_birth += 1;
        dst.members[loc] = m;
    }
}
