use std::cmp::Ordering;

use fastrand::Rng;
use num_traits::Float;

use crate::pop_member::{MemberId, PopMember};
use crate::population::Population;
use crate::random::{choose, poisson_sample, usize_range};

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

pub fn migrate_into<T: Float, Ops, const D: usize>(
    dst: &mut Population<T, Ops, D>,
    migrants: &[PopMember<T, Ops, D>],
    frac: f64,
    rng: &mut Rng,
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
    if !mean.is_finite() {
        return;
    }
    let mut n_replace = poisson_sample(rng, mean);
    n_replace = n_replace.min(dst.len()).min(migrants.len());
    if n_replace == 0 {
        return;
    }

    for _ in 0..n_replace {
        let loc = usize_range(rng, 0..dst.len());
        let src = choose(rng, migrants).expect("migrants is non-empty");
        let mut m = src.clone();
        m.parent = Some(src.id);
        m.id = MemberId(*next_id);
        *next_id += 1;
        m.birth = *next_birth;
        *next_birth += 1;
        dst.members[loc] = m;
    }
}
