use num_traits::Float;

use crate::options::Options;
use crate::pop_member::PopMember;

pub struct HallOfFame<T: Float, Ops, const D: usize> {
    pub best_by_complexity: Vec<Option<PopMember<T, Ops, D>>>,
}

impl<T: Float, Ops, const D: usize> HallOfFame<T, Ops, D> {
    pub fn new(max_complexity: usize) -> Self {
        Self {
            best_by_complexity: vec![None; max_complexity + 1],
        }
    }

    pub fn consider(&mut self, member: &PopMember<T, Ops, D>, options: &Options<T, D>) {
        let c = member.complexity;
        // Match SymbolicRegression.jl `s_r_cycle`: keep entries with `0 < c <= maxsize`.
        if c == 0 || c > options.maxsize {
            return;
        }
        if c >= self.best_by_complexity.len() {
            return;
        }
        match &self.best_by_complexity[c] {
            None => self.best_by_complexity[c] = Some(member.clone()),
            Some(best) => {
                if member
                    .cost
                    .partial_cmp(&best.cost)
                    .unwrap_or(std::cmp::Ordering::Greater)
                    == std::cmp::Ordering::Less
                {
                    self.best_by_complexity[c] = Some(member.clone());
                }
            }
        }
    }

    pub fn update_from_members(&mut self, members: &[PopMember<T, Ops, D>], options: &Options<T, D>) {
        for m in members {
            self.consider(m, options);
        }
    }

    pub fn members(&self) -> impl Iterator<Item = &PopMember<T, Ops, D>> {
        self.best_by_complexity.iter().flatten()
    }

    /// Matches SymbolicRegression.jl `check_for_loss_threshold`: returns true if any existing
    /// member satisfies the callback. Caller passes `(loss, complexity) -> bool`.
    pub fn any_member_satisfies(&self, f: &dyn Fn(T, usize) -> bool) -> bool {
        self.best_by_complexity
            .iter()
            .flatten()
            .any(|m| f(m.loss, m.complexity))
    }

    pub fn pareto_front(&self) -> Vec<PopMember<T, Ops, D>> {
        let mut out = Vec::new();
        let mut best_loss = T::infinity();
        for m in self.best_by_complexity.iter().flatten() {
            if m.loss < best_loss {
                best_loss = m.loss;
                out.push(m.clone());
            }
        }
        out
    }
}
