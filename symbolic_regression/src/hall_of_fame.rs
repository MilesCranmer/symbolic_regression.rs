use num_traits::Float;

use crate::check_constraints::check_constraints;
use crate::options::Options;
use crate::pop_member::PopMember;

/// Pareto-style hall-of-fame indexed by (discrete) complexity.
///
/// The engine maintains (at most) one best member per complexity, then derives an approximate
/// Pareto front via [`HallOfFame::pareto_front`].
pub struct HallOfFame<T: Float, Ops, const D: usize> {
    /// Best member at each complexity (1..=max_complexity), with index equal to complexity.
    pub best_by_complexity: Vec<Option<PopMember<T, Ops, D>>>,
}

impl<T: Float, Ops, const D: usize> HallOfFame<T, Ops, D> {
    /// Create a hall-of-fame with the given maximum complexity.
    pub fn new(max_complexity: usize) -> Self {
        Self {
            best_by_complexity: vec![None; max_complexity + 1],
        }
    }

    /// Consider adding `member` to the hall-of-fame.
    pub fn consider(&mut self, member: &PopMember<T, Ops, D>, options: &Options<T, D>, curmaxsize: usize) {
        if !member.loss.is_finite() {
            return;
        }
        if !check_constraints(&member.expr, options, curmaxsize) {
            return;
        }
        let c = member.complexity;
        if c == 0 {
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

    /// Consider a batch of members.
    pub fn consider_members(&mut self, members: &[PopMember<T, Ops, D>], options: &Options<T, D>, curmaxsize: usize) {
        for m in members {
            self.consider(m, options, curmaxsize);
        }
    }

    /// Iterate all stored members (unordered).
    pub fn members(&self) -> impl Iterator<Item = &PopMember<T, Ops, D>> {
        self.best_by_complexity.iter().flatten()
    }

    /// Return a loss-improving sequence of members, scanning from low to high complexity.
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
