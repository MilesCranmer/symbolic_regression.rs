use num_traits::Float;

use crate::expression::ExprExt;
use crate::options::Options;
use crate::pop_member::PopMember;

pub struct HallOfFame<T: Float, Ops, const D: usize, E = dynamic_expressions::PostfixExpr<T, Ops, D>>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D>,
{
    pub best_by_complexity: Vec<Option<PopMember<T, Ops, D, E>>>,
}

impl<T: Float, Ops, const D: usize, E> HallOfFame<T, Ops, D, E>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D>,
{
    pub fn new(max_complexity: usize) -> Self {
        Self {
            best_by_complexity: vec![None; max_complexity + 1],
        }
    }

    pub fn consider(&mut self, member: &PopMember<T, Ops, D, E>, options: &Options<T, D>, curmaxsize: usize) {
        if !member.expr.check_constraints(options, curmaxsize) {
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

    pub fn update_from_members(
        &mut self,
        members: &[PopMember<T, Ops, D, E>],
        options: &Options<T, D>,
        curmaxsize: usize,
    ) {
        for m in members {
            self.consider(m, options, curmaxsize);
        }
    }

    pub fn members(&self) -> impl Iterator<Item = &PopMember<T, Ops, D, E>> {
        self.best_by_complexity.iter().flatten()
    }

    pub fn pareto_front(&self) -> Vec<PopMember<T, Ops, D, E>> {
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
