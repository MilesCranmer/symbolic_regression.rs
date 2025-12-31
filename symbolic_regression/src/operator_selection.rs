use dynamic_expressions::{OpId, Operators};
use fastrand::Rng;

use crate::random::usize_range;

/// Sampling helpers for choosing operators by arity during mutations, implemented for
/// [`dynamic_expressions::Operators`] and useful for custom search loops.
pub(crate) trait OperatorsSampling {
    /// Total number of operators with arity in `1..=max_arity` (clamped to the implementation's
    /// maximum arity).
    fn total_ops_up_to(&self, max_arity: usize) -> usize;
    /// Sample an arity in `1..=max_arity` proportional to the number of operators at that arity.
    fn sample_arity(&self, rng: &mut Rng, max_arity: usize) -> usize;
    /// Sample an operator `OpId` with the given arity.
    fn sample_op(&self, rng: &mut Rng, arity: usize) -> OpId;
}

impl<const D: usize> OperatorsSampling for Operators<D> {
    fn total_ops_up_to(&self, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        (1..=max_arity).map(|a| self.nops(a)).sum()
    }

    fn sample_arity(&self, rng: &mut Rng, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        let total: usize = (1..=max_arity).map(|a| self.nops(a)).sum();
        assert!(total > 0, "no operators available up to arity={max_arity}");
        let mut r = usize_range(rng, 0..total);
        for arity in 1..=max_arity {
            let n = self.nops(arity);
            if r < n {
                return arity;
            }
            r -= n;
        }
        unreachable!()
    }

    fn sample_op(&self, rng: &mut Rng, arity: usize) -> OpId {
        let v = &self.ops_by_arity[arity - 1];
        let i = usize_range(rng, 0..v.len());
        v[i]
    }
}
