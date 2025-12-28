use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use crate::expression::PostfixExpr;
use crate::traits::OperatorSet;
use crate::{compile, evaluate};

pub trait HasTrees<T, Ops, const D: usize>: Clone
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn n_trees(&self) -> usize {
        1
    }

    fn tree(&self, i: usize) -> &PostfixExpr<T, Ops, D>;
    fn tree_mut(&mut self, i: usize) -> &mut PostfixExpr<T, Ops, D>;

    fn tree_nfeatures(&self, _i: usize, dataset_nfeatures: usize) -> usize {
        dataset_nfeatures
    }
}

pub trait Evaluatable<T, Ops, const D: usize>: HasTrees<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn build_plans(&self, dataset_nfeatures: usize) -> Vec<compile::EvalPlan<D>> {
        let mut out: Vec<compile::EvalPlan<D>> = Vec::with_capacity(self.n_trees());
        for i in 0..self.n_trees() {
            let tree = self.tree(i);
            let nfeatures = self.tree_nfeatures(i, dataset_nfeatures);
            out.push(compile::compile_plan::<D>(&tree.nodes, nfeatures, tree.consts.len()));
        }
        out
    }

    fn eval_with_plans(
        &self,
        plans: &[compile::EvalPlan<D>],
        x: ArrayView2<'_, T>,
        out: &mut [T],
        scratch: &mut Array2<T>,
        eval_options: &evaluate::EvalOptions,
    ) -> bool {
        debug_assert_eq!(plans.len(), self.n_trees());
        debug_assert!(self.n_trees() == 1, "default eval_with_plans expects exactly one tree");

        evaluate::eval_plan_array_into::<T, Ops, D>(out, &plans[0], self.tree(0), x, scratch, eval_options)
    }
}

pub trait ScalarConstants<T, Ops, const D: usize>: HasTrees<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn n_scalars(&self) -> usize {
        (0..self.n_trees()).map(|i| self.tree(i).consts.len()).sum()
    }

    fn pack_scalars(&self, out: &mut Vec<T>) {
        out.clear();
        out.reserve(self.n_scalars());
        for i in 0..self.n_trees() {
            out.extend_from_slice(&self.tree(i).consts);
        }
    }

    fn unpack_scalars(&mut self, scalars: &[T]) {
        assert_eq!(scalars.len(), self.n_scalars(), "scalar constants length mismatch");
        let mut cursor = 0usize;
        for i in 0..self.n_trees() {
            let tree = self.tree_mut(i);
            let n = tree.consts.len();
            tree.consts.clone_from_slice(&scalars[cursor..cursor + n]);
            cursor += n;
        }
        debug_assert_eq!(cursor, scalars.len());
    }
}

pub trait Expression<T, Ops, const D: usize>:
    HasTrees<T, Ops, D> + Evaluatable<T, Ops, D> + ScalarConstants<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
}

impl<T, Ops, const D: usize, E> Expression<T, Ops, D> for E
where
    T: Float,
    Ops: OperatorSet<T = T>,
    E: HasTrees<T, Ops, D> + Evaluatable<T, Ops, D> + ScalarConstants<T, Ops, D>,
{
}

impl<T, Ops, const D: usize> HasTrees<T, Ops, D> for PostfixExpr<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn tree(&self, i: usize) -> &PostfixExpr<T, Ops, D> {
        assert_eq!(i, 0);
        self
    }

    fn tree_mut(&mut self, i: usize) -> &mut PostfixExpr<T, Ops, D> {
        assert_eq!(i, 0);
        self
    }
}

impl<T, Ops, const D: usize> Evaluatable<T, Ops, D> for PostfixExpr<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
}

impl<T, Ops, const D: usize> ScalarConstants<T, Ops, D> for PostfixExpr<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
}
