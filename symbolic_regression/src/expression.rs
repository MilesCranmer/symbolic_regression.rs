use std::ops::AddAssign;

pub use dynamic_expressions::{Evaluatable, HasTrees, ScalarConstants};
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::dataset::Dataset;
use crate::operators::Operators;
use crate::options::Options;
use crate::pop_member::Evaluator;

pub trait ConstantOptimizable<T, Ops, const D: usize>: Evaluatable<T, Ops, D> + ScalarConstants<T, Ops, D>
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    #[allow(clippy::too_many_arguments)]
    fn loss_and_grad(
        &mut self,
        plans: &[dynamic_expressions::EvalPlan<D>],
        dataset: &Dataset<T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
        grad_ctx: &mut dynamic_expressions::GradContext<T, D>,
        eval_opts: &dynamic_expressions::EvalOptions,
        grad_out: &mut [f64],
    ) -> Option<f64>
    where
        T: FromPrimitive + ToPrimitive + AddAssign;
}

pub trait Expression<T, Ops, const D: usize>: dynamic_expressions::Expression<T, Ops, D> + Send + Sync
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
}

impl<T, Ops, const D: usize, E> Expression<T, Ops, D> for E
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: dynamic_expressions::Expression<T, Ops, D> + Send + Sync,
{
}

pub trait ExpressionSpec<T, Ops, const D: usize>: Clone + Send + Sync
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    type Expr: Expression<T, Ops, D>;

    fn random_expr(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        dataset_n_features: usize,
        target_size: usize,
        options: &Options<T, D>,
    ) -> Self::Expr;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TreeSpec;

impl<T, Ops, const D: usize> ExpressionSpec<T, Ops, D> for TreeSpec
where
    T: Float + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    type Expr = dynamic_expressions::PostfixExpr<T, Ops, D>;

    fn random_expr(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        dataset_n_features: usize,
        _target_size: usize,
        options: &Options<T, D>,
    ) -> Self::Expr {
        // Match existing initialization behavior.
        crate::mutation_functions::random_expr_append_ops(rng, operators, dataset_n_features, 3usize, options.maxsize)
    }
}

impl<T, Ops, const D: usize> ConstantOptimizable<T, Ops, D> for dynamic_expressions::PostfixExpr<T, Ops, D>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    fn loss_and_grad(
        &mut self,
        plans: &[dynamic_expressions::EvalPlan<D>],
        dataset: &Dataset<T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
        grad_ctx: &mut dynamic_expressions::GradContext<T, D>,
        eval_opts: &dynamic_expressions::EvalOptions,
        grad_out: &mut [f64],
    ) -> Option<f64> {
        use dynamic_expressions::utils::ZipEq;

        let n_params = self.consts.len();
        let n_rows = dataset.n_rows;
        debug_assert_eq!(plans.len(), 1);
        debug_assert_eq!(grad_out.len(), n_params);

        let x = dataset.x.view();
        let (yhat, dy_dc, ok) = dynamic_expressions::eval_grad_tree_array(self, x, false, grad_ctx, eval_opts);
        if !ok || yhat.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let loss = options.loss.loss(
            &yhat,
            dataset.y.as_slice().unwrap(),
            dataset.weights.as_ref().and_then(|w| w.as_slice()),
        );
        if !loss.is_finite() {
            return None;
        }

        // Populate evaluator.yhat for consistency with other expression types.
        evaluator.yhat.clone_from_slice(&yhat);

        // dloss/dyhat
        let mut dloss_dyhat = vec![T::zero(); n_rows];
        options.loss.dloss_dyhat(
            &yhat,
            dataset.y.as_slice().unwrap(),
            dataset.weights.as_ref().and_then(|w| w.as_slice()),
            &mut dloss_dyhat,
        );

        for (ci, gout) in grad_out.iter_mut().enumerate().take(n_params) {
            let base = ci * n_rows;
            let acc = dloss_dyhat
                .iter()
                .copied()
                .zip_eq(dy_dc.data[base..base + n_rows].iter().copied())
                .fold(T::zero(), |a, (dl, dc)| a + dl * dc);
            *gout = acc.to_f64().unwrap_or(f64::INFINITY);
        }

        Some(loss.to_f64().unwrap_or(f64::INFINITY))
    }
}

pub trait ExprExt<T, Ops, const D: usize>: Expression<T, Ops, D>
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    fn complexity(&self, options: &Options<T, D>) -> usize {
        (0..self.n_trees())
            .map(|i| crate::complexity::compute_complexity(&self.tree(i).nodes, options))
            .sum()
    }

    fn check_constraints(&self, options: &Options<T, D>, curmaxsize: usize) -> bool {
        if self.complexity(options) > curmaxsize {
            return false;
        }

        for i in 0..self.n_trees() {
            let tree = self.tree(i);
            if !crate::check_constraints::check_constraints(tree, options, curmaxsize) {
                return false;
            }

            let max_feature = max_feature_index(&tree.nodes);
            if let Some(m) = max_feature {
                // For expressions whose feature visibility is independent of the dataset
                // (e.g., templates), this enforces var bounds. For plain expressions, this is
                // expected to be satisfied by construction/mutation (so use a large sentinel).
                let nfeatures = self.tree_nfeatures(i, usize::MAX);
                if m >= nfeatures {
                    return false;
                }
            }
        }

        true
    }

    fn compress_constants(&mut self) {
        for i in 0..self.n_trees() {
            dynamic_expressions::compress_constants(self.tree_mut(i));
        }
    }

    fn simplify_in_place(&mut self, eval_opts: &dynamic_expressions::EvalOptions) -> bool {
        let mut any = false;
        for i in 0..self.n_trees() {
            any |= dynamic_expressions::simplify_in_place(self.tree_mut(i), eval_opts);
        }
        any
    }

    fn is_leaf(&self) -> bool {
        for i in 0..self.n_trees() {
            let tree = self.tree(i);
            if tree
                .nodes
                .iter()
                .any(|n| matches!(n, dynamic_expressions::PNode::Op { .. }))
            {
                return false;
            }
        }
        true
    }

    fn has_binary_op(&self) -> bool {
        for i in 0..self.n_trees() {
            let tree = self.tree(i);
            if tree
                .nodes
                .iter()
                .any(|n| matches!(n, dynamic_expressions::PNode::Op { arity: 2, .. }))
            {
                return true;
            }
        }
        false
    }

    fn count_constant_nodes(&self) -> usize {
        (0..self.n_trees())
            .map(|i| dynamic_expressions::count_constant_nodes(&self.tree(i).nodes))
            .sum()
    }

    fn feature_mutation_possible(&self, dataset_nfeatures: usize) -> bool {
        for i in 0..self.n_trees() {
            let tree = self.tree(i);
            let nfeatures = self.tree_nfeatures(i, dataset_nfeatures);
            if nfeatures <= 1 {
                continue;
            }
            if tree
                .nodes
                .iter()
                .any(|n| matches!(n, dynamic_expressions::PNode::Var { .. }))
            {
                return true;
            }
        }
        false
    }

    fn mutate_constant(&mut self, rng: &mut Rng, temperature: f64, options: &Options<T, D>) -> bool
    where
        T: FromPrimitive,
    {
        let n = self.n_scalars();
        if n == 0 {
            return false;
        }
        let mut scalars: Vec<T> = Vec::with_capacity(n);
        self.pack_scalars(&mut scalars);
        debug_assert_eq!(scalars.len(), n);

        let idx = rng.usize(0..n);
        mutate_scalar(rng, &mut scalars[idx], temperature, options);
        self.unpack_scalars(&scalars);
        true
    }

    fn randomize(&self, rng: &mut Rng, operators: &Operators<D>, dataset_nfeatures: usize, target_size: usize) -> Self
    where
        T: FromPrimitive,
    {
        let k = self.n_trees().max(1);
        let total = target_size.max(k);

        let mut sizes = vec![1usize; k];
        for _ in 0..(total - k) {
            sizes[rng.usize(0..k)] += 1;
        }

        let mut out = self.clone();
        for i in 0..out.n_trees() {
            let nfeatures = out.tree_nfeatures(i, dataset_nfeatures);
            let sz = sizes.get(i).copied().unwrap_or(1);
            *out.tree_mut(i) = crate::mutation_functions::random_expr(rng, operators, nfeatures, sz);
        }

        // Randomize any additional scalar parameters (and re-randomize const pools) in a derived way.
        let n_scalars = out.n_scalars();
        if n_scalars > 0 {
            let mut scalars: Vec<T> = Vec::with_capacity(n_scalars);
            out.pack_scalars(&mut scalars);
            for v in &mut scalars {
                let r = crate::random::standard_normal(rng);
                *v = T::from_f64(r).unwrap_or_else(T::zero);
            }
            out.unpack_scalars(&scalars);
        }

        out
    }
}

impl<T, Ops, const D: usize, E> ExprExt<T, Ops, D> for E
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: Expression<T, Ops, D>,
{
}

fn max_feature_index(nodes: &[dynamic_expressions::PNode]) -> Option<usize> {
    let mut maxf: Option<usize> = None;
    for n in nodes {
        let dynamic_expressions::PNode::Var { feature } = *n else {
            continue;
        };
        let f = usize::from(feature);
        maxf = Some(maxf.map_or(f, |m| m.max(f)));
    }
    maxf
}

fn mutate_scalar<T: Float + FromPrimitive, const D: usize>(
    rng: &mut Rng,
    v: &mut T,
    temperature: f64,
    options: &Options<T, D>,
) {
    // Follows SymbolicRegression.jl's `mutate_factor` (mirrors `mutate_constant_in_place`).
    let pf = options.perturbation_factor * temperature.max(0.0);
    let max_change = pf + 1.1;
    let exponent: f64 = rng.f64();
    let mut mul = max_change.powf(exponent);
    let make_bigger: bool = rng.bool();
    mul = if make_bigger { mul } else { 1.0 / mul };
    if rng.f64() > options.probability_negate_constant {
        mul = -mul;
    }
    *v = *v * T::from_f64(mul).unwrap_or_else(T::one);
}
