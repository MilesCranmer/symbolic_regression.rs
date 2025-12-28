use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::Dataset;
use crate::operators::Operators;
use crate::options::Options;
use crate::pop_member::Evaluator;

/// Abstraction over candidate expressions searched by symbolic regression.
///
/// The default implementation is `dynamic_expressions::PostfixExpr`, but composite
/// expressions (e.g., `TemplateExpression`) can participate by implementing this trait.
pub trait SRExpression<T, Ops, const D: usize>: Clone + Send + Sync
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    type Plan: Clone + Send + Sync + 'static;
    type MutationContext: Copy + Send + Sync + 'static;

    fn build_plan(&self, dataset_n_features: usize) -> Self::Plan;

    fn eval_with_plan(
        &self,
        plan: &Self::Plan,
        x: ndarray::ArrayView2<'_, T>,
        evaluator: &mut Evaluator<T, D>,
        eval_options: &dynamic_expressions::EvalOptions,
    ) -> bool;

    fn complexity(&self, options: &Options<T, D>) -> usize;

    fn check_constraints(&self, options: &Options<T, D>, curmaxsize: usize) -> bool;

    fn compress_constants(&mut self);

    fn simplify_in_place(&mut self, eval_opts: &dynamic_expressions::EvalOptions) -> bool;

    fn get_contents_for_mutation(
        &self,
        rng: &mut Rng,
    ) -> (dynamic_expressions::PostfixExpr<T, Ops, D>, Self::MutationContext);

    fn with_contents_for_mutation(
        &self,
        mutated: dynamic_expressions::PostfixExpr<T, Ops, D>,
        ctx: Self::MutationContext,
    ) -> Self;

    fn nfeatures_for_mutation(&self, ctx: Self::MutationContext, dataset_n_features: usize) -> usize;

    fn feature_mutation_possible(&self, dataset_n_features: usize) -> bool;

    fn is_leaf(&self) -> bool;
    fn has_binary_op(&self) -> bool;

    fn count_constant_nodes(&self) -> usize;

    fn count_scalar_constants(&self) -> usize;
    fn get_scalar_constants_flat(&self, out: &mut Vec<T>);
    fn set_scalar_constants_flat(&mut self, values: &[T]);

    #[allow(clippy::too_many_arguments)]
    fn loss_and_grad(
        &mut self,
        plan: &Self::Plan,
        dataset: &Dataset<T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
        grad_ctx: &mut dynamic_expressions::GradContext<T, D>,
        eval_opts: &dynamic_expressions::EvalOptions,
        grad_out: &mut [f64],
    ) -> Option<f64>
    where
        T: FromPrimitive + ToPrimitive + core::ops::AddAssign;

    fn mutate_constant(&mut self, rng: &mut Rng, temperature: f64, options: &Options<T, D>) -> bool;

    fn randomize(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        dataset_n_features: usize,
        target_size: usize,
        options: &Options<T, D>,
    ) -> Self;
}

pub trait ExpressionSpec<T, Ops, const D: usize>: Clone + Send + Sync
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    type Expr: SRExpression<T, Ops, D>;

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

impl<T, Ops, const D: usize> SRExpression<T, Ops, D> for dynamic_expressions::PostfixExpr<T, Ops, D>
where
    T: Float + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    type Plan = dynamic_expressions::EvalPlan<D>;
    type MutationContext = ();

    fn build_plan(&self, dataset_n_features: usize) -> Self::Plan {
        dynamic_expressions::compile_plan(&self.nodes, dataset_n_features, self.consts.len())
    }

    fn eval_with_plan(
        &self,
        plan: &Self::Plan,
        x: ndarray::ArrayView2<'_, T>,
        evaluator: &mut Evaluator<T, D>,
        eval_options: &dynamic_expressions::EvalOptions,
    ) -> bool {
        dynamic_expressions::eval_plan_array_into(
            &mut evaluator.yhat,
            plan,
            self,
            x,
            &mut evaluator.scratch,
            eval_options,
        )
    }

    fn complexity(&self, options: &Options<T, D>) -> usize {
        crate::complexity::compute_complexity(&self.nodes, options)
    }

    fn check_constraints(&self, options: &Options<T, D>, curmaxsize: usize) -> bool {
        crate::check_constraints::check_constraints(self, options, curmaxsize)
    }

    fn compress_constants(&mut self) {
        dynamic_expressions::compress_constants(self);
    }

    fn simplify_in_place(&mut self, eval_opts: &dynamic_expressions::EvalOptions) -> bool {
        dynamic_expressions::simplify_in_place(self, eval_opts)
    }

    fn get_contents_for_mutation(
        &self,
        _rng: &mut Rng,
    ) -> (dynamic_expressions::PostfixExpr<T, Ops, D>, Self::MutationContext) {
        (self.clone(), ())
    }

    fn with_contents_for_mutation(
        &self,
        mutated: dynamic_expressions::PostfixExpr<T, Ops, D>,
        _ctx: Self::MutationContext,
    ) -> Self {
        mutated
    }

    fn nfeatures_for_mutation(&self, _ctx: Self::MutationContext, dataset_n_features: usize) -> usize {
        dataset_n_features
    }

    fn feature_mutation_possible(&self, dataset_n_features: usize) -> bool {
        dataset_n_features > 1
    }

    fn is_leaf(&self) -> bool {
        self.nodes.iter().all(|n| {
            matches!(
                n,
                dynamic_expressions::PNode::Var { .. } | dynamic_expressions::PNode::Const { .. }
            )
        })
    }

    fn has_binary_op(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n, dynamic_expressions::PNode::Op { arity: 2, .. }))
    }

    fn count_constant_nodes(&self) -> usize {
        dynamic_expressions::count_constant_nodes(&self.nodes)
    }

    fn count_scalar_constants(&self) -> usize {
        self.consts.len()
    }

    fn get_scalar_constants_flat(&self, out: &mut Vec<T>) {
        out.clear();
        out.extend_from_slice(&self.consts);
    }

    fn set_scalar_constants_flat(&mut self, values: &[T]) {
        self.consts.clone_from_slice(values);
    }

    fn loss_and_grad(
        &mut self,
        _plan: &Self::Plan,
        dataset: &Dataset<T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
        grad_ctx: &mut dynamic_expressions::GradContext<T, D>,
        eval_opts: &dynamic_expressions::EvalOptions,
        grad_out: &mut [f64],
    ) -> Option<f64>
    where
        T: FromPrimitive + ToPrimitive + core::ops::AddAssign,
    {
        use dynamic_expressions::utils::ZipEq;

        let n_params = self.consts.len();
        let n_rows = dataset.n_rows;
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

    fn mutate_constant(&mut self, rng: &mut Rng, temperature: f64, options: &Options<T, D>) -> bool {
        crate::mutation_functions::mutate_constant_in_place(rng, self, temperature, options)
    }

    fn randomize(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        dataset_n_features: usize,
        target_size: usize,
        _options: &Options<T, D>,
    ) -> Self {
        crate::mutation_functions::random_expr(rng, operators, dataset_n_features, target_size)
    }
}
