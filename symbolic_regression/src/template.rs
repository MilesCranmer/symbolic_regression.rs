use std::fmt;
use std::ops::AddAssign;
use std::sync::Arc;

pub use dynamic_expressions::template::{
    TemplateContext, TemplateExpression, TemplateStructure, string_template_pretty,
};
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::constant_optimization;
use crate::dataset::Dataset;
use crate::expression::{ConstantOptimizable, ExpressionSpec};
use crate::operators::Operators;
use crate::options::Options;
use crate::pop_member::Evaluator;

pub struct TemplateSpec<T, Ops, const D: usize> {
    pub structure: Arc<TemplateStructure<T, Ops, D>>,
}

impl<T, Ops, const D: usize> Clone for TemplateSpec<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            structure: self.structure.clone(),
        }
    }
}

impl<T, Ops, const D: usize> fmt::Debug for TemplateSpec<T, Ops, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemplateSpec")
            .field("structure", &self.structure)
            .finish()
    }
}

impl<T, Ops, const D: usize> TemplateSpec<T, Ops, D> {
    pub fn new(structure: Arc<TemplateStructure<T, Ops, D>>) -> Self {
        Self { structure }
    }

    pub fn new_with_combine_fixed_inputs<const N: usize, F>(
        functions: Vec<(&str, usize)>,
        params: Vec<(&str, usize)>,
        combine: F,
    ) -> Self
    where
        F: for<'a> Fn(&mut dyn TemplateContext<T, Ops, D>, [&'a [T]; N]) -> Vec<T> + Send + Sync + 'static,
    {
        let structure = TemplateStructure::new_fixed_inputs::<N, F>(functions, params, combine);
        Self::new(Arc::new(structure))
    }
}

impl<T, Ops, const D: usize> ExpressionSpec<T, Ops, D> for TemplateSpec<T, Ops, D>
where
    T: Float + Clone + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    type Expr = TemplateExpression<T, Ops, D>;

    fn random_expr(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        _dataset_n_features: usize,
        target_size: usize,
        _options: &Options<T, D>,
    ) -> Self::Expr {
        let k = self.structure.n_functions().max(1);
        let total = target_size.max(k);
        let mut sizes = vec![1usize; k];
        for _ in 0..(total - k) {
            sizes[rng.usize(0..k)] += 1;
        }

        let mut trees = Vec::with_capacity(k);
        for (i, &arity) in self.structure.function_arity.iter().enumerate() {
            let sz = sizes.get(i).copied().unwrap_or(1);
            trees.push(crate::mutation_functions::random_expr(rng, operators, arity, sz));
        }

        let mut params = Vec::with_capacity(self.structure.n_params());
        for &len in &self.structure.param_len {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let v = crate::random::standard_normal(rng);
                values.push(T::from(v).unwrap_or_else(T::zero));
            }
            params.push(values);
        }

        TemplateExpression {
            structure: self.structure.clone(),
            trees,
            params,
        }
    }
}

impl<T, Ops, const D: usize> ConstantOptimizable<T, Ops, D> for TemplateExpression<T, Ops, D>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign + Clone + Send + Sync,
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
        constant_optimization::finite_diff_loss_and_grad::<T, Ops, D, Self>(
            self, plans, dataset, options, evaluator, grad_ctx, eval_opts, grad_out,
        )
    }
}
