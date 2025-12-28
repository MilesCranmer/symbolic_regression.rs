use dynamic_expressions::EvalOptions;
use ndarray::Array2;
use num_traits::Float;

use crate::dataset::TaggedDataset;
use crate::expression::{ExprExt, Expression};
use crate::loss_functions::loss_to_cost;
use crate::options::Options;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MemberId(pub u64);

pub struct PopMember<T: Float, Ops, const D: usize, E = dynamic_expressions::PostfixExpr<T, Ops, D>>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: Expression<T, Ops, D>,
{
    pub id: MemberId,
    pub parent: Option<MemberId>,
    pub birth: u64,
    pub expr: E,
    pub plans: Vec<dynamic_expressions::EvalPlan<D>>,
    pub complexity: usize,
    pub loss: T,
    pub cost: T,
    pub _ops: core::marker::PhantomData<Ops>,
}

impl<T: Float, Ops, const D: usize, E> Clone for PopMember<T, Ops, D, E>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: Expression<T, Ops, D>,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            parent: self.parent,
            birth: self.birth,
            expr: self.expr.clone(),
            plans: self.plans.clone(),
            complexity: self.complexity,
            loss: self.loss,
            cost: self.cost,
            _ops: core::marker::PhantomData,
        }
    }
}

pub struct Evaluator<T: Float, const D: usize> {
    pub eval_opts: EvalOptions,
    pub yhat: Vec<T>,
    pub scratch: Array2<T>,
}

impl<T: Float, const D: usize> Evaluator<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            eval_opts: EvalOptions {
                check_finite: true,
                early_exit: true,
            },
            yhat: vec![T::zero(); n_rows],
            scratch: Array2::zeros((0, 0)),
        }
    }

    pub fn ensure_n_rows(&mut self, n_rows: usize) {
        if self.yhat.len() != n_rows {
            self.yhat.resize(n_rows, T::zero());
        }
    }
}

impl<T: Float, Ops, const D: usize, E> PopMember<T, Ops, D, E>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: Expression<T, Ops, D>,
{
    pub fn from_expr(id: MemberId, parent: Option<MemberId>, birth: u64, expr: E, n_features: usize) -> Self {
        let plans = expr.build_plans(n_features);
        Self {
            id,
            parent,
            birth,
            expr,
            plans,
            complexity: 0,
            loss: T::infinity(),
            cost: T::infinity(),
            _ops: core::marker::PhantomData,
        }
    }

    pub fn rebuild_plan(&mut self, n_features: usize) {
        self.plans = self.expr.build_plans(n_features);
    }

    pub fn evaluate(
        &mut self,
        dataset: &TaggedDataset<'_, T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
    ) -> bool {
        let eval_opts = evaluator.eval_opts;
        evaluator.ensure_n_rows(dataset.n_rows);
        let ok = self.expr.eval_with_plans(
            &self.plans,
            dataset.x.view(),
            &mut evaluator.yhat,
            &mut evaluator.scratch,
            &eval_opts,
        );

        self.complexity = self.expr.complexity(options);

        if !ok {
            self.loss = T::infinity();
            self.cost = T::infinity();
            return false;
        }

        let loss = options.loss.loss(
            &evaluator.yhat,
            dataset.y.as_slice().unwrap(),
            dataset.weights.as_ref().and_then(|w| w.as_slice()),
        );
        if !loss.is_finite() {
            self.loss = T::infinity();
            self.cost = T::infinity();
            return false;
        }
        self.loss = loss;

        self.cost = loss_to_cost(
            loss,
            self.complexity,
            options.parsimony,
            options.use_baseline,
            dataset.baseline_loss,
        );
        true
    }
}
