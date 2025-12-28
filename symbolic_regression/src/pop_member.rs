use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::{EvalOptions, EvalPlan};
use num_traits::Float;
#[cfg(target_arch = "wasm32")]
use web_time::{SystemTime, UNIX_EPOCH};

use crate::complexity::compute_complexity;
use crate::dataset::TaggedDataset;
use crate::loss_functions::loss_to_cost;
use crate::options::Options;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MemberId(pub u64);

#[derive(Debug)]
pub struct PopMember<T: Float, Ops, const D: usize> {
    pub id: MemberId,
    pub parent: Option<MemberId>,
    pub birth: u64,
    pub expr: PostfixExpr<T, Ops, D>,
    pub plan: EvalPlan<D>,
    pub complexity: usize,
    pub loss: T,
    pub cost: T,
}

static PSEUDO_TIME: OnceLock<AtomicU64> = OnceLock::new();

fn pseudo_time() -> &'static AtomicU64 {
    PSEUDO_TIME.get_or_init(|| AtomicU64::new(0))
}

pub(crate) fn get_birth_order(deterministic: bool) -> u64 {
    if deterministic {
        // SymbolicRegression.jl: `pseudo_time[] += 1; return pseudo_time[]`
        return pseudo_time().fetch_add(1, Ordering::Relaxed).saturating_add(1);
    }

    // SymbolicRegression.jl: `round(Int, 1e7 * time())`
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX_EPOCH");
    let secs = dur.as_secs();
    let nanos = dur.subsec_nanos() as u64;
    // Round to the nearest 100ns tick (for positive values, ties round up).
    let ticks_1e7 = nanos.saturating_add(50) / 100;
    secs.saturating_mul(10_000_000).saturating_add(ticks_1e7)
}

#[cfg(test)]
pub(crate) fn reset_pseudo_time_for_tests() {
    pseudo_time().store(0, Ordering::Relaxed);
}

impl<T: Float, Ops, const D: usize> Clone for PopMember<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            parent: self.parent,
            birth: self.birth,
            expr: self.expr.clone(),
            plan: self.plan.clone(),
            complexity: self.complexity,
            loss: self.loss,
            cost: self.cost,
        }
    }
}

pub struct Evaluator<T: Float, const D: usize> {
    pub eval_opts: EvalOptions,
    pub yhat: Vec<T>,
    pub scratch: ndarray::Array2<T>,
}

impl<T: Float, const D: usize> Evaluator<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            eval_opts: EvalOptions {
                check_finite: true,
                early_exit: true,
            },
            yhat: vec![T::zero(); n_rows],
            scratch: ndarray::Array2::zeros((0, 0)),
        }
    }

    pub fn ensure_n_rows(&mut self, n_rows: usize) {
        if self.yhat.len() != n_rows {
            self.yhat.resize(n_rows, T::zero());
        }
    }
}

impl<T: Float, Ops, const D: usize> PopMember<T, Ops, D>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    pub fn from_expr(
        id: MemberId,
        parent: Option<MemberId>,
        expr: PostfixExpr<T, Ops, D>,
        n_features: usize,
        options: &Options<T, D>,
    ) -> Self {
        let plan = dynamic_expressions::compile_plan(&expr.nodes, n_features, expr.consts.len());
        Self {
            id,
            parent,
            birth: get_birth_order(options.deterministic),
            expr,
            plan,
            complexity: 0,
            loss: T::infinity(),
            cost: T::infinity(),
        }
    }

    pub fn from_expr_with_birth(
        id: MemberId,
        parent: Option<MemberId>,
        birth: u64,
        expr: PostfixExpr<T, Ops, D>,
        n_features: usize,
    ) -> Self {
        let plan = dynamic_expressions::compile_plan(&expr.nodes, n_features, expr.consts.len());
        Self {
            id,
            parent,
            birth,
            expr,
            plan,
            complexity: 0,
            loss: T::infinity(),
            cost: T::infinity(),
        }
    }

    pub fn rebuild_plan(&mut self, n_features: usize) {
        self.plan = dynamic_expressions::compile_plan(&self.expr.nodes, n_features, self.expr.consts.len());
    }

    pub fn evaluate(
        &mut self,
        dataset: &TaggedDataset<'_, T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
    ) -> bool {
        evaluator.ensure_n_rows(dataset.n_rows);
        let ok = dynamic_expressions::eval_plan_array_into(
            &mut evaluator.yhat,
            &self.plan,
            &self.expr,
            dataset.x.view(),
            &mut evaluator.scratch,
            &evaluator.eval_opts,
        );

        self.complexity = compute_complexity(&self.expr.nodes, options);

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
        if loss.is_nan() {
            self.loss = loss;
            self.cost = T::nan();
            return false;
        }
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
