use crate::dataset::Dataset;
use crate::options::Options;
use dynamic_expressions::compile_plan;
use dynamic_expressions::eval_plan_array_into;
use dynamic_expressions::expr::PostfixExpr;
use dynamic_expressions::operators::scalar::ScalarOpSet;
use dynamic_expressions::EvalOptions;
use ndarray::ArrayView2;
use num_traits::Float;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MemberId(pub u64);

#[derive(Debug)]
pub struct PopMember<T: Float, Ops, const D: usize> {
    pub id: MemberId,
    pub parent: Option<MemberId>,
    pub birth: u64,
    pub expr: PostfixExpr<T, Ops, D>,
    pub plan: dynamic_expressions::EvalPlan<D>,
    pub complexity: usize,
    pub loss: T,
    pub cost: T,
}

impl<T: Float, Ops, const D: usize> Clone for PopMember<T, Ops, D>
where
    T: Clone,
{
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
    pub scratch: Vec<Vec<T>>,
}

impl<T: Float, const D: usize> Evaluator<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            eval_opts: EvalOptions {
                check_finite: true,
                early_exit: true,
            },
            yhat: vec![T::zero(); n_rows],
            scratch: Vec::new(),
        }
    }
}

fn normalization<T: Float>(dataset: &Dataset<T>) -> T {
    let floor = T::from(0.01).unwrap();
    if dataset.baseline_loss > floor {
        dataset.baseline_loss
    } else {
        floor
    }
}

impl<T: Float, Ops, const D: usize> PopMember<T, Ops, D>
where
    Ops: ScalarOpSet<T>,
{
    pub fn from_expr(
        id: MemberId,
        parent: Option<MemberId>,
        birth: u64,
        expr: PostfixExpr<T, Ops, D>,
        n_features: usize,
    ) -> Self {
        let plan = compile_plan::<D>(&expr.nodes, n_features, expr.consts.len());
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
        self.plan = compile_plan::<D>(&self.expr.nodes, n_features, self.expr.consts.len());
    }

    pub fn evaluate(
        &mut self,
        dataset: &Dataset<T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
    ) -> bool {
        let x: ArrayView2<'_, T> = dataset.x.view();
        let ok = eval_plan_array_into::<T, Ops, D>(
            &mut evaluator.yhat,
            &self.plan,
            &self.expr,
            x,
            &mut evaluator.scratch,
            &evaluator.eval_opts,
        );

        self.complexity = self.expr.nodes.len();

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

        let mut cost = loss;
        if options.use_baseline {
            cost = cost / normalization(dataset);
        }
        let parsimony = T::from(options.parsimony).unwrap_or_else(T::zero);
        cost = cost + parsimony * T::from(self.complexity).unwrap();
        self.cost = cost;
        true
    }
}
