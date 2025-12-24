use std::sync::Arc;

use dynamic_expressions::EvalOptions;
use dynamic_expressions::operator_enum::scalar;
use dynamic_expressions::utils::ZipEq;
use ndarray::Array2;
use num_traits::Float;

use crate::dataset::Dataset;

pub trait LossFn<T: Float>: Send + Sync {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T;
    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]);
}

pub fn baseline_loss_from_zero_expression<T: Float, Ops, const D: usize>(
    dataset: &Dataset<T>,
    loss: &dyn LossFn<T>,
) -> Option<T>
where
    Ops: scalar::ScalarOpSet<T>,
{
    let expr: dynamic_expressions::expression::PostfixExpr<T, Ops, D> =
        dynamic_expressions::expression::PostfixExpr::zero();
    let plan = dynamic_expressions::compile_plan(&expr.nodes, dataset.n_features, expr.consts.len());

    let mut yhat = vec![T::zero(); dataset.n_rows];
    let mut scratch = Array2::<T>::zeros((0, 0));

    let eval_opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let ok =
        dynamic_expressions::eval_plan_array_into(&mut yhat, &plan, &expr, dataset.x.view(), &mut scratch, &eval_opts);
    if !ok {
        return None;
    }

    let base = loss.loss(&yhat, dataset.y_slice(), dataset.weights_slice());
    base.is_finite().then_some(base)
}

pub fn loss_to_cost<T: Float>(
    loss: T,
    complexity: usize,
    parsimony: f64,
    use_baseline: bool,
    baseline_loss: Option<T>,
) -> T {
    let floor = T::from(0.01).unwrap();

    let denom = if use_baseline {
        match baseline_loss {
            Some(b) if b.is_finite() && b >= floor => b,
            _ => floor,
        }
    } else {
        floor
    };

    let parsimony = T::from(parsimony).unwrap_or_else(T::zero);
    loss / denom + parsimony * T::from(complexity).unwrap_or_else(T::zero)
}

pub type LossObject<T> = Arc<dyn LossFn<T> + Send + Sync>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LossKind {
    Mse,
    Mae,
    Rmse,
    Huber { delta: f64 },
    LogCosh,
    Lp { p: f64 },
    Quantile { tau: f64 },
    EpsilonInsensitive { eps: f64 },
}

impl LossKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mse" => Some(Self::Mse),
            "mae" => Some(Self::Mae),
            "rmse" => Some(Self::Rmse),
            "huber" => Some(Self::Huber { delta: 1.0 }),
            "logcosh" => Some(Self::LogCosh),
            "lp" => Some(Self::Lp { p: 2.0 }),
            "quantile" => Some(Self::Quantile { tau: 0.5 }),
            "epsilon-insensitive" | "epsilon_insensitive" | "eps-insensitive" | "eps_insensitive" => {
                Some(Self::EpsilonInsensitive { eps: 0.1 })
            }
            _ => None,
        }
    }
}

pub fn make_loss<T: Float + Send + Sync + 'static>(kind: LossKind) -> LossObject<T> {
    match kind {
        LossKind::Mse => mse::<T>(),
        LossKind::Mae => mae::<T>(),
        LossKind::Rmse => rmse::<T>(),
        LossKind::Huber { delta } => huber::<T>(delta),
        LossKind::LogCosh => log_cosh::<T>(),
        LossKind::Lp { p } => lp::<T>(p),
        LossKind::Quantile { tau } => quantile::<T>(tau),
        LossKind::EpsilonInsensitive { eps } => epsilon_insensitive::<T>(eps),
    }
}

pub trait PointwiseLoss<T: Float> {
    fn point_loss(&self, yhat: T, y: T) -> T;
    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T;
}

#[derive(Clone, Debug)]
pub struct MeanLoss<L>(pub L);

impl<T: Float, L: PointwiseLoss<T> + Send + Sync> LossFn<T> for MeanLoss<L> {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        assert_eq!(yhat.len(), y.len());
        match w {
            None => {
                if y.is_empty() {
                    return T::zero();
                }
                let n = T::from(y.len()).unwrap();
                yhat.iter()
                    .zip_eq(y)
                    .map(|(&a, &b)| self.0.point_loss(a, b))
                    .fold(T::zero(), |acc, v| acc + v)
                    / n
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    return T::zero();
                }
                yhat.iter()
                    .zip_eq(y)
                    .zip_eq(w)
                    .map(|((&a, &b), &wi)| wi * self.0.point_loss(a, b))
                    .fold(T::zero(), |acc, v| acc + v)
                    / sum_w
            }
        }
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        assert_eq!(yhat.len(), y.len());
        assert_eq!(out.len(), y.len());
        match w {
            None => {
                if y.is_empty() {
                    out.fill(T::zero());
                    return;
                }
                let inv = T::from(y.len()).unwrap().recip();
                for ((o, &a), &b) in out.iter_mut().zip_eq(yhat).zip_eq(y) {
                    *o = inv * self.0.point_dloss_dyhat(a, b);
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let inv = sum_w.recip();
                for (((o, &a), &b), &wi) in out.iter_mut().zip_eq(yhat).zip_eq(y).zip_eq(w) {
                    *o = wi * inv * self.0.point_dloss_dyhat(a, b);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SquaredLoss;

impl<T: Float> PointwiseLoss<T> for SquaredLoss {
    fn point_loss(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        r * r
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        T::from(2.0).unwrap() * (yhat - y)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AbsLoss;

impl<T: Float> PointwiseLoss<T> for AbsLoss {
    fn point_loss(&self, yhat: T, y: T) -> T {
        (yhat - y).abs()
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        if r > T::zero() {
            T::one()
        } else if r < T::zero() {
            -T::one()
        } else {
            T::zero()
        }
    }
}

#[derive(Clone, Debug)]
pub struct HuberLoss<T: Float> {
    pub delta: T,
}

impl<T: Float> PointwiseLoss<T> for HuberLoss<T> {
    fn point_loss(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        let ar = r.abs();
        let half = T::from(0.5).unwrap();
        if ar <= self.delta {
            half * r * r
        } else {
            self.delta * (ar - half * self.delta)
        }
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        let ar = r.abs();
        if ar <= self.delta {
            r
        } else if r > T::zero() {
            self.delta
        } else if r < T::zero() {
            -self.delta
        } else {
            T::zero()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LogCoshLoss;

impl<T: Float> PointwiseLoss<T> for LogCoshLoss {
    fn point_loss(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        r.cosh().ln()
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        r.tanh()
    }
}

#[derive(Clone, Debug)]
pub struct LpLoss<T: Float> {
    pub p: T,
}

impl<T: Float> PointwiseLoss<T> for LpLoss<T> {
    fn point_loss(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        r.abs().powf(self.p)
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        if r == T::zero() {
            return T::zero();
        }
        let p = self.p;
        let ar = r.abs();
        let s = if r > T::zero() { T::one() } else { -T::one() };
        p * ar.powf(p - T::one()) * s
    }
}

#[derive(Clone, Debug)]
pub struct QuantileLoss<T: Float> {
    pub tau: T,
}

impl<T: Float> PointwiseLoss<T> for QuantileLoss<T> {
    fn point_loss(&self, yhat: T, y: T) -> T {
        let u = y - yhat;
        if u >= T::zero() {
            self.tau * u
        } else {
            (self.tau - T::one()) * u
        }
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        let u = y - yhat;
        if u > T::zero() {
            -self.tau
        } else if u < T::zero() {
            T::one() - self.tau
        } else {
            T::zero()
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpsilonInsensitiveLoss<T: Float> {
    pub eps: T,
}

impl<T: Float> PointwiseLoss<T> for EpsilonInsensitiveLoss<T> {
    fn point_loss(&self, yhat: T, y: T) -> T {
        let r = (yhat - y).abs() - self.eps;
        if r > T::zero() { r } else { T::zero() }
    }

    fn point_dloss_dyhat(&self, yhat: T, y: T) -> T {
        let r = yhat - y;
        let ar = r.abs();
        if ar > self.eps {
            if r > T::zero() {
                T::one()
            } else if r < T::zero() {
                -T::one()
            } else {
                T::zero()
            }
        } else {
            T::zero()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Rmse;

impl<T: Float> LossFn<T> for Rmse {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        mse::<T>().loss(yhat, y, w).sqrt()
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        assert_eq!(yhat.len(), y.len());
        assert_eq!(out.len(), y.len());

        let m = mse::<T>().loss(yhat, y, w);
        let rmse = m.sqrt();
        if rmse == T::zero() || !rmse.is_finite() {
            out.fill(T::zero());
            return;
        }

        match w {
            None => {
                let inv = T::from(y.len()).unwrap().recip();
                let scale = inv / rmse;
                for ((o, &a), &b) in out.iter_mut().zip_eq(yhat).zip_eq(y) {
                    *o = scale * (a - b);
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let scale = sum_w.recip() / rmse;
                for (((o, &a), &b), &wi) in out.iter_mut().zip_eq(yhat).zip_eq(y).zip_eq(w) {
                    *o = scale * wi * (a - b);
                }
            }
        }
    }
}

pub fn mse<T: Float>() -> LossObject<T> {
    Arc::new(MeanLoss(SquaredLoss))
}

pub fn mae<T: Float>() -> LossObject<T> {
    Arc::new(MeanLoss(AbsLoss))
}

pub fn rmse<T: Float>() -> LossObject<T> {
    Arc::new(Rmse)
}

pub fn huber<T: Float + Send + Sync + 'static>(delta: f64) -> LossObject<T> {
    Arc::new(MeanLoss(HuberLoss {
        delta: T::from(delta).unwrap_or_else(T::one),
    }))
}

pub fn log_cosh<T: Float>() -> LossObject<T> {
    Arc::new(MeanLoss(LogCoshLoss))
}

pub fn lp<T: Float + Send + Sync + 'static>(p: f64) -> LossObject<T> {
    Arc::new(MeanLoss(LpLoss {
        p: T::from(p).unwrap_or_else(|| T::from(2.0).unwrap()),
    }))
}

pub fn quantile<T: Float + Send + Sync + 'static>(tau: f64) -> LossObject<T> {
    Arc::new(MeanLoss(QuantileLoss {
        tau: T::from(tau).unwrap_or_else(|| T::from(0.5).unwrap()),
    }))
}

pub fn epsilon_insensitive<T: Float + Send + Sync + 'static>(eps: f64) -> LossObject<T> {
    Arc::new(MeanLoss(EpsilonInsensitiveLoss {
        eps: T::from(eps).unwrap_or_else(|| T::from(0.1).unwrap()),
    }))
}
