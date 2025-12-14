use num_traits::Float;
use std::sync::Arc;

pub trait LossFn<T: Float>: Send + Sync {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T;
    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]);
}

pub type LossObject<T> = Arc<dyn LossFn<T> + Send + Sync>;

#[derive(Clone, Debug, Default)]
pub struct Mse;

impl<T: Float> LossFn<T> for Mse {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        assert_eq!(yhat.len(), y.len());
        match w {
            None => {
                let n = T::from(y.len()).unwrap();
                yhat.iter()
                    .copied()
                    .zip(y.iter().copied())
                    .map(|(a, b)| {
                        let r = a - b;
                        r * r
                    })
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
                    .copied()
                    .zip(y.iter().copied())
                    .zip(w.iter().copied())
                    .map(|((a, b), wi)| {
                        let r = a - b;
                        wi * r * r
                    })
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
                let inv = T::from(y.len()).unwrap().recip();
                let scale = T::from(2.0).unwrap() * inv;
                for ((o, a), b) in out.iter_mut().zip(yhat.iter()).zip(y.iter()) {
                    *o = scale * (*a - *b);
                }
            }
            Some(w) => {
                assert_eq!(w.len(), y.len());
                let sum_w = w.iter().copied().fold(T::zero(), |a, b| a + b);
                if sum_w == T::zero() {
                    out.fill(T::zero());
                    return;
                }
                let scale = T::from(2.0).unwrap() / sum_w;
                for (((o, a), b), wi) in out.iter_mut().zip(yhat.iter()).zip(y.iter()).zip(w.iter())
                {
                    *o = scale * (*wi) * (*a - *b);
                }
            }
        }
    }
}

pub fn mse<T: Float>() -> LossObject<T> {
    Arc::new(Mse)
}
