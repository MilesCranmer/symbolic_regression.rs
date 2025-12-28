use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use num_traits::Float;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::options::Options;

#[derive(Clone, Debug)]
pub(crate) struct StopController {
    max_evals: u64,
    timeout_in_seconds: f64,
    start_time: Instant,
    cancelled: Arc<AtomicBool>,
}

impl StopController {
    pub(crate) fn from_options<T: Float, const D: usize>(options: &Options<T, D>) -> Self {
        Self {
            max_evals: options.max_evals,
            timeout_in_seconds: options.timeout_in_seconds,
            start_time: Instant::now(),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            return true;
        }
        if self.timeout_in_seconds > 0.0 && self.start_time.elapsed().as_secs_f64() >= self.timeout_in_seconds {
            self.cancelled.store(true, Ordering::Release);
            return true;
        }
        false
    }

    pub(crate) fn should_stop(&self, total_evals: u64) -> bool {
        if self.is_cancelled() {
            return true;
        }
        if self.max_evals > 0 && total_evals >= self.max_evals {
            return true;
        }
        false
    }
}
