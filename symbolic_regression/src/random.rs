use std::ops::{Range, RangeInclusive};

pub(crate) fn usize_range(rng: &mut fastrand::Rng, range: Range<usize>) -> usize {
    rng.usize(range)
}

pub(crate) fn usize_range_excl(rng: &mut fastrand::Rng, range: Range<usize>, exclude: usize) -> usize {
    assert!(range.start < range.end);
    let len = range.end - range.start;
    assert!(len > 1);
    if exclude < range.start || exclude >= range.end {
        usize_range(rng, range)
    } else {
        let exclude_idx = exclude - range.start;
        let r = rng.usize(0..(len - 1));
        range.start + if r >= exclude_idx { r + 1 } else { r }
    }
}

pub(crate) fn usize_range_inclusive(rng: &mut fastrand::Rng, range: RangeInclusive<usize>) -> usize {
    let (start, end) = range.into_inner();
    if start >= end {
        return start;
    }
    start + rng.usize(0..=(end - start))
}

pub(crate) fn standard_normal(rng: &mut fastrand::Rng) -> f64 {
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub(crate) fn poisson_sample(rng: &mut fastrand::Rng, mean: f64) -> usize {
    if !mean.is_finite() || mean <= 0.0 {
        return 0;
    }

    // For small means, Knuth's algorithm is simple and fast enough.
    if mean < 30.0 {
        let limit = (-mean).exp();
        let mut k = 0usize;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= rng.f64();
            if p <= limit {
                return k - 1;
            }
        }
    }

    // For larger means, use PTRS (Poisson Transformed Rejection with Squeeze).
    // Reference: W. Hörmann (1993), "The Transformed Rejection Method for Generating Poisson Random Variables".
    let sqrt_mean = mean.sqrt();
    let b = 0.931 + 2.53 * sqrt_mean;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let v_r = 0.9277 - 3.6224 / (b - 2.0);

    loop {
        let u = rng.f64() - 0.5;
        let v = rng.f64();
        let us = 0.5 - u.abs();
        if us <= 0.0 {
            continue;
        }

        let k = ((2.0 * a / us + b) * u + mean + 0.43).floor() as i64;
        if k < 0 {
            continue;
        }
        let k_usize = k as usize;

        if us >= 0.07 && v <= v_r {
            return k_usize;
        }
        if us < 0.013 && v > us {
            continue;
        }

        let lhs = (v * inv_alpha / (a / (us * us) + b)).ln();
        let rhs = -mean + (k as f64) * mean.ln() - log_factorial(k_usize);
        if lhs <= rhs {
            return k_usize;
        }
    }
}

pub(crate) fn shuffle<T>(rng: &mut fastrand::Rng, values: &mut [T]) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.usize(0..=i);
        values.swap(i, j);
    }
}

pub(crate) fn choose<'a, T>(rng: &mut fastrand::Rng, values: &'a [T]) -> Option<&'a T> {
    if values.is_empty() {
        None
    } else {
        Some(&values[rng.usize(0..values.len())])
    }
}

fn log_factorial(k: usize) -> f64 {
    if k < 2 {
        return 0.0;
    }
    if k < 256 {
        return (2..=k).map(|i| (i as f64).ln()).sum();
    }

    // Stirling series for ln Γ(k+1) = ln(k!) with a couple correction terms.
    // Good enough for acceptance testing here.
    const LN_2PI: f64 = 1.837_877_066_409_345_3; // ln(2π)
    let x = (k as f64) + 1.0;
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    (x - 0.5) * x.ln() - x + 0.5 * LN_2PI + inv / 12.0 - inv2 * inv / 360.0 + inv2 * inv2 * inv / 1260.0
}
