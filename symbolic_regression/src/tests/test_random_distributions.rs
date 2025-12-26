use fastrand::Rng;

#[test]
fn standard_normal_has_reasonable_moments() {
    let mut rng = Rng::with_seed(0);
    let n = 200_000usize;

    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for _ in 0..n {
        let x = crate::random::standard_normal(&mut rng);
        sum += x;
        sumsq += x * x;
    }

    let mean = sum / (n as f64);
    let var = (sumsq / (n as f64)) - mean * mean;

    assert!(mean.abs() < 0.01, "mean={mean}");
    assert!((0.98..=1.02).contains(&var), "var={var}");
}

#[test]
fn poisson_sample_has_reasonable_moments() {
    let mut rng = Rng::with_seed(1);
    let cases: &[(f64, usize)] = &[
        (0.1, 200_000),
        (1.0, 200_000),
        (10.0, 200_000),
        (30.0, 200_000),
        (100.0, 100_000),
        (1_000.0, 50_000),
    ];

    for &(lambda, n) in cases {
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for _ in 0..n {
            let k = crate::random::poisson_sample(&mut rng, lambda) as f64;
            sum += k;
            sumsq += k * k;
        }

        let mean = sum / (n as f64);
        let var = (sumsq / (n as f64)) - mean * mean;

        // Mean std error is sqrt(lambda / n). Allow a wide bound to avoid flaky tests,
        // plus a tiny absolute floor for very small lambda.
        let mean_tol = 8.0 * (lambda / (n as f64)).sqrt() + 0.02;
        let var_tol = 0.15 * lambda.max(1.0); // loose relative tolerance

        assert!(
            (mean - lambda).abs() <= mean_tol,
            "lambda={lambda} mean={mean} tol={mean_tol}"
        );
        assert!(
            (var - lambda).abs() <= var_tol,
            "lambda={lambda} var={var} tol={var_tol}"
        );
    }
}
