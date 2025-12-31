//! Minimal "raw evaluator" benchmark for the GPU path.
//!
//! This isolates the *evaluation* kernel + readback path (batched) from the rest of the
//! genetic algorithm. It's useful to answer: "is the shader itself slow, or is the search
//! machinery / batching / optimizer driving us into tiny batches?"
//!
//! Run (Metal/Vulkan):
//!   cargo run -p symbolic_regression --example gpu_eval_bench --features wgpu --release
//!
//! You can tune sizes with env vars:
//!   N_ROWS=300 N_FEAT=5 P=8192 ITERS=200 BATCH_MAX=16384 \
//!     cargo run -p symbolic_regression --example gpu_eval_bench --features wgpu --release

#[cfg(wgpu)]
fn main() {
    use std::time::Instant;

    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::expression::{Metadata, PostfixExpr};
    use dynamic_expressions::node::PNode;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Sizes (defaults match a typical SR workload).
    let n_rows: usize = std::env::var("N_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let n_features: usize = std::env::var("N_FEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let p: usize = std::env::var("P").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let batch_max: usize = std::env::var("BATCH_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384);

    eprintln!("gpu_eval_bench: n_rows={n_rows} n_features={n_features} p={p} iters={iters} batch_max={batch_max}");

    // Deterministic random dataset.
    let mut rng = StdRng::seed_from_u64(0);
    let mut x = Array2::<f32>::zeros((n_features, n_rows));
    let mut y = Array1::<f32>::zeros(n_rows);
    for row in 0..n_rows {
        let mut s = 0.0f32;
        for f in 0..n_features {
            let v = rng.random_range(-1.0f32..1.0f32);
            x[(f, row)] = v;
            s += v;
        }
        y[row] = s;
    }
    let dataset = symbolic_regression::Dataset::new(x, y);

    // GPU evaluator.
    let gpu = match symbolic_regression::GpuClient::spawn(&dataset, batch_max) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("gpu_eval_bench: GPU init failed: {e:?}");
            return;
        }
    };

    // Build a single moderately "typical" expression and duplicate it P times.
    //
    // Expr: ((x0 * x1) + (x2 * x3)) + x4
    let add = BuiltinOpsF32::lookup("+").expect("builtin has +");
    let mul = BuiltinOpsF32::lookup("*").expect("builtin has *");
    let expr = PostfixExpr::<f32, BuiltinOpsF32, 8>::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Var { feature: 1 },
            PNode::Op {
                arity: mul.arity,
                op: mul.id,
            },
            PNode::Var { feature: 2 },
            PNode::Var { feature: 3 },
            PNode::Op {
                arity: mul.arity,
                op: mul.id,
            },
            PNode::Op {
                arity: add.arity,
                op: add.id,
            },
            PNode::Var { feature: 4 },
            PNode::Op {
                arity: add.arity,
                op: add.id,
            },
        ],
        Vec::new(),
        Metadata::default(),
    );

    let packed_one = symbolic_regression::pack_expr(&expr).expect("expr should be gpu-packable");
    let programs: Vec<_> = std::iter::repeat_n(packed_one, p).collect();

    // Warmup (Metal drivers can have one-time costs).
    {
        let warm_p = programs.len().min(1024);
        let mut warm_loss = vec![0.0f32; warm_p];
        gpu.eval_mse_many(&programs[..warm_p], &mut warm_loss);
    }

    // Timed run.
    let mut losses = vec![0.0f32; p];
    let t0 = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..iters {
        gpu.eval_mse_many(&programs, &mut losses);
        checksum += losses[0];
    }
    let dt = t0.elapsed().as_secs_f64();
    let total_evals = (iters as f64) * (p as f64);
    let eval_per_s = total_evals / dt;

    println!(
        "gpu_eval_bench: total_evals={total_evals} time_s={dt:.6} eval_per_s={eval_per_s:.3e} checksum={checksum}"
    );
}

#[cfg(not(wgpu))]
fn main() {
    eprintln!("Run with `--features wgpu` on a native (non-wasm32) target.");
}
