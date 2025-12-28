use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolic_regression::prelude::*;

// Roughly mirrors SymbolicRegression.jl's template-expression + parameter usage,
// but using the Rust `TemplateExpression` API.

fn main() {
    const D: usize = 3;
    const N_FEATURES: usize = 3;
    let n_rows = 200;

    let mut rng = StdRng::seed_from_u64(0);

    let mut x = Array2::zeros((N_FEATURES, n_rows));
    let mut y = Array1::zeros(n_rows);

    // Ground truth:
    // y = sin(x1) + (a0) * x3^2, where a0 = 2.0
    for i in 0..n_rows {
        let x1 = rng.random_range(-2.0f32..2.0f32);
        let x2 = rng.random_range(-2.0f32..2.0f32);
        let x3 = rng.random_range(-2.0f32..2.0f32);

        x[(0, i)] = x1;
        x[(1, i)] = x2;
        x[(2, i)] = x3;

        y[i] = x1.sin() + 2.0 * (x3 * x3);
    }

    let dataset = Dataset::new(x, y);

    let operators = Operators::<D>::from_names_by_arity::<BuiltinOpsF32>(&["sin"], &["+", "*"], &[]).unwrap();

    let options = Options::<f32, D> {
        operators,
        maxsize: 20,
        niterations: 200,
        populations: 8,
        population_size: 64,
        should_optimize_constants: true,
        optimizer_probability: 0.1,
        ..Default::default()
    };

    // Template:
    // out = f(x1, x2) + a[0] * g(x3)
    //
    // `f` can only see (x1, x2), and `g` can only see (x3).
    // `a` is a parameter vector accessible from the combine function.
    let spec = TemplateSpec::<f32, BuiltinOpsF32, D>::new_with_combine_fixed_inputs::<N_FEATURES, _>(
        vec![("f", 2), ("g", 1)],
        vec![("a", 1)],
        |ctx, [x1, x2, x3]| {
            let f = ctx.call("f", &[x1, x2]);
            let g = ctx.call("g", &[x3]);

            let a0 = ctx.param("a").expect("param a missing")[0];
            let mut out = vec![0.0f32; ctx.n_rows()];
            for (dst, (&fv, &gv)) in out.iter_mut().zip(f.iter().zip(g.iter())) {
                *dst = fv + a0 * gv;
            }

            out
        },
    );

    let result = equation_search_with_spec::<_, BuiltinOpsF32, D, _>(&dataset, &options, spec);
    let dominating = result.hall_of_fame.pareto_front();

    println!("Final Pareto front:");
    println!("Complexity\tMSE\tEquation");
    for member in dominating {
        println!("{}\t{}\t{}", member.complexity, member.loss, member.expr);
    }
}
