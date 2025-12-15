use super::linalg::{dot, inf_norm, matvec};
use super::line_search::{backtracking_linesearch, LineSearchInput};
use super::options::{BackTracking, EvalBudget, Objective, OptimOptions, OptimResult};

pub(crate) fn bfgs_minimize(
    x0: &[f64],
    obj: &mut impl Objective,
    opts: OptimOptions,
    ls: BackTracking,
) -> Option<OptimResult> {
    let n = x0.len();
    let mut budget = EvalBudget::default();

    let mut x = x0.to_vec();
    let mut x_prev = vec![0.0; n];
    let mut x_ls = vec![0.0; n];

    let mut g = vec![0.0; n];
    let mut g_prev = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut dx = vec![0.0; n];
    let mut dg = vec![0.0; n];

    let mut inv_h = vec![0.0; n * n];
    for i in 0..n {
        inv_h[i * n + i] = 1.0;
    }

    let mut fx = obj.fg(&x, &mut g, &mut budget)?;
    if !fx.is_finite() {
        return None;
    }

    for _ in 0..opts.iterations {
        if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
            break;
        }

        let g_inf = inf_norm(&g);
        if !g_inf.is_finite() || g_inf <= opts.g_abstol {
            break;
        }

        matvec(&mut s, &inv_h, &g);
        for v in &mut s {
            *v = -*v;
        }
        let mut dphi0 = dot(&g, &s);
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            inv_h.fill(0.0);
            for i in 0..n {
                inv_h[i * n + i] = 1.0;
            }
            for (dst, &gi) in s.iter_mut().zip(g.iter()) {
                *dst = -gi;
            }
            dphi0 = dot(&g, &s);
        }
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            break;
        }

        let phi0 = fx;
        let mut phi_at = |x_trial: &[f64]| -> Option<f64> {
            if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
                return None;
            }
            let v = obj.f_only(x_trial, &mut budget)?;
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        };

        let (_alpha, _) = backtracking_linesearch(
            &ls,
            LineSearchInput {
                x: &x,
                s: &s,
                alpha0: 1.0,
                phi0,
                dphi0,
            },
            &mut x_ls,
            &mut phi_at,
        )?;

        x_prev.copy_from_slice(&x);
        g_prev.copy_from_slice(&g);
        x.copy_from_slice(&x_ls);

        fx = obj.fg(&x, &mut g, &mut budget)?;
        if !fx.is_finite() {
            break;
        }

        // BFGS update (inverse Hessian):
        for ((dst, &xi), &xpi) in dx.iter_mut().zip(x.iter()).zip(x_prev.iter()) {
            *dst = xi - xpi;
        }
        for ((dst, &gcur), &gpi) in dg.iter_mut().zip(g.iter()).zip(g_prev.iter()) {
            *dst = gcur - gpi;
        }

        let dx_dg = dot(&dx, &dg);
        let curvature_eps = 1e-12;
        if !(dx_dg.is_finite() && dx_dg > curvature_eps) {
            inv_h.fill(0.0);
            for i in 0..n {
                inv_h[i * n + i] = 1.0;
            }
            continue;
        }
        if dx_dg > 0.0 {
            matvec(&mut u, &inv_h, &dg);
            let dg_u = dot(&dg, &u);
            if dg_u.is_finite() {
                let c1 = (dx_dg + dg_u) / (dx_dg * dx_dg);
                let c2 = 1.0 / dx_dg;
                for i in 0..n {
                    for j in 0..n {
                        inv_h[i * n + j] += c1 * dx[i] * dx[j] - c2 * (u[i] * dx[j] + dx[i] * u[j]);
                    }
                }
                // Keep symmetry; floating point drift can accumulate.
                for i in 0..n {
                    for j in (i + 1)..n {
                        let a = 0.5 * (inv_h[i * n + j] + inv_h[j * n + i]);
                        inv_h[i * n + j] = a;
                        inv_h[j * n + i] = a;
                    }
                }
            }
        }
    }

    Some(OptimResult {
        minimizer: x,
        minimum: fx,
        f_calls: budget.f_calls,
    })
}

pub(crate) fn newton_1d_minimize(
    x0: f64,
    obj: &mut impl Objective,
    opts: OptimOptions,
    ls: BackTracking,
) -> Option<OptimResult> {
    let mut budget = EvalBudget::default();
    let mut x = x0;
    let mut best_x = x0;
    let mut best_f = f64::INFINITY;
    let mut g_tmp = [0.0f64];

    for _ in 0..opts.iterations {
        if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
            break;
        }
        let x_vec = [x];
        let f = obj.fg(&x_vec, &mut g_tmp, &mut budget)?;
        let g = g_tmp[0];
        if !f.is_finite() || !g.is_finite() {
            return None;
        }
        if f < best_f {
            best_f = f;
            best_x = x;
        }
        if g.abs() <= opts.g_abstol {
            break;
        }

        // Finite-difference Hessian estimate:
        let eps = f64::EPSILON.sqrt() * (x.abs() + 1.0);
        let x_plus = [x + eps];
        let x_minus = [x - eps];
        let _ = obj.fg(&x_plus, &mut g_tmp, &mut budget)?;
        let g_plus = g_tmp[0];
        let _ = obj.fg(&x_minus, &mut g_tmp, &mut budget)?;
        let g_minus = g_tmp[0];
        let h = (g_plus - g_minus) / (2.0 * eps);
        let mut h_mod = h.abs().max(1e-12);
        if !h_mod.is_finite() {
            h_mod = 1e-12;
        }

        let mut s = -g / h_mod;
        let mut dphi0 = g * s;
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            s = -g;
            dphi0 = g * s;
        }
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            break;
        }

        let phi0 = f;
        let mut phi_at = |x_t: &[f64]| -> Option<f64> {
            if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
                return None;
            }
            let v = obj.f_only(x_t, &mut budget)?;
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        };

        let x_vec = [x];
        let s_vec = [s];
        let mut x_new = [x];
        let (alpha, _) = backtracking_linesearch(
            &ls,
            LineSearchInput {
                x: &x_vec,
                s: &s_vec,
                alpha0: 1.0,
                phi0,
                dphi0,
            },
            &mut x_new,
            &mut phi_at,
        )?;
        x += alpha * s;
    }

    Some(OptimResult {
        minimizer: vec![best_x],
        minimum: best_f,
        f_calls: budget.f_calls,
    })
}
