use crate::loss_functions;

fn assert_close(a: f64, b: f64, tol: f64) {
    let d = (a - b).abs();
    assert!(d <= tol, "expected |{a} - {b}| <= {tol} (diff={d})");
}

#[test]
fn mse_mae_rmse_match_known_values_unweighted() {
    let y = [1.0_f64, 2.0, 3.0];
    let yhat = [2.0_f64, 0.0, 4.0];

    let l_mse = loss_functions::mse::<f64>().loss(&yhat, &y, None);
    let l_mae = loss_functions::mae::<f64>().loss(&yhat, &y, None);
    let l_rmse = loss_functions::rmse::<f64>().loss(&yhat, &y, None);

    // residuals: [1, -2, 1]
    // mse = (1 + 4 + 1) / 3 = 2
    // mae = (1 + 2 + 1) / 3 = 4/3
    // rmse = sqrt(2)
    assert_close(l_mse, 2.0, 1e-12);
    assert_close(l_mae, 4.0 / 3.0, 1e-12);
    assert_close(l_rmse, 2.0_f64.sqrt(), 1e-12);
}

#[test]
fn mse_mae_weighted_respects_sum_w() {
    let y = [1.0_f64, 2.0, 3.0];
    let yhat = [2.0_f64, 0.0, 4.0];
    let w = [1.0_f64, 3.0, 1.0];

    let l_mse = loss_functions::mse::<f64>().loss(&yhat, &y, Some(&w));
    let l_mae = loss_functions::mae::<f64>().loss(&yhat, &y, Some(&w));

    // residuals: [1, -2, 1], abs: [1, 2, 1], sq: [1, 4, 1]
    // sum_w = 5
    // weighted mse = (1*1 + 3*4 + 1*1) / 5 = 14/5
    // weighted mae = (1*1 + 3*2 + 1*1) / 5 = 8/5
    assert_close(l_mse, 14.0 / 5.0, 1e-12);
    assert_close(l_mae, 8.0 / 5.0, 1e-12);
}

#[test]
fn huber_behaves_quadratic_then_linear() {
    let y = [0.0_f64, 0.0];
    let yhat = [0.5_f64, 2.0];
    let delta = 1.0;

    // r=[0.5,2.0]; huber = mean([0.5*r^2, delta*(|r|-0.5*delta)]) = mean([0.125, 1.5])
    let l = loss_functions::huber::<f64>(delta).loss(&yhat, &y, None);
    assert_close(l, (0.125 + 1.5) / 2.0, 1e-12);
}

#[test]
fn make_loss_kind_dispatches() {
    let y = [1.0_f64, 2.0, 3.0];
    let yhat = [2.0_f64, 0.0, 4.0];
    let l = loss_functions::make_loss::<f64>(loss_functions::LossKind::Mae).loss(&yhat, &y, None);
    assert_close(l, 4.0 / 3.0, 1e-12);
}
