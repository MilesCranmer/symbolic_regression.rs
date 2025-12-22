use crate::loss_functions::loss_to_cost;

#[test]
fn loss_to_cost_always_uses_floor_when_baseline_disabled() {
    let loss = 2.0f64;
    let complexity = 3usize;
    let parsimony = 0.1f64;

    let cost = loss_to_cost(loss, complexity, parsimony, false, Some(123.0));
    assert!((cost - (loss / 0.01 + parsimony * complexity as f64)).abs() < 1e-12);
}

#[test]
fn loss_to_cost_uses_floor_when_baseline_missing_or_too_small() {
    let loss = 2.0f64;
    let complexity = 3usize;
    let parsimony = 0.1f64;

    let cost_none = loss_to_cost(loss, complexity, parsimony, true, None);
    assert!((cost_none - (loss / 0.01 + parsimony * complexity as f64)).abs() < 1e-12);

    let cost_small = loss_to_cost(loss, complexity, parsimony, true, Some(0.001));
    assert!((cost_small - (loss / 0.01 + parsimony * complexity as f64)).abs() < 1e-12);
}

#[test]
fn loss_to_cost_uses_baseline_when_valid() {
    let loss = 2.0f64;
    let complexity = 3usize;
    let parsimony = 0.1f64;
    let baseline = 0.2f64;

    let cost = loss_to_cost(loss, complexity, parsimony, true, Some(baseline));
    assert!((cost - (loss / baseline + parsimony * complexity as f64)).abs() < 1e-12);
}
