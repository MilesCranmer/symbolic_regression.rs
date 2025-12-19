mod common;

use approx::assert_relative_eq;
use common::{make_x, TestOps};
use dynamic_expressions::math::{cos, div, exp, fma, log, neg, sin};
use dynamic_expressions::operator_enum::scalar::OpId;
use dynamic_expressions::strings::{string_tree, OpNames, StringTreeOptions};
use dynamic_expressions::{eval_tree_array, lit, EvalOptions};

#[test]
fn algebra_overloads_and_string_paths_are_exercised() {
    let (x_data, x) = make_x(2, 16);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut a = common::var(0);
    a.meta.variable_names = vec!["a".into(), "b".into()];
    let b = common::var(1);

    let exprs = [
        a.clone() + b.clone(),
        a.clone() - b.clone(),
        a.clone() * b.clone(),
        a.clone() / b.clone(),
        -a.clone(),
        a.clone() + 1.0,
        a.clone() - 1.0,
        a.clone() * 2.0,
        a.clone() / 2.0,
        lit(2.0) + b.clone(),
        lit(2.0) - b.clone(),
        lit(2.0) * b.clone(),
        lit(2.0) / b.clone(),
        div(a.clone(), b.clone()),
        neg(a.clone()),
        cos(a.clone()),
        sin(a.clone()),
        exp(a.clone()),
        log(a.clone()),
        fma(a.clone(), b.clone(), common::c(0.7)),
    ];

    assert_eq!(exprs[0].meta.variable_names, vec!["a", "b"]);

    for ex in exprs {
        let (_y, ok) = eval_tree_array::<f64, TestOps, 3>(&ex, x_view, &opts);
        assert!(ok);
    }

    let s = string_tree(&(-(a.clone() + b.clone())), StringTreeOptions::default());
    assert_eq!(s, "-(a + b)");

    let s = string_tree(&cos(a.clone() + b.clone()), StringTreeOptions::default());
    assert_eq!(s, "cos(a + b)");

    let s = string_tree(
        &fma(a.clone(), b.clone(), common::c(0.7)),
        StringTreeOptions::default(),
    );
    assert_eq!(s, "fma(a, b, 0.7)");

    let s_pretty = string_tree(
        &fma(a.clone(), b.clone(), common::c(0.7)),
        StringTreeOptions {
            pretty: true,
            ..StringTreeOptions::default()
        },
    );
    assert_eq!(s_pretty, "fma(a, b, 0.7)");

    let display = format!("{}", a.clone() + b.clone());
    assert_eq!(display, "a + b");

    assert_eq!(
        <TestOps as OpNames>::op_name(OpId { arity: 9, id: 0 }),
        "unknown_op"
    );
    assert_eq!(
        <TestOps as OpNames>::op_name(OpId { arity: 1, id: 999 }),
        "unknown_op"
    );

    // Sanity check unary "-" does not add parentheses for a leaf.
    let s = string_tree(&(-a.clone()), StringTreeOptions::default());
    assert_eq!(s, "-a");

    // Exercise default variable naming when metadata doesn't provide names.
    let anon = common::var(0) + common::var(1);
    let s = string_tree(&anon, StringTreeOptions::default());
    assert_eq!(s, "x0 + x1");

    // Exercise `Sub<T>` output via a manual check.
    let ex = common::var(0) - 3.2;
    let (y, ok) = eval_tree_array::<f64, TestOps, 3>(&ex, x_view, &opts);
    assert!(ok);
    for (i, &v) in y.iter().enumerate() {
        assert_relative_eq!(
            v,
            x_data[i * x_view.ncols()] - 3.2,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
}
