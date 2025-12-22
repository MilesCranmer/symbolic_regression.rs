use dynamic_expressions::operator_enum::builtin::*;

fn fd_unary<Op: BuiltinOp<f64, 1>>(x: f64) -> f64 {
    let eps = 1e-7;
    let vp = <Op as BuiltinOp<f64, 1>>::eval(&[x + eps]);
    let vm = <Op as BuiltinOp<f64, 1>>::eval(&[x - eps]);
    (vp - vm) / (2.0 * eps)
}

fn check_unary<Op: BuiltinOp<f64, 1>>(x: f64, tol: f64) {
    let d = <Op as BuiltinOp<f64, 1>>::partial(&[x], 0);
    let fd = fd_unary::<Op>(x);
    assert!(
        (d - fd).abs() <= tol.max(tol * fd.abs()),
        "d={} fd={} (x={})",
        d,
        fd,
        x
    );
}

fn fd_binary_x<Op: BuiltinOp<f64, 2>>(x: f64, y: f64) -> f64 {
    let eps = 1e-7;
    let vp = <Op as BuiltinOp<f64, 2>>::eval(&[x + eps, y]);
    let vm = <Op as BuiltinOp<f64, 2>>::eval(&[x - eps, y]);
    (vp - vm) / (2.0 * eps)
}

fn fd_binary_y<Op: BuiltinOp<f64, 2>>(x: f64, y: f64) -> f64 {
    let eps = 1e-7;
    let vp = <Op as BuiltinOp<f64, 2>>::eval(&[x, y + eps]);
    let vm = <Op as BuiltinOp<f64, 2>>::eval(&[x, y - eps]);
    (vp - vm) / (2.0 * eps)
}

fn check_binary<Op: BuiltinOp<f64, 2>>(x: f64, y: f64, tol: f64) {
    let dx = <Op as BuiltinOp<f64, 2>>::partial(&[x, y], 0);
    let dy = <Op as BuiltinOp<f64, 2>>::partial(&[x, y], 1);
    let fdx = fd_binary_x::<Op>(x, y);
    let fdy = fd_binary_y::<Op>(x, y);
    assert!(
        (dx - fdx).abs() <= tol.max(tol * fdx.abs()),
        "dx={} fdx={} (x={}, y={})",
        dx,
        fdx,
        x,
        y
    );
    assert!(
        (dy - fdy).abs() <= tol.max(tol * fdy.abs()),
        "dy={} fdy={} (x={}, y={})",
        dy,
        fdy,
        x,
        y
    );
}

#[test]
fn unary_derivatives_match_finite_difference() {
    // Choose points away from singularities / domain boundaries.
    check_unary::<Sin>(0.3, 1e-6);
    check_unary::<Cos>(0.3, 1e-6);
    check_unary::<Tan>(0.3, 1e-5);

    check_unary::<Asin>(0.2, 1e-5);
    check_unary::<Acos>(0.2, 1e-5);
    check_unary::<Atan>(0.2, 1e-6);

    check_unary::<Sinh>(0.2, 1e-6);
    check_unary::<Cosh>(0.2, 1e-6);
    check_unary::<Tanh>(0.2, 1e-6);

    check_unary::<Asinh>(0.2, 1e-6);
    check_unary::<Acosh>(2.0, 1e-6);
    check_unary::<Atanh>(0.2, 1e-5);

    check_unary::<Sec>(0.3, 1e-5);
    check_unary::<Csc>(1.2, 1e-5);
    check_unary::<Cot>(1.2, 1e-5);

    check_unary::<Exp>(0.2, 1e-6);
    check_unary::<Exp2>(0.2, 1e-6);
    check_unary::<Expm1>(0.2, 1e-6);

    check_unary::<Log>(1.3, 1e-6);
    check_unary::<Log2>(1.3, 1e-6);
    check_unary::<Log10>(1.3, 1e-6);
    check_unary::<Log1p>(0.2, 1e-6);

    check_unary::<Sqrt>(2.0, 1e-6);
    check_unary::<Cbrt>(2.0, 1e-5);

    check_unary::<Inv>(2.0, 1e-6);
    check_unary::<Abs2>(-0.7, 1e-6);
}

#[test]
fn binary_derivatives_match_finite_difference() {
    check_binary::<Add>(0.3, -0.7, 1e-6);
    check_binary::<Sub>(0.3, -0.7, 1e-6);
    check_binary::<Mul>(0.3, -0.7, 1e-6);
    check_binary::<Div>(0.3, 1.7, 1e-6);

    check_binary::<Atan2>(0.3, 1.7, 1e-6);
    check_binary::<Pow>(1.3, 0.7, 1e-5);
}

#[test]
fn nonsmooth_ops_have_reasonable_partials_off_boundaries() {
    // Abs: derivative is ±1 away from 0.
    assert_eq!(<Abs as BuiltinOp<f64, 1>>::partial(&[2.0], 0), 1.0);
    assert_eq!(<Abs as BuiltinOp<f64, 1>>::partial(&[-2.0], 0), -1.0);
    assert_eq!(<Abs as BuiltinOp<f64, 1>>::partial(&[0.0], 0), 0.0);

    // Sign/Identity are simple.
    assert_eq!(<Sign as BuiltinOp<f64, 1>>::partial(&[2.0], 0), 0.0);
    assert_eq!(<Identity as BuiltinOp<f64, 1>>::partial(&[2.0], 0), 1.0);

    // Min/Max: away from ties they are 0/1.
    assert_eq!(<Min as BuiltinOp<f64, 2>>::partial(&[1.0, 2.0], 0), 1.0);
    assert_eq!(<Min as BuiltinOp<f64, 2>>::partial(&[1.0, 2.0], 1), 0.0);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::partial(&[1.0, 2.0], 0), 0.0);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::partial(&[1.0, 2.0], 1), 1.0);
    assert_eq!(<Min as BuiltinOp<f64, 2>>::partial(&[2.0, 1.0], 0), 0.0);
    assert_eq!(<Min as BuiltinOp<f64, 2>>::partial(&[2.0, 1.0], 1), 1.0);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::partial(&[2.0, 1.0], 0), 1.0);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::partial(&[2.0, 1.0], 1), 0.0);
    // Reverse order exercises the other eval branch.
    assert_eq!(<Min as BuiltinOp<f64, 2>>::eval(&[2.0, 1.0]), 1.0);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::eval(&[2.0, 1.0]), 2.0);
    // Tie splits gradient evenly.
    assert_eq!(<Min as BuiltinOp<f64, 2>>::partial(&[2.0, 2.0], 0), 0.5);
    assert_eq!(<Min as BuiltinOp<f64, 2>>::partial(&[2.0, 2.0], 1), 0.5);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::partial(&[2.0, 2.0], 0), 0.5);
    assert_eq!(<Max as BuiltinOp<f64, 2>>::partial(&[2.0, 2.0], 1), 0.5);

    // Clamp: inside bounds, derivative wrt x is 1; below/above it's 0.
    let args = [0.0, -1.0, 1.0];
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 0), 1.0);
    let args = [-2.0, -1.0, 1.0];
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 0), 0.0);
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 1), 1.0);
    // When x >= lo, derivative wrt lo is 0.
    let args = [0.0, -1.0, 1.0];
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 1), 0.0);
    let args = [2.0, -1.0, 1.0];
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 0), 0.0);
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 2), 1.0);
    // When x <= hi, derivative wrt hi is 0.
    let args = [0.0, -1.0, 1.0];
    assert_eq!(<Clamp as BuiltinOp<f64, 3>>::partial(&args, 2), 0.0);
}
