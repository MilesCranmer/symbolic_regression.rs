use dynamic_expressions::math::*;
use dynamic_expressions::{eval_tree_array, EvalOptions, PNode, PostfixExpr};
use ndarray::Array2;

dynamic_expressions::opset! {
    pub struct AllOps<f64>;
    ops {
        (1, U1) {
            Sin, Cos, Tan,
            Asin, Acos, Atan,
            Sinh, Cosh, Tanh,
            Asinh, Acosh, Atanh,
            Sec, Csc, Cot,
            Exp, Exp2, Expm1,
            Log, Log2, Log10, Log1p,
            Sqrt, Cbrt,
            Abs, Abs2, Inv,
            Sign, Identity,
            Neg,
        }
        (2, B2) {
            Add, Sub, Mul, Div,
            Pow, Atan2,
            Min, Max,
        }
        (3, T3) { Fma, Clamp, }
    }
}

fn var(feature: u16) -> PostfixExpr<f64, AllOps, 3> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Default::default())
}

fn c(value: f64) -> PostfixExpr<f64, AllOps, 3> {
    PostfixExpr::new(
        vec![PNode::Const { idx: 0 }],
        vec![value],
        Default::default(),
    )
}

#[test]
fn math_wrappers_build_and_eval() {
    let n_rows = 4usize;
    let data = vec![0.2f64; 2 * n_rows];
    let x = Array2::from_shape_vec((n_rows, 2), data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    // Unary wrappers (constants chosen to satisfy domain constraints).
    let unary = [
        sin(c(0.3)),
        cos(c(0.3)),
        tan(c(0.3)),
        asin(c(0.2)),
        acos(c(0.2)),
        atan(c(0.2)),
        sinh(c(0.2)),
        cosh(c(0.2)),
        tanh(c(0.2)),
        asinh(c(0.2)),
        acosh(c(2.0)),
        atanh(c(0.2)),
        sec(c(0.3)),
        csc(c(1.2)),
        cot(c(1.2)),
        exp(c(0.2)),
        exp2(c(0.2)),
        expm1(c(0.2)),
        log(c(1.3)),
        log2(c(1.3)),
        log10(c(1.3)),
        log1p(c(0.2)),
        sqrt(c(2.0)),
        cbrt(c(2.0)),
        abs(c(-0.7)),
        abs2(c(-0.7)),
        inv(c(2.0)),
        sign(c(-0.7)),
        identity(c(0.7)),
        neg(c(0.7)),
    ];
    for ex in unary {
        let (_y, ok) = eval_tree_array::<f64, AllOps, 3>(&ex, x.view(), &opts);
        assert!(ok);
    }

    // Unary wrappers using variables (exercise var path).
    let y0 = eval_tree_array::<f64, AllOps, 3>(&cos(var(0)), x.view(), &opts).0;
    assert!((y0[0] - 0.2f64.cos()).abs() < 1e-12);
    let y1 = eval_tree_array::<f64, AllOps, 3>(&neg(var(0)), x.view(), &opts).0;
    assert!((y1[0] + 0.2).abs() < 1e-12);

    // Binary wrappers.
    let bin = [
        add(c(1.0), c(2.0)),
        sub(c(1.0), c(2.0)),
        mul(c(2.0), c(3.0)),
        div(c(3.0), c(2.0)),
        pow(c(1.3), c(0.7)),
        atan2(c(0.3), c(1.7)),
        min(c(1.0), c(2.0)),
        max(c(1.0), c(2.0)),
    ];
    for ex in bin {
        let (_y, ok) = eval_tree_array::<f64, AllOps, 3>(&ex, x.view(), &opts);
        assert!(ok);
    }

    // Ternary wrappers.
    let tri = [
        fma(c(2.0), c(4.0), c(3.0)),
        clamp(c(-2.0), c(-1.0), c(1.0)),
        clamp(c(0.0), c(-1.0), c(1.0)),
        clamp(c(2.0), c(-1.0), c(1.0)),
    ];
    for ex in tri {
        let (_y, ok) = eval_tree_array::<f64, AllOps, 3>(&ex, x.view(), &opts);
        assert!(ok);
    }
}
