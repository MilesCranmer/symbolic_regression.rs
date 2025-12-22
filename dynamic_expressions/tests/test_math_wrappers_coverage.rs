use dynamic_expressions::{EvalOptions, PNode, PostfixExpr, eval_tree_array};
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
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![value], Default::default())
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
        dynamic_expressions::operators::sin(c(0.3)),
        dynamic_expressions::operators::cos(c(0.3)),
        dynamic_expressions::operators::tan(c(0.3)),
        dynamic_expressions::operators::asin(c(0.2)),
        dynamic_expressions::operators::acos(c(0.2)),
        dynamic_expressions::operators::atan(c(0.2)),
        dynamic_expressions::operators::sinh(c(0.2)),
        dynamic_expressions::operators::cosh(c(0.2)),
        dynamic_expressions::operators::tanh(c(0.2)),
        dynamic_expressions::operators::asinh(c(0.2)),
        dynamic_expressions::operators::acosh(c(2.0)),
        dynamic_expressions::operators::atanh(c(0.2)),
        dynamic_expressions::operators::sec(c(0.3)),
        dynamic_expressions::operators::csc(c(1.2)),
        dynamic_expressions::operators::cot(c(1.2)),
        dynamic_expressions::operators::exp(c(0.2)),
        dynamic_expressions::operators::exp2(c(0.2)),
        dynamic_expressions::operators::expm1(c(0.2)),
        dynamic_expressions::operators::log(c(1.3)),
        dynamic_expressions::operators::log2(c(1.3)),
        dynamic_expressions::operators::log10(c(1.3)),
        dynamic_expressions::operators::log1p(c(0.2)),
        dynamic_expressions::operators::sqrt(c(2.0)),
        dynamic_expressions::operators::cbrt(c(2.0)),
        dynamic_expressions::operators::abs(c(-0.7)),
        dynamic_expressions::operators::abs2(c(-0.7)),
        dynamic_expressions::operators::inv(c(2.0)),
        dynamic_expressions::operators::sign(c(-0.7)),
        dynamic_expressions::operators::identity(c(0.7)),
        dynamic_expressions::operators::neg(c(0.7)),
    ];
    for ex in unary {
        let (_y, ok) = eval_tree_array::<f64, AllOps, 3>(&ex, x.view(), &opts);
        assert!(ok);
    }

    // Unary wrappers using variables (exercise var path).
    let y0 = eval_tree_array::<f64, AllOps, 3>(&dynamic_expressions::operators::cos(var(0)), x.view(), &opts).0;
    assert!((y0[0] - 0.2f64.cos()).abs() < 1e-12);
    let y1 = eval_tree_array::<f64, AllOps, 3>(&dynamic_expressions::operators::neg(var(0)), x.view(), &opts).0;
    assert!((y1[0] + 0.2).abs() < 1e-12);

    // Binary wrappers.
    let bin = [
        dynamic_expressions::operators::add(c(1.0), c(2.0)),
        dynamic_expressions::operators::sub(c(1.0), c(2.0)),
        dynamic_expressions::operators::mul(c(2.0), c(3.0)),
        dynamic_expressions::operators::div(c(3.0), c(2.0)),
        dynamic_expressions::operators::pow(c(1.3), c(0.7)),
        dynamic_expressions::operators::atan2(c(0.3), c(1.7)),
        dynamic_expressions::operators::min(c(1.0), c(2.0)),
        dynamic_expressions::operators::max(c(1.0), c(2.0)),
    ];
    for ex in bin {
        let (_y, ok) = eval_tree_array::<f64, AllOps, 3>(&ex, x.view(), &opts);
        assert!(ok);
    }

    // Ternary wrappers.
    let tri = [
        dynamic_expressions::operators::fma(c(2.0), c(4.0), c(3.0)),
        dynamic_expressions::operators::clamp(c(-2.0), c(-1.0), c(1.0)),
        dynamic_expressions::operators::clamp(c(0.0), c(-1.0), c(1.0)),
        dynamic_expressions::operators::clamp(c(2.0), c(-1.0), c(1.0)),
    ];
    for ex in tri {
        let (_y, ok) = eval_tree_array::<f64, AllOps, 3>(&ex, x.view(), &opts);
        assert!(ok);
    }
}
