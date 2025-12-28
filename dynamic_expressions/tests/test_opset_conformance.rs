use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{
    DiffContext, EvalContext, EvalOptions, GradContext, OpId, OperatorSet, PNode, PostfixExpr, eval_diff_tree_array,
    eval_grad_tree_array, eval_tree_array, eval_tree_array_into,
};
use ndarray::Array2;

type Ops = BuiltinOpsF64;
const DMAX: usize = 3;

fn assert_eq_nan_ok(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        assert!(x == y, "mismatch at {i}: {x:?} vs {y:?}");
    }
}

fn make_x<const D: usize>(n_rows: usize) -> Array2<f64> {
    let mut data = vec![0.0f64; D * n_rows];
    for feature in 0..D {
        for row in 0..n_rows {
            let base = (row as f64 + 1.0) * 0.01 * (feature as f64 + 1.0);
            data[feature * n_rows + row] = if feature == 2 { -base } else { base };
        }
    }
    Array2::from_shape_vec((D, n_rows), data).unwrap()
}

fn expr_for_op_vars<Opset, const D: usize>(op: OpId) -> PostfixExpr<f64, Opset, D>
where
    Opset: OperatorSet<T = f64>,
{
    let mut nodes = Vec::with_capacity(op.arity as usize + 1);
    for feature in 0..op.arity {
        nodes.push(PNode::Var {
            feature: feature as u16,
        });
    }
    nodes.push(PNode::Op {
        arity: op.arity,
        op: op.id,
    });
    PostfixExpr::new(nodes, vec![], Default::default())
}

fn expr_for_op_consts<Opset, const D: usize>(op: OpId) -> PostfixExpr<f64, Opset, D>
where
    Opset: OperatorSet<T = f64>,
{
    let seed = [0.2, 1.3, -0.7];
    let arity = op.arity as usize;
    let consts = seed[..arity].to_vec();
    let mut nodes = Vec::with_capacity(arity + 1);
    for idx in 0..arity {
        nodes.push(PNode::Const { idx: idx as u16 });
    }
    nodes.push(PNode::Op {
        arity: op.arity,
        op: op.id,
    });
    PostfixExpr::new(nodes, consts, Default::default())
}

fn run_opset_conformance<Opset, const D: usize>()
where
    Opset: OperatorSet<T = f64>,
{
    let n_rows = 8;
    let x = make_x::<D>(n_rows);
    let x_view = x.view();

    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut ectx = EvalContext::<f64, D>::new(n_rows);
    let mut gctx = GradContext::<f64, D>::new(n_rows);
    let mut dctx = DiffContext::<f64, D>::new(n_rows);

    Opset::for_each_op(|op| {
        assert!(Opset::meta(op).is_some());

        let name = Opset::name(op);
        let display = Opset::display(op);
        assert!(!name.is_empty());
        assert!(!display.is_empty());

        let expr_v = expr_for_op_vars::<Opset, D>(op);

        let (y, ok_eval) = eval_tree_array::<f64, Opset, D>(&expr_v, x_view, &opts);

        let mut out = vec![f64::NAN; n_rows];
        let ok_into = eval_tree_array_into::<f64, Opset, D>(&mut out, &expr_v, x_view, &mut ectx, &opts);
        assert_eq!(ok_eval, ok_into);
        assert_eq_nan_ok(&y, &out);

        let (_yv, grad, ok_grad) = eval_grad_tree_array::<f64, Opset, D>(&expr_v, x_view, true, &mut gctx, &opts);
        assert_eq!(ok_grad, ok_eval);

        for dir in 0..x.nrows() {
            let (_yd, d, ok_diff) = eval_diff_tree_array::<f64, Opset, D>(&expr_v, x_view, dir, &mut dctx, &opts);
            assert_eq!(ok_diff, ok_eval);
            let gdir = &grad.data[dir * n_rows..(dir + 1) * n_rows];
            assert_eq_nan_ok(&d, gdir);
        }

        let expr_c = expr_for_op_consts::<Opset, D>(op);

        let (_yc, gconst, _ok_const) = eval_grad_tree_array::<f64, Opset, D>(&expr_c, x_view, false, &mut gctx, &opts);
        assert_eq!(gconst.n_dir, op.arity as usize);

        let s = string_tree(&expr_v, StringTreeOptions::default());
        assert!(!s.is_empty());
    });
}

#[test]
fn builtin_ops_f64_conformance() {
    run_opset_conformance::<Ops, DMAX>();
}

#[test]
fn root_var_and_root_const_branches_are_exercised() {
    let n_rows = 8;
    let x = make_x::<DMAX>(n_rows);
    let x_view = x.view();

    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };

    let mut ectx = EvalContext::<f64, DMAX>::new(n_rows);
    let mut gctx = GradContext::<f64, DMAX>::new(n_rows);
    let mut dctx = DiffContext::<f64, DMAX>::new(n_rows);

    let expr_var: PostfixExpr<f64, Ops, DMAX> =
        PostfixExpr::new(vec![PNode::Var { feature: 0 }], vec![], Default::default());
    let (y, ok_eval) = eval_tree_array::<f64, Ops, DMAX>(&expr_var, x_view, &opts);
    assert!(ok_eval);
    let mut out = vec![f64::NAN; n_rows];
    let ok_into = eval_tree_array_into::<f64, Ops, DMAX>(&mut out, &expr_var, x_view, &mut ectx, &opts);
    assert!(ok_into);
    assert_eq_nan_ok(&y, &out);

    let (_y, grad, ok_grad) = eval_grad_tree_array::<f64, Ops, DMAX>(&expr_var, x_view, true, &mut gctx, &opts);
    assert!(ok_grad);
    for dir in 0..x.nrows() {
        let (_y, d, ok_diff) = eval_diff_tree_array::<f64, Ops, DMAX>(&expr_var, x_view, dir, &mut dctx, &opts);
        assert!(ok_diff);
        let gdir = &grad.data[dir * n_rows..(dir + 1) * n_rows];
        assert_eq_nan_ok(&d, gdir);
    }

    let expr_const: PostfixExpr<f64, Ops, DMAX> =
        PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![f64::NAN], Default::default());
    let (y, ok_eval) = eval_tree_array::<f64, Ops, DMAX>(&expr_const, x_view, &opts);
    assert!(!ok_eval);
    assert!(y.iter().all(|v| v.is_nan()));
    let mut out = vec![0.0f64; n_rows];
    let ok_into = eval_tree_array_into::<f64, Ops, DMAX>(&mut out, &expr_const, x_view, &mut ectx, &opts);
    assert!(!ok_into);
    assert!(out.iter().all(|v| v.is_nan()));
}
