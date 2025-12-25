use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::{PNode, Src};
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::operator_enum::builtin::BuiltinOp;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_enum::scalar::{GradKernelCtx, GradRef, HasOp, SrcRef, grad_nary};
use dynamic_expressions::operator_registry::OpRegistry;
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{
    EvalOptions, EvalPlan, GradContext, compile_plan, eval_grad_tree_array, eval_tree_array, proptest_utils,
};
use ndarray::Array2;
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

const N_FEATURES: usize = 3;
const MAX_CONSTS: usize = 20;
const D_TEST: usize = 2;
const FD_STEP: f64 = 1e-4;
const ATOL_FD: f64 = 1e-7;
const RTOL_FD: f64 = 1e-7;
const ATOL_EXACT: f64 = 1e-12;
const RTOL_EXACT: f64 = 1e-12;

dynamic_expressions::custom_opset! {
    struct TestOps<f64> {
        1 {
            sin {
                eval(args) { args[0].sin() },
                partial(args, idx) {
                    match idx {
                        0 => args[0].cos(),
                        _ => unreachable!(),
                    }
                },
            }
            cos {
                eval(args) { args[0].cos() },
                partial(args, idx) {
                    match idx {
                        0 => -args[0].sin(),
                        _ => unreachable!(),
                    }
                },
            }
        }
        2 {
            add {
                eval(args) { args[0] + args[1] },
                partial(_args, idx) {
                    match idx {
                        0 | 1 => 1.0,
                        _ => unreachable!(),
                    }
                },
            }
            sub {
                eval(args) { args[0] - args[1] },
                partial(_args, idx) {
                    match idx {
                        0 => 1.0,
                        1 => -1.0,
                        _ => unreachable!(),
                    }
                },
            }
            mul {
                eval(args) { args[0] * args[1] },
                partial(args, idx) {
                    match idx {
                        0 => args[1],
                        1 => args[0],
                        _ => unreachable!(),
                    }
                },
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy, Debug)]
enum UnaryOp {
    Sin,
    Cos,
}

fn op_unary<Op>() -> u16
where
    BuiltinOpsF64: HasOp<Op, 1>,
{
    <BuiltinOpsF64 as HasOp<Op, 1>>::ID
}

fn op_binary<Op>() -> u16
where
    BuiltinOpsF64: HasOp<Op, 2>,
{
    <BuiltinOpsF64 as HasOp<Op, 2>>::ID
}

fn safe_unary_ops() -> Vec<u16> {
    vec![op_unary::<builtin::Cos>(), op_unary::<builtin::Sin>()]
}

fn safe_binary_ops() -> Vec<u16> {
    vec![
        op_binary::<builtin::Add>(),
        op_binary::<builtin::Sub>(),
        op_binary::<builtin::Mul>(),
    ]
}

fn arb_forward_slot_nodes(
    n_features: usize,
    n_consts: usize,
    unary_ops: Vec<u16>,
    binary_ops: Vec<u16>,
) -> BoxedStrategy<Vec<PNode>> {
    let leaf = proptest_utils::arb_leaf_node(n_features, n_consts).boxed();
    let simple = proptest_utils::arb_shallow_postfix_nodes(n_features, n_consts, &unary_ops, &binary_ops, false);

    let template_a = {
        let ops = binary_ops.clone();
        (
            simple.clone(),
            simple.clone(),
            simple.clone(),
            leaf.clone(),
            prop::sample::select(ops.clone()),
            prop::sample::select(ops.clone()),
            prop::sample::select(ops),
        )
            .prop_map(|(a, b, c, leaf_d, op_ab, op_cd, op_final)| {
                let mut nodes = Vec::with_capacity(a.len() + b.len() + c.len() + 4);
                nodes.extend(a);
                nodes.extend(b);
                nodes.push(PNode::Op { arity: 2, op: op_ab });
                nodes.extend(c);
                nodes.push(leaf_d);
                nodes.push(PNode::Op { arity: 2, op: op_cd });
                nodes.push(PNode::Op { arity: 2, op: op_final });
                nodes
            })
    };

    let template_b = {
        let bin_ops = binary_ops.clone();
        let unary_ops = unary_ops.clone();
        (
            simple.clone(),
            simple,
            leaf,
            prop::sample::select(bin_ops.clone()),
            prop::sample::select(unary_ops),
            prop::sample::select(bin_ops),
        )
            .prop_map(|(a, b, leaf_c, op_ab, op_unary, op_final)| {
                let mut nodes = Vec::with_capacity(a.len() + b.len() + 3);
                nodes.extend(a);
                nodes.extend(b);
                nodes.push(PNode::Op { arity: 2, op: op_ab });
                nodes.push(leaf_c);
                nodes.push(PNode::Op { arity: 1, op: op_unary });
                nodes.push(PNode::Op { arity: 2, op: op_final });
                nodes
            })
    };

    prop_oneof![template_a, template_b].boxed()
}

fn custom_op_id(name: &str, arity: u8) -> u16 {
    for info in TestOps::registry() {
        if info.op.arity == arity && info.name == name {
            return info.op.id;
        }
    }
    panic!("missing custom op {name} arity {arity}");
}

fn custom_unary_ops() -> Vec<u16> {
    vec![custom_op_id("sin", 1), custom_op_id("cos", 1)]
}

fn custom_binary_ops() -> Vec<u16> {
    vec![custom_op_id("add", 2), custom_op_id("sub", 2), custom_op_id("mul", 2)]
}

#[derive(Clone, Copy, Debug)]
struct OpIds {
    add: u16,
    sub: u16,
    mul: u16,
    sin: u16,
    cos: u16,
}

fn builtin_op_ids() -> OpIds {
    OpIds {
        add: op_binary::<builtin::Add>(),
        sub: op_binary::<builtin::Sub>(),
        mul: op_binary::<builtin::Mul>(),
        sin: op_unary::<builtin::Sin>(),
        cos: op_unary::<builtin::Cos>(),
    }
}

fn custom_op_ids() -> OpIds {
    OpIds {
        add: custom_op_id("add", 2),
        sub: custom_op_id("sub", 2),
        mul: custom_op_id("mul", 2),
        sin: custom_op_id("sin", 1),
        cos: custom_op_id("cos", 1),
    }
}

fn eval_reference<Ops>(
    expr: &PostfixExpr<f64, Ops, D_TEST>,
    x_data: &[f64],
    n_rows: usize,
    variable: bool,
    ids: OpIds,
) -> (Vec<f64>, Vec<f64>) {
    let n_dir = if variable { N_FEATURES } else { expr.consts.len() };

    let mut out_val = vec![0.0f64; n_rows];
    let mut out_grad = vec![0.0f64; n_rows * n_dir];

    for row in 0..n_rows {
        let mut stack: Vec<(f64, Vec<f64>)> = Vec::new();
        for node in &expr.nodes {
            match node {
                PNode::Var { feature } => {
                    let idx = *feature as usize;
                    let val = x_data[idx * n_rows + row];
                    let mut grad = vec![0.0f64; n_dir];
                    if variable && idx < n_dir {
                        grad[idx] = 1.0;
                    }
                    stack.push((val, grad));
                }
                PNode::Const { idx } => {
                    let cidx = *idx as usize;
                    let val = expr.consts[cidx];
                    let mut grad = vec![0.0f64; n_dir];
                    if !variable && cidx < n_dir {
                        grad[cidx] = 1.0;
                    }
                    stack.push((val, grad));
                }
                PNode::Op { arity, op } => match *arity {
                    1 => {
                        let (xv, gx) = stack.pop().expect("unary operand");
                        let (val, grad) = if *op == ids.sin {
                            let v = xv.sin();
                            let scale = xv.cos();
                            let grad = gx.iter().map(|g| g * scale).collect();
                            (v, grad)
                        } else if *op == ids.cos {
                            let v = xv.cos();
                            let scale = -xv.sin();
                            let grad = gx.iter().map(|g| g * scale).collect();
                            (v, grad)
                        } else {
                            panic!("unsupported unary op id {op}");
                        };
                        stack.push((val, grad));
                    }
                    2 => {
                        let (yv, gy) = stack.pop().expect("rhs");
                        let (xv, gx) = stack.pop().expect("lhs");
                        let (val, grad) = if *op == ids.add {
                            let v = xv + yv;
                            let grad = gx.iter().zip_eq(gy.iter()).map(|(a, b)| a + b).collect();
                            (v, grad)
                        } else if *op == ids.sub {
                            let v = xv - yv;
                            let grad = gx.iter().zip_eq(gy.iter()).map(|(a, b)| a - b).collect();
                            (v, grad)
                        } else if *op == ids.mul {
                            let v = xv * yv;
                            let grad = gx.iter().zip_eq(gy.iter()).map(|(a, b)| a * yv + b * xv).collect();
                            (v, grad)
                        } else {
                            panic!("unsupported binary op id {op}");
                        };
                        stack.push((val, grad));
                    }
                    _ => panic!("unsupported arity {arity}"),
                },
            }
        }

        let (val, grad) = stack.pop().expect("final");
        assert!(stack.is_empty(), "stack not empty");
        out_val[row] = val;
        for dir in 0..n_dir {
            out_grad[dir * n_rows + row] = grad[dir];
        }
    }

    (out_val, out_grad)
}

fn eval_with_x(expr: &PostfixExpr<f64, BuiltinOpsF64, D_TEST>, x_data: &[f64], n_rows: usize) -> Vec<f64> {
    let x = Array2::from_shape_vec((N_FEATURES, n_rows), x_data.to_vec()).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (y, ok) = eval_tree_array::<f64, BuiltinOpsF64, D_TEST>(expr, x.view(), &opts);
    assert!(ok);
    y
}

fn nodes_have_var(nodes: &[dynamic_expressions::node::PNode]) -> bool {
    nodes
        .iter()
        .any(|n| matches!(n, dynamic_expressions::node::PNode::Var { .. }))
}

fn nodes_have_op(nodes: &[dynamic_expressions::node::PNode]) -> bool {
    nodes
        .iter()
        .any(|n| matches!(n, dynamic_expressions::node::PNode::Op { .. }))
}

fn plan_has_forward_slot(plan: &EvalPlan<D_TEST>) -> bool {
    plan.instrs.iter().any(|instr| {
        instr
            .args
            .iter()
            .take(instr.arity as usize)
            .any(|src| matches!(src, Src::Slot(s) if *s > instr.dst))
    })
}

fn finite_diff_feature(
    expr: &PostfixExpr<f64, BuiltinOpsF64, D_TEST>,
    x_data: &[f64],
    n_rows: usize,
    dir: usize,
) -> Vec<f64> {
    let mut plus1 = x_data.to_vec();
    let mut minus1 = x_data.to_vec();
    let mut plus2 = x_data.to_vec();
    let mut minus2 = x_data.to_vec();
    for row in 0..n_rows {
        let idx = dir * n_rows + row;
        plus1[idx] += FD_STEP;
        minus1[idx] -= FD_STEP;
        plus2[idx] += 2.0 * FD_STEP;
        minus2[idx] -= 2.0 * FD_STEP;
    }
    let y_plus1 = eval_with_x(expr, &plus1, n_rows);
    let y_minus1 = eval_with_x(expr, &minus1, n_rows);
    let y_plus2 = eval_with_x(expr, &plus2, n_rows);
    let y_minus2 = eval_with_x(expr, &minus2, n_rows);
    y_plus2
        .iter()
        .zip_eq(&y_plus1)
        .zip_eq(y_minus1.iter().zip_eq(&y_minus2))
        .map(|((p2, p1), (m1, m2))| (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * FD_STEP))
        .collect()
}

fn assert_close_tol(a: &[f64], b: &[f64], atol: f64, rtol: f64) {
    for (&av, &bv) in a.iter().zip_eq(b) {
        let diff = (av - bv).abs();
        let tol = atol + rtol * av.abs().max(bv.abs());
        assert!(diff <= tol, "diff {diff} > tol {tol} (a={av}, b={bv})");
    }
}

fn assert_close_fd(a: &[f64], b: &[f64]) {
    assert_close_tol(a, b, ATOL_FD, RTOL_FD);
}

fn assert_close_exact(a: &[f64], b: &[f64]) {
    assert_close_tol(a, b, ATOL_EXACT, RTOL_EXACT);
}

#[derive(Clone, Copy, Debug)]
enum ArgKind {
    Slice,
    Const,
}

#[derive(Clone, Copy, Debug)]
enum GradKind {
    Slice,
    Basis,
    Zero,
}

fn arg_value(kind: ArgKind, slice: &[f64], c: f64, row: usize) -> f64 {
    match kind {
        ArgKind::Slice => slice[row],
        ArgKind::Const => c,
    }
}

fn grad_value_dir(kind: GradKind, slice: &[f64], dir: usize, row: usize, n_rows: usize, basis_dir: usize) -> f64 {
    match kind {
        GradKind::Slice => slice[dir * n_rows + row],
        GradKind::Basis => {
            if dir == basis_dir {
                1.0
            } else {
                0.0
            }
        }
        GradKind::Zero => 0.0,
    }
}

#[test]
fn grad_custom_ops_const_const_zero_grad() {
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let n_rows = 3usize;
    let x_data = vec![
        0.2, -0.1, 0.3, // feature 0
        -0.4, 0.9, 0.1, // feature 1
        0.5, 0.6, -0.3, // feature 2
    ];
    let x: Array2<f64> = Array2::from_shape_vec((N_FEATURES, n_rows), x_data).unwrap();

    let add_id = custom_op_id("add", 2);
    let expr = PostfixExpr::<f64, TestOps, D_TEST>::new(
        vec![
            PNode::Const { idx: 0 },
            PNode::Const { idx: 1 },
            PNode::Op { arity: 2, op: add_id },
        ],
        vec![0.2, -0.3, 0.5],
        Metadata::default(),
    );

    let mut gctx = GradContext::<f64, D_TEST>::new(n_rows);
    let (_eval, grad, ok) = eval_grad_tree_array::<f64, TestOps, D_TEST>(&expr, x.view(), true, &mut gctx, &opts);
    assert!(ok);
    assert!(grad.data.iter().all(|&g| g == 0.0));
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn grad_apply_binary_matches_manual_formula(
        op in prop_oneof![Just(BinaryOp::Add), Just(BinaryOp::Sub), Just(BinaryOp::Mul)],
        ax in prop_oneof![Just(ArgKind::Slice), Just(ArgKind::Const)],
        ay in prop_oneof![Just(ArgKind::Slice), Just(ArgKind::Const)],
        gx in prop_oneof![Just(GradKind::Slice), Just(GradKind::Basis), Just(GradKind::Zero)],
        gy in prop_oneof![Just(GradKind::Slice), Just(GradKind::Basis), Just(GradKind::Zero)],
        x_const in -1.0f64..=1.0,
        y_const in -1.0f64..=1.0,
        (n_dir, n_rows, x_slice, y_slice, dx_slice, dy_slice, basis_x, basis_y) in
            (1usize..4, 1usize..6).prop_flat_map(|(n_dir, n_rows)| {
                (
                    Just(n_dir),
                    Just(n_rows),
                    prop::collection::vec(-1.0f64..=1.0, n_rows),
                    prop::collection::vec(-1.0f64..=1.0, n_rows),
                    prop::collection::vec(-1.0f64..=1.0, n_dir * n_rows),
                    prop::collection::vec(-1.0f64..=1.0, n_dir * n_rows),
                    0usize..n_dir,
                    0usize..n_dir,
                )
            }),
    ) {
        prop_assume!(!(matches!(ax, ArgKind::Const) && matches!(gx, GradKind::Slice)));
        prop_assume!(!(matches!(ay, ArgKind::Const) && matches!(gy, GradKind::Slice)));

        let opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        let args = [
            match ax {
                ArgKind::Slice => SrcRef::Slice(&x_slice),
                ArgKind::Const => SrcRef::Const(x_const),
            },
            match ay {
                ArgKind::Slice => SrcRef::Slice(&y_slice),
                ArgKind::Const => SrcRef::Const(y_const),
            },
        ];

        let arg_grads = [
            match gx {
                GradKind::Slice => GradRef::Slice(&dx_slice),
                GradKind::Basis => GradRef::Basis(basis_x),
                GradKind::Zero => GradRef::Zero,
            },
            match gy {
                GradKind::Slice => GradRef::Slice(&dy_slice),
                GradKind::Basis => GradRef::Basis(basis_y),
                GradKind::Zero => GradRef::Zero,
            },
        ];

        let mut out_val = vec![f64::NAN; n_rows];
        let mut out_grad = vec![f64::NAN; n_rows * n_dir];
        let ctx = GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir,
            n_rows,
            opts: &opts,
        };

        let ok = match op {
            BinaryOp::Add => grad_nary::<2, f64>(builtin::Add::eval, builtin::Add::partial, ctx),
            BinaryOp::Sub => grad_nary::<2, f64>(builtin::Sub::eval, builtin::Sub::partial, ctx),
            BinaryOp::Mul => grad_nary::<2, f64>(builtin::Mul::eval, builtin::Mul::partial, ctx),
        };
        prop_assert!(ok);

        let mut expected = vec![0.0f64; n_rows * n_dir];
        for dir in 0..n_dir {
            for row in 0..n_rows {
                let x = arg_value(ax, &x_slice, x_const, row);
                let y = arg_value(ay, &y_slice, y_const, row);
                let dx = grad_value_dir(gx, &dx_slice, dir, row, n_rows, basis_x);
                let dy = grad_value_dir(gy, &dy_slice, dir, row, n_rows, basis_y);
                expected[dir * n_rows + row] = match op {
                    BinaryOp::Add => dx + dy,
                    BinaryOp::Sub => dx - dy,
                    BinaryOp::Mul => dx * y + dy * x,
                };
            }
        }
        assert_close_exact(&out_grad, &expected);
    }

    #[test]
    fn grad_apply_unary_matches_manual_formula(
        op in prop_oneof![Just(UnaryOp::Sin), Just(UnaryOp::Cos)],
        ax in prop_oneof![Just(ArgKind::Slice), Just(ArgKind::Const)],
        gx in prop_oneof![Just(GradKind::Slice), Just(GradKind::Basis), Just(GradKind::Zero)],
        x_const in -1.0f64..=1.0,
        (n_dir, n_rows, x_slice, dx_slice, basis_x) in (1usize..4, 1usize..6).prop_flat_map(
            |(n_dir, n_rows)| {
                (
                    Just(n_dir),
                    Just(n_rows),
                    prop::collection::vec(-1.0f64..=1.0, n_rows),
                    prop::collection::vec(-1.0f64..=1.0, n_dir * n_rows),
                    0usize..n_dir,
                )
            },
        ),
    ) {
        prop_assume!(!(matches!(ax, ArgKind::Const) && matches!(gx, GradKind::Slice)));

        let opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        let args = [match ax {
            ArgKind::Slice => SrcRef::Slice(&x_slice),
            ArgKind::Const => SrcRef::Const(x_const),
        }];

        let arg_grads = [match gx {
            GradKind::Slice => GradRef::Slice(&dx_slice),
            GradKind::Basis => GradRef::Basis(basis_x),
            GradKind::Zero => GradRef::Zero,
        }];

        let mut out_val = vec![f64::NAN; n_rows];
        let mut out_grad = vec![f64::NAN; n_rows * n_dir];
        let ctx = GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir,
            n_rows,
            opts: &opts,
        };

        let ok = match op {
            UnaryOp::Sin => grad_nary::<1, f64>(builtin::Sin::eval, builtin::Sin::partial, ctx),
            UnaryOp::Cos => grad_nary::<1, f64>(builtin::Cos::eval, builtin::Cos::partial, ctx),
        };
        prop_assert!(ok);

        let mut expected = vec![0.0f64; n_rows * n_dir];
        for dir in 0..n_dir {
            for row in 0..n_rows {
                let x = arg_value(ax, &x_slice, x_const, row);
                let dx = grad_value_dir(gx, &dx_slice, dir, row, n_rows, basis_x);
                expected[dir * n_rows + row] = match op {
                    UnaryOp::Sin => dx * x.cos(),
                    UnaryOp::Cos => -dx * x.sin(),
                };
            }
        }
        assert_close_exact(&out_grad, &expected);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn grad_matches_reference_impl(
        (n_consts, nodes, consts) in (0usize..=MAX_CONSTS).prop_flat_map(|n_consts| {
            (
                Just(n_consts),
                proptest_utils::arb_postfix_nodes(
                    N_FEATURES,
                    n_consts,
                    safe_unary_ops(),
                    safe_binary_ops(),
                    Vec::new(),
                    4,
                    48,
                    6,
                ),
                prop::collection::vec(-1.0f64..=1.0, n_consts),
            )
        }),
        (n_rows, x_data) in (1usize..4).prop_flat_map(|n_rows| {
            prop::collection::vec(-1.0f64..=1.0, N_FEATURES * n_rows)
                .prop_map(move |x| (n_rows, x))
        }),
        variable in any::<bool>(),
    ) {
        let expr = PostfixExpr::new(nodes, consts, Metadata::default());
        prop_assert_eq!(expr.consts.len(), n_consts);
        let x: Array2<f64> = Array2::from_shape_vec((N_FEATURES, n_rows), x_data.clone()).unwrap();
        let opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        let mut gctx = GradContext::<f64, D_TEST>::new(n_rows);
        let (eval, grad, ok_grad) =
            eval_grad_tree_array::<f64, BuiltinOpsF64, D_TEST>(&expr, x.view(), variable, &mut gctx, &opts);
        prop_assert!(ok_grad);

        let (ref_val, ref_grad) = eval_reference(&expr, &x_data, n_rows, variable, builtin_op_ids());
        assert_close_exact(&eval, &ref_val);
        assert_close_exact(&grad.data, &ref_grad);
    }

    #[test]
    fn grad_custom_ops_matches_reference_impl(
        (n_consts, nodes, consts) in (0usize..=MAX_CONSTS).prop_flat_map(|n_consts| {
            (
                Just(n_consts),
                proptest_utils::arb_postfix_nodes(
                    N_FEATURES,
                    n_consts,
                    custom_unary_ops(),
                    custom_binary_ops(),
                    Vec::new(),
                    4,
                    48,
                    6,
                ),
                prop::collection::vec(-1.0f64..=1.0, n_consts),
            )
        }),
        (n_rows, x_data) in (1usize..4).prop_flat_map(|n_rows| {
            prop::collection::vec(-1.0f64..=1.0, N_FEATURES * n_rows)
                .prop_map(move |x| (n_rows, x))
        }),
        variable in any::<bool>(),
    ) {
        let expr = PostfixExpr::<f64, TestOps, D_TEST>::new(nodes, consts, Metadata::default());
        prop_assert_eq!(expr.consts.len(), n_consts);
        let x: Array2<f64> = Array2::from_shape_vec((N_FEATURES, n_rows), x_data.clone()).unwrap();
        let opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        let mut gctx = GradContext::<f64, D_TEST>::new(n_rows);
        let (eval, grad, ok_grad) =
            eval_grad_tree_array::<f64, TestOps, D_TEST>(&expr, x.view(), variable, &mut gctx, &opts);
        prop_assert!(ok_grad);

        let (ref_val, ref_grad) = eval_reference(&expr, &x_data, n_rows, variable, custom_op_ids());
        assert_close_exact(&eval, &ref_val);
        assert_close_exact(&grad.data, &ref_grad);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, .. ProptestConfig::default() })]

    #[test]
    fn grad_matches_finite_diff_variables(
        (n_consts, nodes, consts) in (0usize..=MAX_CONSTS).prop_flat_map(|n_consts| {
            (
                Just(n_consts),
                proptest_utils::arb_postfix_nodes(
                    N_FEATURES,
                    n_consts,
                    safe_unary_ops(),
                    safe_binary_ops(),
                    Vec::new(),
                    4,
                    48,
                    6,
                )
                .prop_filter("needs op+var", |nodes| nodes_have_op(nodes) && nodes_have_var(nodes)),
                prop::collection::vec(-1.0f64..=1.0, n_consts),
            )
        }),
        (n_rows, x_data) in (1usize..4).prop_flat_map(|n_rows| {
            prop::collection::vec(-1.0f64..=1.0, N_FEATURES * n_rows)
                .prop_map(move |x| (n_rows, x))
        }),
    ) {
        let expr = PostfixExpr::new(nodes, consts, Metadata::default());
        prop_assert_eq!(expr.consts.len(), n_consts);
        let x: Array2<f64> = Array2::from_shape_vec((N_FEATURES, n_rows), x_data.clone()).unwrap();
        let opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        let mut gctx = GradContext::<f64, D_TEST>::new(n_rows);
        let (_eval, grad, ok) = eval_grad_tree_array::<f64, BuiltinOpsF64, D_TEST>(
            &expr,
            x.view(),
            true,
            &mut gctx,
            &opts,
        );
        prop_assert!(ok);
        prop_assert_eq!(grad.n_dir, N_FEATURES);
        prop_assert_eq!(grad.n_rows, n_rows);

        for dir in 0..N_FEATURES {
            let fd = finite_diff_feature(&expr, &x_data, n_rows, dir);
            let grad_dir = &grad.data[dir * n_rows..(dir + 1) * n_rows];
            assert_close_fd(grad_dir, &fd);
        }
    }

    #[test]
    fn grad_matches_finite_diff_with_forward_slots(
        (n_consts, nodes, consts) in (0usize..=MAX_CONSTS).prop_flat_map(|n_consts| {
            (
                Just(n_consts),
                arb_forward_slot_nodes(N_FEATURES, n_consts, safe_unary_ops(), safe_binary_ops()),
                prop::collection::vec(-1.0f64..=1.0, n_consts),
            )
        }),
        (n_rows, x_data) in (1usize..4).prop_flat_map(|n_rows| {
            prop::collection::vec(-1.0f64..=1.0, N_FEATURES * n_rows)
                .prop_map(move |x| (n_rows, x))
        }),
    ) {
        let expr = PostfixExpr::new(nodes, consts, Metadata::default());
        prop_assert_eq!(expr.consts.len(), n_consts);
        let plan = compile_plan::<D_TEST>(&expr.nodes, N_FEATURES, expr.consts.len());
        prop_assert!(plan_has_forward_slot(&plan));
        let x: Array2<f64> = Array2::from_shape_vec((N_FEATURES, n_rows), x_data.clone()).unwrap();
        let opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        let mut gctx = GradContext::<f64, D_TEST>::new(n_rows);
        let (_eval, grad, ok) =
            eval_grad_tree_array::<f64, BuiltinOpsF64, D_TEST>(&expr, x.view(), true, &mut gctx, &opts);
        prop_assert!(ok);

        for dir in 0..N_FEATURES {
            let fd = finite_diff_feature(&expr, &x_data, n_rows, dir);
            let grad_dir = &grad.data[dir * n_rows..(dir + 1) * n_rows];
            assert_close_fd(grad_dir, &fd);
        }
    }
}
