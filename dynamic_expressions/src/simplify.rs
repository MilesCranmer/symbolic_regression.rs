use crate::evaluate::EvalOptions;
use crate::expression::PostfixExpr;
use crate::node::PNode;
use crate::operator_enum::scalar::{EvalKernelCtx, OpId, ScalarOpSet, SrcRef};
use crate::operator_registry::OpRegistry;
use crate::utils::compress_constants;
use num_traits::Float;

#[derive(Clone, Copy)]
struct Frame<T> {
    start: usize,
    is_const: bool,
    const_value: T,
}

#[derive(Clone, Copy)]
struct ArenaNode<const D: usize> {
    kind: ArenaNodeKind<D>,
}

#[derive(Clone, Copy)]
enum ArenaNodeKind<const D: usize> {
    Var(u16),
    Const(u16),
    Op {
        arity: u8,
        op: u16,
        children: [usize; D],
    },
}

#[derive(Clone, Copy, Debug)]
struct CombineCfg {
    add_id: Option<u16>,
    mul_id: Option<u16>,
    sub_id: Option<u16>,
}

fn lookup_id_with_arity<Ops: OpRegistry>(token: &str, arity: u8) -> Option<u16> {
    Ops::registry()
        .iter()
        .find(|info| info.op.arity == arity && info.matches_token(token))
        .map(|info| info.op.id)
}

fn is_const<const D: usize>(arena: &[ArenaNode<D>], id: usize) -> Option<u16> {
    match arena[id].kind {
        ArenaNodeKind::Const(idx) => Some(idx),
        _ => None,
    }
}

fn is_op<const D: usize>(
    arena: &[ArenaNode<D>],
    id: usize,
    op_id: u16,
) -> Option<(u8, [usize; D])> {
    match arena[id].kind {
        ArenaNodeKind::Op {
            arity,
            op,
            children,
        } if op == op_id => Some((arity, children)),
        _ => None,
    }
}

fn append_const<T: Float>(consts: &mut Vec<T>, v: T) -> Option<u16> {
    if !v.is_finite() {
        return None;
    }
    if consts.len() > u16::MAX as usize {
        return None;
    }
    let idx = consts.len() as u16;
    consts.push(v);
    Some(idx)
}

fn combine_node<T: Float, const D: usize>(
    id: usize,
    arena: &mut Vec<ArenaNode<D>>,
    consts: &mut Vec<T>,
    cfg: CombineCfg,
    changed: &mut bool,
) -> usize {
    let ArenaNodeKind::Op {
        arity,
        op,
        mut children,
    } = arena[id].kind
    else {
        return id;
    };

    for cid in children.iter_mut().take(arity as usize) {
        *cid = combine_node(*cid, arena, consts, cfg, changed);
    }
    arena[id].kind = ArenaNodeKind::Op {
        arity,
        op,
        children,
    };

    if arity != 2 {
        return id;
    }

    let op_is_add = cfg.add_id == Some(op);
    let op_is_mul = cfg.mul_id == Some(op);
    let op_is_sub = cfg.sub_id == Some(op);

    if op_is_add || op_is_mul {
        let mut left = children[0];
        let mut right = children[1];

        // DE.jl behavior: if left is const, swap so const is on the right (even if right is also const).
        if is_const(arena, left).is_some() {
            core::mem::swap(&mut left, &mut right);
            *changed = true;
        }

        let Some(top_idx) = is_const(arena, right) else {
            let mut new_children = children;
            new_children[0] = left;
            new_children[1] = right;
            arena[id].kind = ArenaNodeKind::Op {
                arity,
                op,
                children: new_children,
            };
            return id;
        };
        let top_v = consts[top_idx as usize];

        if let Some((below_arity, below_children)) = is_op(arena, left, op) {
            if below_arity == 2 {
                let b0 = below_children[0];
                let b1 = below_children[1];
                let (const_leaf, const_idx) = if let Some(ci) = is_const(arena, b0) {
                    (b0, ci)
                } else if let Some(ci) = is_const(arena, b1) {
                    (b1, ci)
                } else {
                    let mut new_children = children;
                    new_children[0] = left;
                    new_children[1] = right;
                    arena[id].kind = ArenaNodeKind::Op {
                        arity,
                        op,
                        children: new_children,
                    };
                    return id;
                };

                let below_v = consts[const_idx as usize];
                if !below_v.is_finite() || !top_v.is_finite() {
                    let mut new_children = children;
                    new_children[0] = left;
                    new_children[1] = right;
                    arena[id].kind = ArenaNodeKind::Op {
                        arity,
                        op,
                        children: new_children,
                    };
                    return id;
                }

                let new_v = if op_is_add {
                    below_v + top_v
                } else {
                    below_v * top_v
                };
                let Some(new_idx) = append_const(consts, new_v) else {
                    let mut new_children = children;
                    new_children[0] = left;
                    new_children[1] = right;
                    arena[id].kind = ArenaNodeKind::Op {
                        arity,
                        op,
                        children: new_children,
                    };
                    return id;
                };
                arena[const_leaf].kind = ArenaNodeKind::Const(new_idx);
                *changed = true;
                return left;
            }
        }

        let mut new_children = children;
        new_children[0] = left;
        new_children[1] = right;
        arena[id].kind = ArenaNodeKind::Op {
            arity,
            op,
            children: new_children,
        };
        return id;
    }

    if op_is_sub {
        // Patterns cloned from DynamicExpressions.jl:
        // 1) (cA - (cB - x)) => (x - (cB - cA))
        // 2) (cA - (x - cB)) => ((cA + cB) - x)
        // 3) ((cA - x) - cB) => ((cA - cB) - x)
        // 4) ((x - cA) - cB) => (x - (cA + cB))

        let left = children[0];
        let right = children[1];

        // Pattern 1 / 2: left is const, right is (sub ...).
        if let Some(a_idx) = is_const(arena, left) {
            if let Some((_r_arity, r_children)) = is_op(arena, right, op) {
                let r0 = r_children[0];
                let r1 = r_children[1];
                let a_v = consts[a_idx as usize];

                if let Some(b_idx) = is_const(arena, r0) {
                    // Pattern 1: cA - (cB - x) => x - (cB - cA)
                    let b_v = consts[b_idx as usize];
                    if a_v.is_finite() && b_v.is_finite() {
                        let new_v = b_v - a_v;
                        if let Some(new_idx) = append_const(consts, new_v) {
                            arena[r0].kind = ArenaNodeKind::Const(new_idx);
                            let mut new_children = children;
                            new_children[0] = r1;
                            new_children[1] = r0;
                            arena[id].kind = ArenaNodeKind::Op {
                                arity,
                                op,
                                children: new_children,
                            };
                            *changed = true;
                            return id;
                        }
                    }
                } else if let Some(b_idx) = is_const(arena, r1) {
                    // Pattern 2: cA - (x - cB) => (cA + cB) - x
                    let b_v = consts[b_idx as usize];
                    if a_v.is_finite() && b_v.is_finite() {
                        let new_v = a_v + b_v;
                        if let Some(new_idx) = append_const(consts, new_v) {
                            arena[left].kind = ArenaNodeKind::Const(new_idx);
                            let mut new_children = children;
                            new_children[0] = left;
                            new_children[1] = r0;
                            arena[id].kind = ArenaNodeKind::Op {
                                arity,
                                op,
                                children: new_children,
                            };
                            *changed = true;
                            return id;
                        }
                    }
                }
            }
        }

        // Pattern 3 / 4: right is const, left is (sub ...).
        if let Some(b_idx) = is_const(arena, right) {
            if let Some((_l_arity, l_children)) = is_op(arena, left, op) {
                let l0 = l_children[0];
                let l1 = l_children[1];
                let b_v = consts[b_idx as usize];

                if let Some(a_idx) = is_const(arena, l0) {
                    // Pattern 3: (cA - x) - cB => (cA - cB) - x
                    let a_v = consts[a_idx as usize];
                    if a_v.is_finite() && b_v.is_finite() {
                        let new_v = a_v - b_v;
                        if let Some(new_idx) = append_const(consts, new_v) {
                            arena[l0].kind = ArenaNodeKind::Const(new_idx);
                            let mut new_children = children;
                            new_children[0] = l0;
                            new_children[1] = l1;
                            arena[id].kind = ArenaNodeKind::Op {
                                arity,
                                op,
                                children: new_children,
                            };
                            *changed = true;
                            return id;
                        }
                    }
                } else if let Some(a_idx) = is_const(arena, l1) {
                    // Pattern 4: (x - cA) - cB => x - (cA + cB)
                    let a_v = consts[a_idx as usize];
                    if a_v.is_finite() && b_v.is_finite() {
                        let new_v = a_v + b_v;
                        if let Some(new_idx) = append_const(consts, new_v) {
                            arena[l1].kind = ArenaNodeKind::Const(new_idx);
                            let mut new_children = children;
                            new_children[0] = l0;
                            new_children[1] = l1;
                            arena[id].kind = ArenaNodeKind::Op {
                                arity,
                                op,
                                children: new_children,
                            };
                            *changed = true;
                            return id;
                        }
                    }
                }
            }
        }
    }

    id
}

fn emit_postfix<const D: usize>(id: usize, arena: &[ArenaNode<D>], out: &mut Vec<PNode>) {
    match arena[id].kind {
        ArenaNodeKind::Var(f) => out.push(PNode::Var { feature: f }),
        ArenaNodeKind::Const(idx) => out.push(PNode::Const { idx }),
        ArenaNodeKind::Op {
            arity,
            op,
            children,
        } => {
            for &cid in children.iter().take(arity as usize) {
                emit_postfix(cid, arena, out);
            }
            out.push(PNode::Op { arity, op });
        }
    }
}

fn parse_postfix_to_arena<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
) -> Option<(Vec<ArenaNode<D>>, usize)>
where
    T: Float,
{
    let mut arena: Vec<ArenaNode<D>> = Vec::with_capacity(expr.nodes.len());
    let mut st: Vec<usize> = Vec::new();
    for n in expr.nodes.iter().copied() {
        match n {
            PNode::Var { feature } => {
                let id = arena.len();
                arena.push(ArenaNode {
                    kind: ArenaNodeKind::Var(feature),
                });
                st.push(id);
            }
            PNode::Const { idx } => {
                let id = arena.len();
                arena.push(ArenaNode {
                    kind: ArenaNodeKind::Const(idx),
                });
                st.push(id);
            }
            PNode::Op { arity, op } => {
                let a = arity as usize;
                if st.len() < a {
                    return None;
                }
                let mut children = [0usize; D];
                for j in (0..a).rev() {
                    children[j] = st.pop().expect("arity checked");
                }
                let id = arena.len();
                arena.push(ArenaNode {
                    kind: ArenaNodeKind::Op {
                        arity,
                        op,
                        children,
                    },
                });
                st.push(id);
            }
        }
    }
    if st.len() != 1 {
        return None;
    }
    Some((arena, st[0]))
}

fn combine_operators_in_place_with_cfg<T, Ops, const D: usize>(
    expr: &mut PostfixExpr<T, Ops, D>,
    cfg: CombineCfg,
) -> bool
where
    T: Float,
    Ops: OpRegistry,
{
    let Some((mut arena, root)) = parse_postfix_to_arena(expr) else {
        return false;
    };

    let mut changed = false;
    let new_root = combine_node(root, &mut arena, &mut expr.consts, cfg, &mut changed);
    if !changed && new_root == root {
        return false;
    }

    let mut out: Vec<PNode> = Vec::with_capacity(expr.nodes.len());
    emit_postfix(new_root, &arena, &mut out);
    expr.nodes = out;
    changed
}

pub fn simplify_tree_in_place<T, Ops, const D: usize>(
    expr: &mut PostfixExpr<T, Ops, D>,
    eval_opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    fn push_nonconst_frame<T: Float>(stack: &mut Vec<Frame<T>>, start: usize) {
        stack.push(Frame {
            start,
            is_const: false,
            const_value: T::zero(),
        });
    }

    fn push_const_frame<T: Float>(stack: &mut Vec<Frame<T>>, start: usize, v: T) {
        stack.push(Frame {
            start,
            is_const: true,
            const_value: v,
        });
    }

    let mut out_nodes: Vec<PNode> = Vec::with_capacity(expr.nodes.len());
    let mut out_consts: Vec<T> = expr.consts.clone();
    let mut stack: Vec<Frame<T>> = Vec::new();
    let mut changed = false;

    for node in expr.nodes.iter().copied() {
        match node {
            PNode::Var { .. } => {
                out_nodes.push(node);
                push_nonconst_frame(&mut stack, out_nodes.len() - 1);
            }
            PNode::Const { idx } => {
                let v = out_consts[idx as usize];
                out_nodes.push(node);
                push_const_frame(&mut stack, out_nodes.len() - 1, v);
            }
            PNode::Op { arity, op } => {
                let a = arity as usize;
                debug_assert!(a <= D);
                let stack_len = stack.len();
                if stack_len < a {
                    // Invalid postfix; keep as-is.
                    out_nodes.push(node);
                    push_nonconst_frame(&mut stack, out_nodes.len() - 1);
                    continue;
                }

                let child_start = stack[stack_len - a].start;
                let children: Vec<Frame<T>> = stack.drain(stack_len - a..).collect();
                let all_const = children.iter().all(|f| f.is_const);

                if !all_const {
                    out_nodes.push(node);
                    push_nonconst_frame(&mut stack, child_start);
                    continue;
                }

                let mut vals: [T; D] = core::array::from_fn(|_| T::zero());
                for (j, f) in children.iter().enumerate() {
                    vals[j] = f.const_value;
                }

                if vals[..a].iter().any(|v| !v.is_finite()) {
                    out_nodes.push(node);
                    push_nonconst_frame(&mut stack, child_start);
                    continue;
                }

                let mut out = [T::zero()];
                let mut args: [SrcRef<'_, T>; D] =
                    core::array::from_fn(|_| SrcRef::Const(T::zero()));
                for j in 0..a {
                    args[j] = SrcRef::Const(vals[j]);
                }

                let ok = Ops::eval(
                    OpId { arity, id: op },
                    EvalKernelCtx {
                        out: &mut out,
                        args: &args[..a],
                        opts: eval_opts,
                    },
                );
                let folded = ok && out[0].is_finite();
                if !folded {
                    out_nodes.push(node);
                    push_nonconst_frame(&mut stack, child_start);
                    continue;
                }

                if out_consts.len() > u16::MAX as usize {
                    out_nodes.push(node);
                    push_nonconst_frame(&mut stack, child_start);
                    continue;
                }

                out_nodes.truncate(child_start);
                let new_idx = out_consts.len() as u16;
                out_consts.push(out[0]);
                out_nodes.push(PNode::Const { idx: new_idx });
                push_const_frame(&mut stack, child_start, out[0]);
                changed = true;
            }
        }
    }

    if changed {
        expr.nodes = out_nodes;
        expr.consts = out_consts;
    }

    changed
}

pub fn combine_operators_in_place<T, Ops, const D: usize>(expr: &mut PostfixExpr<T, Ops, D>) -> bool
where
    T: Float,
    Ops: OpRegistry,
{
    let cfg = CombineCfg {
        add_id: lookup_id_with_arity::<Ops>("+", 2),
        mul_id: lookup_id_with_arity::<Ops>("*", 2),
        sub_id: lookup_id_with_arity::<Ops>("-", 2),
    };
    combine_operators_in_place_with_cfg(expr, cfg)
}

pub fn simplify_in_place<T, Ops, const D: usize>(
    expr: &mut PostfixExpr<T, Ops, D>,
    eval_opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T> + OpRegistry,
{
    let c1 = simplify_tree_in_place(expr, eval_opts);
    let c2 = combine_operators_in_place(expr);
    let c3 = compress_constants(expr);
    c1 || c2 || c3
}
