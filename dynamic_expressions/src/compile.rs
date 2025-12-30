use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::node::{PNode, Src};

#[derive(Clone, Debug)]
pub struct EvalPlan<const D: usize> {
    pub instrs: Vec<Instr<D>>,
    pub n_slots: usize,
    pub root: Src,
    pub hash: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct Instr<const D: usize> {
    pub arity: u8,
    pub op: u16,
    pub args: [Src; D],
    pub dst: u16,
}

pub fn compile_plan<const D: usize>(nodes: &[PNode], n_features: usize, n_consts: usize) -> EvalPlan<D> {
    assert!(
        n_features <= (u16::MAX as usize),
        "n_features={} exceeds u16::MAX",
        n_features
    );
    assert!(
        n_consts <= (u16::MAX as usize),
        "n_consts={} exceeds u16::MAX",
        n_consts
    );
    let n_features_u16 = n_features as u16;
    let n_consts_u16 = n_consts as u16;

    let mut stack: Vec<Src> = Vec::new();
    let mut instrs: Vec<Instr<D>> = Vec::new();

    let mut free_slots: Vec<u16> = Vec::new();
    let mut next_slot: u16 = 0;
    let mut max_slot: u16 = 0;

    let alloc_slot = |free_slots: &mut Vec<u16>, next_slot: &mut u16, max_slot: &mut u16| -> u16 {
        let s = free_slots.pop().unwrap_or_else(|| {
            let s = *next_slot;
            *next_slot += 1;
            s
        });
        *max_slot = (*max_slot).max(s + 1);
        s
    };

    for node in nodes {
        match *node {
            PNode::Var { feature } => {
                assert!(feature < n_features_u16, "Var index out of bounds");
                stack.push(Src::Var(feature));
            }
            PNode::Const { idx } => {
                assert!(idx < n_consts_u16, "Const index out of bounds");
                stack.push(Src::Const(idx));
            }
            PNode::Op { arity, op } => {
                let arity_u8 = arity;
                let arity = arity as usize;
                assert!(arity >= 1 && arity <= D, "Unsupported arity {} (D={})", arity, D);

                let mut args: [Src; D] = [Src::Const(0); D];
                for j in (0..arity).rev() {
                    args[j] = stack.pop().expect("stack underflow (op)");
                }

                let dst = alloc_slot(&mut free_slots, &mut next_slot, &mut max_slot);
                instrs.push(Instr {
                    arity: arity_u8,
                    op,
                    args,
                    dst,
                });

                for src in args.iter().take(arity) {
                    if let Src::Slot(s) = *src {
                        free_slots.push(s);
                    }
                }

                stack.push(Src::Slot(dst));
            }
        }
    }

    assert_eq!(stack.len(), 1, "Postfix did not reduce to a single root");
    let root = stack.pop().unwrap();
    let n_slots = max_slot as usize;
    let mut hasher = FxHasher::default();
    nodes.hash(&mut hasher);
    let hash = hasher.finish();
    EvalPlan {
        instrs,
        n_slots,
        root,
        hash,
    }
}
