use crate::compile::EvalPlan;
use num_traits::Float;

#[derive(Copy, Clone, Debug)]
pub struct EvalOptions {
    pub check_finite: bool,
    pub early_exit: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            check_finite: true,
            early_exit: true,
        }
    }
}

#[derive(Debug)]
pub struct EvalContext<T: Float, const D: usize> {
    pub scratch: Vec<Vec<T>>, // slot-major, each len n_rows
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> EvalContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            scratch: Vec::new(),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize) {
        if self.scratch.len() < n_slots {
            self.scratch.resize_with(n_slots, Vec::new);
        }
        for slot in &mut self.scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
    }
}

#[derive(Debug)]
pub struct DiffContext<T: Float, const D: usize> {
    pub val_scratch: Vec<Vec<T>>,
    pub der_scratch: Vec<Vec<T>>,
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> DiffContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            val_scratch: Vec::new(),
            der_scratch: Vec::new(),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize) {
        if self.val_scratch.len() < n_slots {
            self.val_scratch.resize_with(n_slots, Vec::new);
        }
        if self.der_scratch.len() < n_slots {
            self.der_scratch.resize_with(n_slots, Vec::new);
        }
        for slot in &mut self.val_scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
        for slot in &mut self.der_scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
    }
}

#[derive(Debug)]
pub struct GradContext<T: Float, const D: usize> {
    pub val_scratch: Vec<Vec<T>>,
    pub grad_scratch: Vec<Vec<T>>, // slot-major, each len n_dir*n_rows
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> GradContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            val_scratch: Vec::new(),
            grad_scratch: Vec::new(),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize, n_dir: usize) {
        if self.val_scratch.len() < n_slots {
            self.val_scratch.resize_with(n_slots, Vec::new);
        }
        if self.grad_scratch.len() < n_slots {
            self.grad_scratch.resize_with(n_slots, Vec::new);
        }
        for slot in &mut self.val_scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
        let grad_len = n_dir * self.n_rows;
        for slot in &mut self.grad_scratch[..n_slots] {
            if slot.len() != grad_len {
                slot.resize(grad_len, T::zero());
            }
        }
    }
}
