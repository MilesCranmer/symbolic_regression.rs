use std::mem;

use num_traits::Float;

use crate::node::PNode;

/// Cached value for a subtree evaluation.
///
/// We cache the evaluated vector (len = n_rows) and whether all values are finite.
#[derive(Clone, Debug)]
pub(crate) struct CachedSubtreeValue<T: Float> {
    pub val: Vec<T>,
    pub all_finite: bool,
}

/// A memory-bounded cache of **constant-free** subtree evaluations.
///
/// This is designed for the common inner-loop where only constants change (e.g. during
/// constant optimization). Any subtree that does **not** depend on constants can be
/// cached once per dataset and then reused across evaluations.
///
/// The cache is keyed by:
/// - `dataset_key`: user-provided identifier for the dataset contents (e.g. a version counter)
/// - `expr_sig`: a structural signature of `expr.nodes` (so any tree edit invalidates the cache)
///
/// Each cache entry corresponds to an operator node / instruction index (postfix operator order).
#[derive(Debug)]
pub struct SubtreeCache<T: Float> {
    n_rows: usize,
    max_bytes: usize,
    // Derived budget: max number of cached subtrees (each costs n_rows * sizeof(T))
    max_cached: usize,
    cached_count: usize,

    dataset_key: Option<u64>,
    expr_sig: u64,

    // Per-instruction flags (len = number of operator nodes / plan.instrs.len())
    cacheable: Vec<bool>,
    // Per-instruction cached values
    values: Vec<Option<CachedSubtreeValue<T>>>,
}

impl<T: Float> SubtreeCache<T> {
    /// Create a cache for an evaluation with `n_rows` outputs.
    ///
    /// `max_bytes` bounds total cached value memory (approximately).
    /// Set `max_bytes = 0` to disable caching.
    pub fn new(n_rows: usize, max_bytes: usize) -> Self {
        let bytes_per_entry = n_rows.saturating_mul(mem::size_of::<T>()).max(1);
        let max_cached = max_bytes / bytes_per_entry;
        Self {
            n_rows,
            max_bytes,
            max_cached,
            cached_count: 0,
            dataset_key: None,
            expr_sig: 0,
            cacheable: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Returns `true` if caching is enabled (budget allows caching at least one subtree).
    pub fn enabled(&self) -> bool {
        self.max_cached > 0
    }

    /// Clear all cached subtree values.
    pub fn clear(&mut self) {
        for v in &mut self.values {
            *v = None;
        }
        self.cached_count = 0;
    }

    /// Ensure the cache is configured for the given expression and dataset.
    ///
    /// This will automatically invalidate cached values if:
    /// - the dataset key changes
    /// - the expression structure changes
    /// - `n_rows` changes
    pub fn ensure(&mut self, expr_nodes: &[PNode], dataset_key: u64, n_rows: usize) {
        // If output length changes, recompute budget and drop cached values.
        if self.n_rows != n_rows {
            self.n_rows = n_rows;
            let bytes_per_entry = n_rows.saturating_mul(mem::size_of::<T>()).max(1);
            self.max_cached = self.max_bytes / bytes_per_entry;
            self.clear();
        }

        if self.dataset_key != Some(dataset_key) {
            self.dataset_key = Some(dataset_key);
            self.clear();
        }

        let sig = expr_nodes_signature(expr_nodes);
        if self.expr_sig != sig {
            self.expr_sig = sig;
            self.recompute_cacheable(expr_nodes);
            self.clear();
        }

        // Ensure storage length matches the number of operator nodes.
        if self.values.len() != self.cacheable.len() {
            self.values.resize_with(self.cacheable.len(), || None);
            self.clear();
        }
    }

    fn recompute_cacheable(&mut self, expr_nodes: &[PNode]) {
        // cacheable[i] corresponds to the i'th operator node (postfix operator order).
        let mut stack: Vec<bool> = Vec::with_capacity(expr_nodes.len());
        let mut cacheable: Vec<bool> = Vec::new();

        for n in expr_nodes.iter().copied() {
            match n {
                PNode::Var { .. } => stack.push(false),
                PNode::Const { .. } => stack.push(true),
                PNode::Op { arity, .. } => {
                    let a = arity as usize;
                    let mut depends_on_const = false;
                    for _ in 0..a {
                        depends_on_const |= stack.pop().expect("invalid postfix (stack underflow)");
                    }
                    // This operator subtree is cacheable if it does not depend on any constants.
                    cacheable.push(!depends_on_const);
                    stack.push(depends_on_const);
                }
            }
        }

        // If postfix is valid, stack reduces to one.
        debug_assert_eq!(stack.len(), 1, "invalid postfix");

        self.cacheable = cacheable;
        self.values.resize_with(self.cacheable.len(), || None);
        self.cached_count = 0;
    }

    pub(crate) fn get(&self, instr_index: usize) -> Option<&CachedSubtreeValue<T>> {
        if !self.enabled() {
            return None;
        }
        if instr_index >= self.cacheable.len() || !self.cacheable[instr_index] {
            return None;
        }
        self.values.get(instr_index).and_then(|v| v.as_ref())
    }

    pub(crate) fn maybe_store(&mut self, instr_index: usize, dst_buf: &[T]) {
        if !self.enabled() {
            return;
        }
        if instr_index >= self.cacheable.len() || !self.cacheable[instr_index] {
            return;
        }
        if self.cached_count >= self.max_cached {
            return;
        }
        if self.values[instr_index].is_some() {
            return;
        }

        let all_finite = dst_buf.iter().all(|v| v.is_finite());
        self.values[instr_index] = Some(CachedSubtreeValue {
            val: dst_buf.to_vec(),
            all_finite,
        });
        self.cached_count += 1;
    }
}

fn expr_nodes_signature(nodes: &[PNode]) -> u64 {
    // A small, stable 64-bit hash of the postfix structure. This does not include
    // constant *values* (only constant indices and ops).
    // FNV-1a 64-bit.
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    h ^= nodes.len() as u64;
    h = h.wrapping_mul(PRIME);
    for n in nodes.iter().copied() {
        let v: u64 = match n {
            PNode::Var { feature } => /* (0u64 << 48) | */ feature as u64,
            PNode::Const { idx } => (1u64 << 48) | (idx as u64),
            PNode::Op { arity, op } => (2u64 << 48) | ((arity as u64) << 32) | (op as u64),
        };
        h ^= v;
        h = h.wrapping_mul(PRIME);
    }
    h
}
