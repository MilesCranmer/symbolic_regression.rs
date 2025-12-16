use dynamic_expressions::operators::scalar::OpId;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct OpConstraints<const D: usize> {
    /// Per-operator, per-argument complexity limits.
    /// A value of `-1` means no constraint for that argument.
    pub limits: HashMap<OpId, [i32; D]>,
}

impl<const D: usize> Default for OpConstraints<D> {
    fn default() -> Self {
        Self {
            limits: HashMap::new(),
        }
    }
}

impl<const D: usize> OpConstraints<D> {
    pub fn set_op_arg_constraint(&mut self, op: OpId, arg_idx: usize, max_complexity: i32) {
        assert!(arg_idx < D);
        let entry = self.limits.entry(op).or_insert_with(|| [-1; D]);
        entry[arg_idx] = max_complexity;
    }
}

#[derive(Clone, Debug, Default)]
pub struct NestedConstraints {
    /// Root operator -> list of (nested operator, max nestedness).
    pub limits: HashMap<OpId, Vec<(OpId, u8)>>,
}

impl NestedConstraints {
    pub fn add_nested_constraint(&mut self, root: OpId, nested: OpId, max_nestedness: u8) {
        self.limits
            .entry(root)
            .or_default()
            .push((nested, max_nestedness));
    }
}
