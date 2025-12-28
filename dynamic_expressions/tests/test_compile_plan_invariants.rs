use dynamic_expressions::{PNode, compile_plan};

#[test]
#[should_panic(expected = "Var index out of bounds")]
fn compile_plan_panics_if_var_index_out_of_bounds() {
    let nodes = vec![PNode::Var { feature: 2 }];
    let _ = compile_plan::<3>(&nodes, 2, 0);
}

#[test]
#[should_panic(expected = "Const index out of bounds")]
fn compile_plan_panics_if_const_index_out_of_bounds() {
    let nodes = vec![PNode::Const { idx: 1 }];
    let _ = compile_plan::<3>(&nodes, 0, 1);
}

#[test]
#[should_panic(expected = "stack underflow (op)")]
fn compile_plan_panics_on_stack_underflow() {
    let nodes = vec![PNode::Op { arity: 2, op: 0 }];
    let _ = compile_plan::<3>(&nodes, 0, 0);
}

#[test]
#[should_panic(expected = "Postfix did not reduce to a single root")]
fn compile_plan_panics_if_not_single_root() {
    let nodes = vec![PNode::Var { feature: 0 }, PNode::Var { feature: 1 }];
    let _ = compile_plan::<3>(&nodes, 2, 0);
}

#[test]
#[should_panic(expected = "n_features=")]
fn compile_plan_panics_if_n_features_exceeds_u16() {
    let nodes = vec![PNode::Var { feature: 0 }];
    let _ = compile_plan::<3>(&nodes, (u16::MAX as usize) + 1, 0);
}

#[test]
#[should_panic(expected = "n_consts=")]
fn compile_plan_panics_if_n_consts_exceeds_u16() {
    let nodes = vec![PNode::Var { feature: 0 }];
    let _ = compile_plan::<3>(&nodes, 1, (u16::MAX as usize) + 1);
}
