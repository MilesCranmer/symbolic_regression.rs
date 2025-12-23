use dynamic_expressions::{PNode, compile_plan};

#[test]
fn tree_utils_reject_invalid_nodes() {
    // Stack underflow (operator without children) should panic.
    let underflow = vec![PNode::Op { arity: 2, op: 0 }];
    assert!(std::panic::catch_unwind(|| dynamic_expressions::count_depth(&underflow)).is_err());
    assert!(std::panic::catch_unwind(|| dynamic_expressions::subtree_sizes(&underflow)).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            dynamic_expressions::tree_mapreduce(&underflow, |_| 1usize, |_| 1usize, |p, _| p)
        })
        .is_err()
    );
}

#[test]
fn compile_plan_rejects_invalid_programs() {
    // Var out of bounds.
    let bad = vec![PNode::Var { feature: 1 }];
    assert!(std::panic::catch_unwind(|| compile_plan::<3>(&bad, 1, 0)).is_err());

    // Const out of bounds.
    let bad = vec![PNode::Const { idx: 0 }];
    assert!(std::panic::catch_unwind(|| compile_plan::<3>(&bad, 0, 0)).is_err());

    // Stack underflow (operator without children).
    let bad = vec![PNode::Op { arity: 2, op: 0 }];
    assert!(std::panic::catch_unwind(|| compile_plan::<3>(&bad, 1, 0)).is_err());

    // Arity exceeds D.
    let bad = vec![PNode::Var { feature: 0 }, PNode::Op { arity: 4, op: 0 }];
    assert!(std::panic::catch_unwind(|| compile_plan::<3>(&bad, 1, 0)).is_err());
}
