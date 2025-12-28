#[test]
fn lookup_with_dash_is_ambiguous_without_arity() {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use dynamic_expressions::{LookupError, OperatorSet};

    let err = BuiltinOpsF64::lookup("-").unwrap_err();
    match err {
        LookupError::Ambiguous { token, candidates } => {
            assert_eq!(token, "-");
            assert!(candidates.iter().any(|c| c.eq_ignore_ascii_case("neg")));
            assert!(candidates.iter().any(|c| c.eq_ignore_ascii_case("sub")));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn lookup_with_dash_and_arity_selects_sub() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    let op = BuiltinOpsF64::lookup_with_arity("-", 2).unwrap();
    assert_eq!(op.arity, 2);
    assert!(BuiltinOpsF64::name(op).eq_ignore_ascii_case("sub"));
}

#[test]
fn lookup_with_dash_and_arity_selects_neg() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    let op = BuiltinOpsF64::lookup_with_arity("-", 1).unwrap();
    assert_eq!(op.arity, 1);
    assert!(BuiltinOpsF64::name(op).eq_ignore_ascii_case("neg"));
}

#[test]
fn name_returns_unknown_op_for_invalid_opid() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    assert_eq!(
        BuiltinOpsF64::name(dynamic_expressions::OpId { arity: 9, id: 0 }),
        "unknown_op"
    );
    assert_eq!(
        BuiltinOpsF64::name(dynamic_expressions::OpId { arity: 1, id: 999 }),
        "unknown_op"
    );
}
