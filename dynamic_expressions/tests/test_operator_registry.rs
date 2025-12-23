#[test]
fn lookup_prefers_binary_sub_for_dash() {
    let info =
        <dynamic_expressions::operator_enum::presets::BuiltinOpsF64 as dynamic_expressions::operator_registry::OpRegistry>::lookup("-")
            .unwrap();
    assert_eq!(info.op.arity, 2);
    assert!(info.name.eq_ignore_ascii_case("sub"));
}
