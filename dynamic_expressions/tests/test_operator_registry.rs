use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_registry::OpRegistry;

#[test]
fn lookup_prefers_binary_sub_for_dash() {
    let info = <BuiltinOpsF64 as OpRegistry>::lookup("-").unwrap();
    assert_eq!(info.op.arity, 2);
    assert!(info.name.eq_ignore_ascii_case("sub"));
}
