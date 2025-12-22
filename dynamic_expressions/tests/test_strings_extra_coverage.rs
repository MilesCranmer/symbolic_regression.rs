use dynamic_expressions::strings::default_string_variable;
use std::hint::black_box;

#[test]
fn default_string_variable_uses_provided_name() {
    let names = vec!["hello".to_string()];
    let f: fn(u16, Option<&[String]>) -> String = default_string_variable;
    let f = black_box(f);
    let out = f(black_box(0), Some(black_box(names.as_slice())));
    assert_eq!(out, "hello");
}

#[test]
fn default_string_variable_falls_back_when_name_missing() {
    let names: Vec<String> = vec![];
    let out = default_string_variable(black_box(0), Some(black_box(names.as_slice())));
    assert_eq!(out, "x0");
}
