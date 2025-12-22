mod common;

use common::expr_readme_like;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};

#[test]
fn string_tree_matches_expected() {
    let ex = expr_readme_like();
    let s = string_tree(&ex, StringTreeOptions::default());
    assert_eq!(s, "x0 * cos(x1 - 3.2)");
}

#[test]
fn string_tree_uses_variable_names() {
    let mut ex = expr_readme_like();
    ex.meta.variable_names = vec!["x".into(), "y".into()];
    let s = string_tree(&ex, StringTreeOptions::default());
    assert_eq!(s, "x * cos(y - 3.2)");
}
