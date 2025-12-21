# Contributing

## Style

### Qualification rules

- Import names you *reason about locally*. Qualify names you want to *locate*.
- If a type is central in this file (appears >= 2 times, or is part of the "main story"), import it:
  - `use crate::path::Type;` then `Type::new()`.
- If a thing is used once, do not add a `use` just for it.
- If you use multiple items from a module, import the module and call through it:
  - `use std::fs; fs::read_to_string(...); fs::write(...);`
- Avoid importing free functions from std or other crates unless they are "domain verbs" used repeatedly.
  - Prefer `std::mem::replace(...)` over `use std::mem::replace; replace(...)`.
- Traits should be imported when needed for method resolution (common with `Iterator`, `Read`, `Write`, etc.).
- Macros should be imported explicitly.
- Never import constructors as bare names (no `use Type::new;`, no `use default;`).
- Prefer `use foo::{Bar, Baz};` over deep glob imports. Avoid `use foo::*;` except in tests or prelude modules.
- Keep `use` groups tidy and minimal:
  - std first, then external crates, then `crate::...`, separated by blank lines.
- If a name collision is likely (e.g., many `Error`, `Result`, `Context`), qualify the name.
- When in doubt: choose the option that makes it easiest to answer "where did this come from?" while keeping the code readable.