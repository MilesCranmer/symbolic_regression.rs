# Changelog

## [0.10.0](https://github.com/astroautomata/symbolic_regression.rs/compare/dynamic_expressions-v0.9.1...dynamic_expressions-v0.10.0) (2025-12-30)


### ⚠ BREAKING CHANGES

* simple and minimal operator syntax
* nicer syntax for operator definitions

### Features

* remove unused member id's ([be2fa95](https://github.com/astroautomata/symbolic_regression.rs/commit/be2fa95fba4f3a07e0a4f763cfae539824debede))
* simple and minimal operator syntax ([b8f3f4f](https://github.com/astroautomata/symbolic_regression.rs/commit/b8f3f4f27e9ed1fdf1371ab59af1015d00b7ea5d))


### Performance Improvements

* better vectorization for finite checks ([813f73a](https://github.com/astroautomata/symbolic_regression.rs/commit/813f73aead3514ee5edf4a3de891b02f1ef5731c))
* better vectorization for finite checks ([233af1c](https://github.com/astroautomata/symbolic_regression.rs/commit/233af1c355bec882ae0c241368e62acce5b9ecc1))


### Code Refactoring

* nicer syntax for operator definitions ([e51cabb](https://github.com/astroautomata/symbolic_regression.rs/commit/e51cabbadba355f1180a4beb6d853492874ff67e))

## [0.9.1](https://github.com/astroautomata/symbolic_regression.rs/compare/dynamic_expressions-v0.9.0...dynamic_expressions-v0.9.1) (2025-12-27)


### Bug Fixes

* infix-inside-infix strings ([63ec801](https://github.com/astroautomata/symbolic_regression.rs/commit/63ec801adb0a8e5e72c26c017c0a28345eef282e))

## [0.9.0](https://github.com/astroautomata/symbolic_regression.rs/compare/dynamic_expressions-v0.8.0...dynamic_expressions-v0.9.0) (2025-12-26)


### ⚠ BREAKING CHANGES

* switch to a faster RNG
* greatly simplify operator traits

### Features

* switch to a faster RNG ([1ba12f3](https://github.com/astroautomata/symbolic_regression.rs/commit/1ba12f37a741dd8566dbd08ce1b9172a5e184666))


### Performance Improvements

* skip depth check if maxsize faster ([6023fd7](https://github.com/astroautomata/symbolic_regression.rs/commit/6023fd717ce6f0da380574ff6efa67300cb33752))
* thread local stack for tree mapreduce ([4cbcad6](https://github.com/astroautomata/symbolic_regression.rs/commit/4cbcad6e9c7cea0b168fd15f6ebfa5841e60c6d5))


### Code Refactoring

* greatly simplify operator traits ([a4baa8e](https://github.com/astroautomata/symbolic_regression.rs/commit/a4baa8e1d2d9eca68406564c97427ee248ee2d4a))

## [0.8.0](https://github.com/astroautomata/symbolic_regression.rs/compare/dynamic_expressions-v0.7.0...dynamic_expressions-v0.8.0) (2025-12-25)


### ⚠ BREAKING CHANGES

* various improvements for speed

### Features

* create proptest utils ([60495db](https://github.com/astroautomata/symbolic_regression.rs/commit/60495db79f815d196c21b7fb772f426a70580cef))
* faster gradients ([a87bd12](https://github.com/astroautomata/symbolic_regression.rs/commit/a87bd12486c5324e7f20b66df4b6fec77a91422a))
* internal safe zip function ([d7d9c0d](https://github.com/astroautomata/symbolic_regression.rs/commit/d7d9c0d357952b34cd1cd5dd5b6b0fc8b1375db8))
* rayon parallelism for both local and web ([e3a1831](https://github.com/astroautomata/symbolic_regression.rs/commit/e3a1831d5b4758365630a0af0140c9316650024d))
* rayon parallelism for both local and web ([987915a](https://github.com/astroautomata/symbolic_regression.rs/commit/987915a6472823db03449f3b4fdd1802f237679d))
* various improvements for speed ([7157a83](https://github.com/astroautomata/symbolic_regression.rs/commit/7157a835b589eeddda28abf7c49b57c23706704a))
* zero init method ([e56f516](https://github.com/astroautomata/symbolic_regression.rs/commit/e56f516d4c784ae6794afe882d0740de16c9d9a5))


### Bug Fixes

* early-exit outputs, wasm search state ([600281f](https://github.com/astroautomata/symbolic_regression.rs/commit/600281f97b547c6b62fa5d38b3914bda8e57c06d))
* ensure function specialization ([0abf595](https://github.com/astroautomata/symbolic_regression.rs/commit/0abf5954384bb4bbfbfa482092db9da820172f96))
* recompile upon different hash ([2f6915d](https://github.com/astroautomata/symbolic_regression.rs/commit/2f6915dad75e481e0488831a8b5e6785640b4d19))

## [0.7.0](https://github.com/astroautomata/symbolic_regression.rs/compare/dynamic_expressions-v0.6.0...dynamic_expressions-v0.7.0) (2025-12-19)


### ⚠ BREAKING CHANGES

* change operator construction syntax

### Features

* change operator construction syntax ([d8b39b9](https://github.com/astroautomata/symbolic_regression.rs/commit/d8b39b98015e3c9df592435aadc6793d328e04a3))

## [0.6.0](https://github.com/astro-automata/symbolic_regression.rs/compare/dynamic_expressions-v0.5.0...dynamic_expressions-v0.6.0) (2025-12-17)


### ⚠ BREAKING CHANGES

* string format x{} is now 0-indexed

### Bug Fixes

* string format x{} is now 0-indexed ([9001971](https://github.com/astro-automata/symbolic_regression.rs/commit/9001971498bd5796a5b10c2f03e27f52c5535409))

## [0.5.0](https://github.com/astro-automata/symbolic_regression.rs/compare/dynamic_expressions-v0.4.0...dynamic_expressions-v0.5.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* make dynamic_expressions have closer structure to DynamicExpressions.jl

### Code Refactoring

* make dynamic_expressions have closer structure to DynamicExpressions.jl ([80e3b31](https://github.com/astro-automata/symbolic_regression.rs/commit/80e3b314f9fc6ee61e8ab19fadda70ff61af689c))

## [0.4.0](https://github.com/astro-automata/symbolic_regression.rs/compare/dynamic_expressions-v0.3.0...dynamic_expressions-v0.4.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* move `compress_constants` to de

### Features

* add simplification utilities ([06f738c](https://github.com/astro-automata/symbolic_regression.rs/commit/06f738cece71f0a3923f3f2e520a8d418182ac17))
* move `compress_constants` to de ([2031494](https://github.com/astro-automata/symbolic_regression.rs/commit/203149490cb786b9fb95a6b8fb7c73b51bfc1c6a))

## [0.3.0](https://github.com/astro-automata/symbolic_regression.rs/compare/dynamic_expressions-v0.2.0...dynamic_expressions-v0.3.0) (2025-12-15)


### ⚠ BREAKING CHANGES

* upgrade core deps

### deps

* upgrade core deps ([3b57512](https://github.com/astro-automata/symbolic_regression.rs/commit/3b57512637955c6fab0e5fe2c65b8f93455d94b1))


### Features

* create operator registry ([8fce02b](https://github.com/astro-automata/symbolic_regression.rs/commit/8fce02bd948fbadf1c82100b125f1e9eed8ad03d))

## [0.2.0](https://github.com/astro-automata/symbolic_regression.rs/compare/dynamic_expressions-v0.1.0...dynamic_expressions-v0.2.0) (2025-12-14)


### ⚠ BREAKING CHANGES

* DRY principles and better hierarchy

### Features

* create dynamic_expressions.rs and symbolic_regression.rs ([de3803a](https://github.com/astro-automata/symbolic_regression.rs/commit/de3803a65eb5b8b7b1892b0ba299ae92b07de98a))


### Code Refactoring

* DRY principles and better hierarchy ([fcce381](https://github.com/astro-automata/symbolic_regression.rs/commit/fcce3812e00acb0f55309c878d8f03b9b1064088))
