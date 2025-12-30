# Changelog

## [0.7.3](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.7.2...symbolic_regression_wasm-v0.7.3) (2025-12-30)


### Features

* remove unused member id's ([be2fa95](https://github.com/astroautomata/symbolic_regression.rs/commit/be2fa95fba4f3a07e0a4f763cfae539824debede))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.12.0 to 0.12.1
    * dynamic_expressions bumped from 0.9.1 to 0.10.0

## [0.7.2](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.7.1...symbolic_regression_wasm-v0.7.2) (2025-12-28)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.11.0 to 0.12.0

## [0.7.1](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.7.0...symbolic_regression_wasm-v0.7.1) (2025-12-27)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.10.0 to 0.11.0
    * dynamic_expressions bumped from 0.9.0 to 0.9.1

## [0.7.0](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.6.0...symbolic_regression_wasm-v0.7.0) (2025-12-26)


### ⚠ BREAKING CHANGES

* switch to a faster RNG

### Features

* switch to a faster RNG ([1ba12f3](https://github.com/astroautomata/symbolic_regression.rs/commit/1ba12f37a741dd8566dbd08ce1b9172a5e184666))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.9.0 to 0.10.0
    * dynamic_expressions bumped from 0.8.0 to 0.9.0

## [0.6.0](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.5.0...symbolic_regression_wasm-v0.6.0) (2025-12-25)


### ⚠ BREAKING CHANGES

* various improvements for speed
* wasm with atomics

### Features

* faster gradients ([a87bd12](https://github.com/astroautomata/symbolic_regression.rs/commit/a87bd12486c5324e7f20b66df4b6fec77a91422a))
* make threading more robust on web ([4dc4659](https://github.com/astroautomata/symbolic_regression.rs/commit/4dc46594bcacda1972dd9e796e18c1536fa19ab0))
* rayon parallelism for both local and web ([e3a1831](https://github.com/astroautomata/symbolic_regression.rs/commit/e3a1831d5b4758365630a0af0140c9316650024d))
* rayon parallelism for both local and web ([987915a](https://github.com/astroautomata/symbolic_regression.rs/commit/987915a6472823db03449f3b4fdd1802f237679d))
* various improvements for speed ([7157a83](https://github.com/astroautomata/symbolic_regression.rs/commit/7157a835b589eeddda28abf7c49b57c23706704a))


### Bug Fixes

* wasm rust flags ([b533e74](https://github.com/astroautomata/symbolic_regression.rs/commit/b533e7438e0793bdb4b9bcaec0d6d1fbc05e31da))


### Build System

* wasm with atomics ([8495251](https://github.com/astroautomata/symbolic_regression.rs/commit/84952515ce1f684d78c7a9f31484688723a930e8))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.8.0 to 0.9.0
    * dynamic_expressions bumped from 0.7.0 to 0.8.0

## [0.5.0](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.4.1...symbolic_regression_wasm-v0.5.0) (2025-12-19)


### ⚠ BREAKING CHANGES

* change operator construction syntax

### Features

* change operator construction syntax ([d8b39b9](https://github.com/astroautomata/symbolic_regression.rs/commit/d8b39b98015e3c9df592435aadc6793d328e04a3))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.7.0 to 0.8.0
    * dynamic_expressions bumped from 0.6.0 to 0.7.0

## [0.4.1](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.4.0...symbolic_regression_wasm-v0.4.1) (2025-12-19)


### Features

* switch to u16 for custom complexities ([8e6d78c](https://github.com/astroautomata/symbolic_regression.rs/commit/8e6d78ccf97b7841d5b5ae39bd24fe42d510fd90))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.6.0 to 0.7.0

## [0.4.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.3.0...symbolic_regression_wasm-v0.4.0) (2025-12-17)


### ⚠ BREAKING CHANGES

* string format x{} is now 0-indexed

### Bug Fixes

* string format x{} is now 0-indexed ([9001971](https://github.com/astro-automata/symbolic_regression.rs/commit/9001971498bd5796a5b10c2f03e27f52c5535409))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.5.0 to 0.6.0
    * dynamic_expressions bumped from 0.5.0 to 0.6.0

## [0.3.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.2.0...symbolic_regression_wasm-v0.3.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* make dynamic_expressions have closer structure to DynamicExpressions.jl

### Code Refactoring

* make dynamic_expressions have closer structure to DynamicExpressions.jl ([80e3b31](https://github.com/astro-automata/symbolic_regression.rs/commit/80e3b314f9fc6ee61e8ab19fadda70ff61af689c))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.4.0 to 0.5.0
    * dynamic_expressions bumped from 0.4.0 to 0.5.0

## [0.2.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression_wasm-v0.1.0...symbolic_regression_wasm-v0.2.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* rename `sr_wasm` -> `symbolic_regression_wasm`
* create WASM target demo

### Features

* batched dataset ([5483597](https://github.com/astro-automata/symbolic_regression.rs/commit/5483597ab8f4b5c8191028c638335a1d8499e402))
* create WASM target demo ([b41dfd2](https://github.com/astro-automata/symbolic_regression.rs/commit/b41dfd232513c4f4b5b129485ac963ab3ba55ee8))
* custom complexities ([c0f5b31](https://github.com/astro-automata/symbolic_regression.rs/commit/c0f5b31d51603d053fe994709122df5bcf3dc5ec))
* greatly improve web interface ([7eae075](https://github.com/astro-automata/symbolic_regression.rs/commit/7eae0757291f812e9a41e9ffda930e39987b90bb))
* rename `sr_wasm` -&gt; `symbolic_regression_wasm` ([5c974be](https://github.com/astro-automata/symbolic_regression.rs/commit/5c974becc819cd22e026060fafda156751c2fb7e))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * symbolic_regression bumped from 0.3.0 to 0.4.0
    * dynamic_expressions bumped from 0.3.0 to 0.4.0
