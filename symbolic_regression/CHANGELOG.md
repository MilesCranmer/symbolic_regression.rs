# Changelog

## [0.9.0](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression-v0.8.0...symbolic_regression-v0.9.0) (2025-12-25)


### ⚠ BREAKING CHANGES

* various improvements for speed

### Features

* faster gradients ([a87bd12](https://github.com/astroautomata/symbolic_regression.rs/commit/a87bd12486c5324e7f20b66df4b6fec77a91422a))
* mutation optimization ([491b23c](https://github.com/astroautomata/symbolic_regression.rs/commit/491b23cc2fb49da29b1b64de587cca3497ebd9ec))
* rayon parallelism for both local and web ([e3a1831](https://github.com/astroautomata/symbolic_regression.rs/commit/e3a1831d5b4758365630a0af0140c9316650024d))
* rayon parallelism for both local and web ([987915a](https://github.com/astroautomata/symbolic_regression.rs/commit/987915a6472823db03449f3b4fdd1802f237679d))
* various improvements for speed ([7157a83](https://github.com/astroautomata/symbolic_regression.rs/commit/7157a835b589eeddda28abf7c49b57c23706704a))


### Bug Fixes

* additional similarities to SR.jl ([1ebb838](https://github.com/astroautomata/symbolic_regression.rs/commit/1ebb838e4701853372d49b17a0217157f0cf3717))
* correct baseline loss ([bccc349](https://github.com/astroautomata/symbolic_regression.rs/commit/bccc349cf83669564b1cf69d019f71123f4379a3))
* early-exit outputs, wasm search state ([600281f](https://github.com/astroautomata/symbolic_regression.rs/commit/600281f97b547c6b62fa5d38b3914bda8e57c06d))
* match SR.jl migration logic ([efb3c96](https://github.com/astroautomata/symbolic_regression.rs/commit/efb3c961d0366889290fcbb178d2ab9a6a1b2c76))
* range clamping of pivot pos ([53bef2c](https://github.com/astroautomata/symbolic_regression.rs/commit/53bef2c552d9867f295f07c2ce63e8c9859170e3))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * dynamic_expressions bumped from 0.7.0 to 0.8.0

## [0.8.0](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression-v0.7.0...symbolic_regression-v0.8.0) (2025-12-19)


### ⚠ BREAKING CHANGES

* change operator construction syntax

### Features

* change operator construction syntax ([d8b39b9](https://github.com/astroautomata/symbolic_regression.rs/commit/d8b39b98015e3c9df592435aadc6793d328e04a3))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * dynamic_expressions bumped from 0.6.0 to 0.7.0

## [0.7.0](https://github.com/astroautomata/symbolic_regression.rs/compare/symbolic_regression-v0.6.0...symbolic_regression-v0.7.0) (2025-12-19)


### ⚠ BREAKING CHANGES

* corrected tree rotation for n-arity nodes

### Features

* closer to original SR.jl algorithm ([b64b6a1](https://github.com/astroautomata/symbolic_regression.rs/commit/b64b6a1b84284b979dcc0cdfe9072a880a16913d))
* switch to u16 for custom complexities ([8e6d78c](https://github.com/astroautomata/symbolic_regression.rs/commit/8e6d78ccf97b7841d5b5ae39bd24fe42d510fd90))


### Bug Fixes

* corrected tree rotation for n-arity nodes ([6cab7ee](https://github.com/astroautomata/symbolic_regression.rs/commit/6cab7ee7b9b535348c6ac1a59ae9e1ee7bc2ed9b))

## [0.6.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression-v0.5.0...symbolic_regression-v0.6.0) (2025-12-17)


### ⚠ BREAKING CHANGES

* string format x{} is now 0-indexed

### Bug Fixes

* clippy warnings ([2e98667](https://github.com/astro-automata/symbolic_regression.rs/commit/2e98667b914877485e76813112eebea08fd99921))
* clippy warnings ([dc189d5](https://github.com/astro-automata/symbolic_regression.rs/commit/dc189d5261a8f31e442b09bbe75a95714d0dcefa))
* string format x{} is now 0-indexed ([9001971](https://github.com/astro-automata/symbolic_regression.rs/commit/9001971498bd5796a5b10c2f03e27f52c5535409))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * dynamic_expressions bumped from 0.5.0 to 0.6.0

## [0.5.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression-v0.4.0...symbolic_regression-v0.5.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* set defaults back to SR.jl standard
* change symbolic regression structure to match julia layout
* make dynamic_expressions have closer structure to DynamicExpressions.jl

### Features

* better alignment to SymbolicRegression.jl cost/loss difference ([0b978ec](https://github.com/astro-automata/symbolic_regression.rs/commit/0b978ecfd99b304e857d1654bac46f910c484e5d))
* set defaults back to SR.jl standard ([4926893](https://github.com/astro-automata/symbolic_regression.rs/commit/4926893b1fe0fb43f2baad1ff1fe5f2952ec1791))


### Code Refactoring

* change symbolic regression structure to match julia layout ([0c0e4d7](https://github.com/astro-automata/symbolic_regression.rs/commit/0c0e4d7ba9f93bdc13c56cd403498a477a6010a0))
* make dynamic_expressions have closer structure to DynamicExpressions.jl ([80e3b31](https://github.com/astro-automata/symbolic_regression.rs/commit/80e3b314f9fc6ee61e8ab19fadda70ff61af689c))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * dynamic_expressions bumped from 0.4.0 to 0.5.0

## [0.4.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression-v0.3.0...symbolic_regression-v0.4.0) (2025-12-16)


### ⚠ BREAKING CHANGES

* move `compress_constants` to de

### Features

* add simplification utilities ([06f738c](https://github.com/astro-automata/symbolic_regression.rs/commit/06f738cece71f0a3923f3f2e520a8d418182ac17))
* additional loss functions ([78e3da2](https://github.com/astro-automata/symbolic_regression.rs/commit/78e3da29110fc80f3b5caf72c5fca24ab3f94587))
* batched dataset ([5483597](https://github.com/astro-automata/symbolic_regression.rs/commit/5483597ab8f4b5c8191028c638335a1d8499e402))
* custom complexities ([c0f5b31](https://github.com/astro-automata/symbolic_regression.rs/commit/c0f5b31d51603d053fe994709122df5bcf3dc5ec))
* move `compress_constants` to de ([2031494](https://github.com/astro-automata/symbolic_regression.rs/commit/203149490cb786b9fb95a6b8fb7c73b51bfc1c6a))
* simplify by default ([8dd6332](https://github.com/astro-automata/symbolic_regression.rs/commit/8dd6332e4cc36d71006747bb1233852ef4704f9d))
* wasm compatibility ([67c071f](https://github.com/astro-automata/symbolic_regression.rs/commit/67c071f5d8cbbdb3760e2293a42019e237a95290))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * dynamic_expressions bumped from 0.3.0 to 0.4.0

## [0.3.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression-v0.2.0...symbolic_regression-v0.3.0) (2025-12-15)


### ⚠ BREAKING CHANGES

* upgrade core deps
* add command line interface

### deps

* upgrade core deps ([3b57512](https://github.com/astro-automata/symbolic_regression.rs/commit/3b57512637955c6fab0e5fe2c65b8f93455d94b1))


### Features

* add command line interface ([77f4b5e](https://github.com/astro-automata/symbolic_regression.rs/commit/77f4b5eda8d072f89915db6b5f0eddcce9138538))
* more ergonomic CLI ([8475acd](https://github.com/astro-automata/symbolic_regression.rs/commit/8475acdda6059db82b08a2a20054882b2721f32f))
* more robust BFGS ([258d599](https://github.com/astro-automata/symbolic_regression.rs/commit/258d599fb5741205eafe5742598442c536960055))
* use operator registry from sr ([52ff642](https://github.com/astro-automata/symbolic_regression.rs/commit/52ff6425ab7ff90b071083e44105533455a90d7d))

## [0.2.0](https://github.com/astro-automata/symbolic_regression.rs/compare/symbolic_regression-v0.1.0...symbolic_regression-v0.2.0) (2025-12-14)


### ⚠ BREAKING CHANGES

* DRY principles and better hierarchy

### Features

* create dynamic_expressions.rs and symbolic_regression.rs ([de3803a](https://github.com/astro-automata/symbolic_regression.rs/commit/de3803a65eb5b8b7b1892b0ba299ae92b07de98a))


### Code Refactoring

* DRY principles and better hierarchy ([fcce381](https://github.com/astro-automata/symbolic_regression.rs/commit/fcce3812e00acb0f55309c878d8f03b9b1064088))
