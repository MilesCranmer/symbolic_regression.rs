use dynamic_expressions::operator_enum::builtin::OpMeta;
use dynamic_expressions::operator_enum::builtin::{Add, Div, Mul, Sub};
use dynamic_expressions::operator_enum::scalar::{HasOp, OpId};
use dynamic_expressions::operator_registry::OpRegistry;
use rand::Rng;
use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct OpSpec {
    pub op: OpId,
    pub commutative: bool,
    pub associative: bool,
    pub complexity: u16,
}

#[derive(Clone, Debug)]
pub struct Operators<const D: usize> {
    pub ops_by_arity: [Vec<OpSpec>; D],
}

/// Convenience constructors for [`Operators`] based on a registry's declared maximum arity.
pub trait OperatorRegistryExt: Sized {
    type OperatorSet;

    fn from_names(names: &[&str]) -> Result<Self::OperatorSet, OperatorSelectError>;

    fn from_names_by_arity(
        unary: &[&str],
        binary: &[&str],
        ternary: &[&str],
    ) -> Result<Self::OperatorSet, OperatorSelectError>;
}

pub const fn max_arity(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! __impl_operator_registry_ext_for {
    ($Ops:ty; $($arity:literal),+ $(,)?) => {
        $crate::__impl_operator_registry_ext_for!(@max $Ops; 0; $($arity),+);
    };

    (@max $Ops:ty; $cur:expr; $head:literal $(, $tail:literal)*) => {
        $crate::__impl_operator_registry_ext_for!(
            @max $Ops;
            { $crate::operators::max_arity($cur, $head as usize) };
            $($tail),*
        );
    };

    (@max $Ops:ty; $max:expr; ) => {
        impl $crate::OperatorRegistryExt for $Ops {
            type OperatorSet = $crate::Operators<{ $max }>;

            fn from_names(names: &[&str]) -> Result<Self::OperatorSet, $crate::OperatorSelectError> {
                $crate::Operators::<{ $max }>::from_names::<Self>(names)
            }

            fn from_names_by_arity(
                unary: &[&str],
                binary: &[&str],
                ternary: &[&str],
            ) -> Result<Self::OperatorSet, $crate::OperatorSelectError> {
                $crate::Operators::<{ $max }>::from_names_by_arity::<Self>(unary, binary, ternary)
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __sr_custom_opset_bind_idx {
    ($idx_val:ident, $idx_name:pat_param) => {
        let $idx_name = $idx_val;
    };
    ($idx_val:ident) => {};
}

#[macro_export]
macro_rules! custom_opset {
    // New DSL:
    //
    // custom_opset! {
    //     pub struct CustomOps<T = f64>;
    //
    //     1 => {
    //         square { eval: |[x]| x * x, partial: |[x]| 2.0 * x },
    //     },
    // }
    (
        $(#[$meta:meta])*
        $vis:vis struct $Ops:ident<$T:ident = $t:ty>;

        $(
            $arity:literal => {
                $(
                    $op_name:ident {
                        $(display: $display:expr,)?
                        $(infix: $infix:expr,)?
                        $(commutative: $commutative:expr,)?
                        $(associative: $associative:expr,)?
                        $(complexity: $complexity:expr,)?
                        eval: |[ $($eval_pat:pat),* $(,)? ]| $eval_body:expr,
                        partial: |[ $($partial_pat:pat),* $(,)? ] $(, $idx:pat_param)?| $partial_body:expr $(,)?
                    }
                ),* $(,)?
            }
        ),+ $(,)?
    ) => {
        $crate::__dynamic_expressions_custom_opset! {
            $(#[$meta])*
            $vis struct $Ops<$t> {
                $(
                    $arity {
                        $(
                            $op_name {
                                $(display: $display,)?
                                $(infix: $infix,)?
                                $(commutative: $commutative,)?
                                $(associative: $associative,)?
                                $(complexity: $complexity,)?
                                eval: |__sr_custom_opset_args: &[$t; $arity]| {
                                    let [ $($eval_pat),* ] = *__sr_custom_opset_args;
                                    $eval_body
                                },
                                partial: |__sr_custom_opset_args: &[$t; $arity], __sr_custom_opset_idx: usize| {
                                    let [ $($partial_pat),* ] = *__sr_custom_opset_args;
                                    $crate::__sr_custom_opset_bind_idx!(
                                        __sr_custom_opset_idx $(, $idx)?
                                    );
                                    $partial_body
                                },
                            }
                        )*
                    }
                )*
            }
        }
        $crate::__impl_operator_registry_ext_for!($Ops; $($arity),+);
    };

    // Legacy syntax (passthrough to `dynamic_expressions::custom_opset!`).
    (
        $(#[$meta:meta])* $vis:vis struct $Ops:ident<$t:ty> {
            $( $arity:literal { $( $op_name:ident { $($op_body:tt)* } )* } )*
        }
    ) => {
        $crate::__dynamic_expressions_custom_opset! {
            $(#[$meta])*
            $vis struct $Ops<$t> {
                $( $arity { $( $op_name { $($op_body)* } )* } )*
            }
        }
        $crate::__impl_operator_registry_ext_for!($Ops; $($arity),*);
    };
}

__impl_operator_registry_ext_for!(
    dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
    1, 2, 3
);
__impl_operator_registry_ext_for!(
    dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    1, 2, 3
);

#[derive(Debug, Clone)]
pub enum OperatorSelectError {
    Lookup(dynamic_expressions::operator_registry::LookupError),
    ArityMismatch {
        token: String,
        expected: u8,
        found: u8,
    },
    ArityTooLarge {
        token: String,
        arity: u8,
        max_arity: usize,
    },
    Duplicate(String),
    Empty,
}

impl fmt::Display for OperatorSelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperatorSelectError::Lookup(e) => write!(f, "{e:?}"),
            OperatorSelectError::ArityMismatch {
                token,
                expected,
                found,
            } => write!(
                f,
                "operator token {token:?} has arity={found} but was provided for arity={expected}"
            ),
            OperatorSelectError::ArityTooLarge {
                token,
                arity,
                max_arity,
            } => write!(
                f,
                "operator token {token:?} has arity={arity} which exceeds D={max_arity}"
            ),
            OperatorSelectError::Duplicate(tok) => write!(f, "duplicate operator token {tok:?}"),
            OperatorSelectError::Empty => write!(f, "no operators provided"),
        }
    }
}

impl std::error::Error for OperatorSelectError {}

impl<const D: usize> Operators<D> {
    pub fn new() -> Self {
        Self {
            ops_by_arity: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn push(&mut self, arity: usize, spec: OpSpec) {
        assert!((1..=D).contains(&arity));
        self.ops_by_arity[arity - 1].push(spec);
    }

    pub fn nops(&self, arity: usize) -> usize {
        self.ops_by_arity[arity - 1].len()
    }

    pub fn total_ops_up_to(&self, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        (1..=max_arity).map(|a| self.nops(a)).sum()
    }

    pub fn get_op_complexity(&self, op: OpId) -> Option<u16> {
        let a = op.arity as usize;
        if !(1..=D).contains(&a) {
            return None;
        }
        self.ops_by_arity[a - 1]
            .iter()
            .find(|s| s.op.id == op.id)
            .map(|s| s.complexity)
    }

    pub fn sample_arity<R: Rng>(&self, rng: &mut R, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        let total: usize = (1..=max_arity).map(|a| self.nops(a)).sum();
        assert!(total > 0, "no operators available up to arity={max_arity}");
        let mut r = rng.random_range(0..total);
        for arity in 1..=max_arity {
            let n = self.nops(arity);
            if r < n {
                return arity;
            }
            r -= n;
        }
        unreachable!()
    }

    pub fn sample_op<R: Rng>(&self, rng: &mut R, arity: usize) -> &OpSpec {
        let v = &self.ops_by_arity[arity - 1];
        let i = rng.random_range(0..v.len());
        &v[i]
    }

    pub fn from_names<Ops: OpRegistry>(names: &[&str]) -> Result<Self, OperatorSelectError> {
        if names.is_empty() {
            return Err(OperatorSelectError::Empty);
        }
        let mut out = Operators::new();
        let mut seen: HashSet<(u8, u16)> = HashSet::new();

        for &tok in names {
            let info = Ops::lookup(tok).map_err(OperatorSelectError::Lookup)?;
            if (info.op.arity as usize) > D {
                return Err(OperatorSelectError::ArityTooLarge {
                    token: tok.to_string(),
                    arity: info.op.arity,
                    max_arity: D,
                });
            }
            let key = (info.op.arity, info.op.id);
            if !seen.insert(key) {
                return Err(OperatorSelectError::Duplicate(tok.to_string()));
            }
            out.push(
                info.op.arity as usize,
                OpSpec {
                    op: info.op,
                    commutative: info.commutative,
                    associative: info.associative,
                    complexity: info.complexity,
                },
            );
        }
        Ok(out)
    }

    pub fn from_names_by_arity<Ops: OpRegistry>(
        unary: &[&str],
        binary: &[&str],
        ternary: &[&str],
    ) -> Result<Self, OperatorSelectError> {
        if unary.is_empty() && binary.is_empty() && ternary.is_empty() {
            return Err(OperatorSelectError::Empty);
        }

        let mut out = Operators::new();
        let mut seen: HashSet<(u8, u16)> = HashSet::new();

        for (expected, toks) in [(1u8, unary), (2u8, binary), (3u8, ternary)] {
            for &tok in toks {
                let info =
                    Ops::lookup_with_arity(tok, expected).map_err(OperatorSelectError::Lookup)?;
                if info.op.arity != expected {
                    return Err(OperatorSelectError::ArityMismatch {
                        token: tok.to_string(),
                        expected,
                        found: info.op.arity,
                    });
                }
                if (info.op.arity as usize) > D {
                    return Err(OperatorSelectError::ArityTooLarge {
                        token: tok.to_string(),
                        arity: info.op.arity,
                        max_arity: D,
                    });
                }
                let key = (info.op.arity, info.op.id);
                if !seen.insert(key) {
                    return Err(OperatorSelectError::Duplicate(tok.to_string()));
                }
                out.push(
                    info.op.arity as usize,
                    OpSpec {
                        op: info.op,
                        commutative: info.commutative,
                        associative: info.associative,
                        complexity: info.complexity,
                    },
                );
            }
        }

        Ok(out)
    }
}

impl<const D: usize> Default for Operators<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! sr_ops {
    ($Ops:ty, D = $D:literal; $($arity:literal => ( $($op:path),* $(,)? ) ),* $(,)?) => {{
        let mut b = $crate::operators::Operators::<$D>::builder::<$Ops>();
        $(
            $(
                b = b.nary::<$arity, $op>();
            )*
        )*
        b.build()
    }};
}

#[derive(Clone, Debug)]
pub struct OperatorsBuilder<Ops, const D: usize> {
    operators: Operators<D>,
    _ops: PhantomData<Ops>,
}

impl<const D: usize> Operators<D> {
    pub fn builder<Ops>() -> OperatorsBuilder<Ops, D> {
        OperatorsBuilder {
            operators: Operators::new(),
            _ops: PhantomData,
        }
    }
}

impl<Ops, const D: usize> OperatorsBuilder<Ops, D> {
    pub fn build(self) -> Operators<D> {
        self.operators
    }
}

impl<Ops, const D: usize> OperatorsBuilder<Ops, D> {
    pub fn sr_default_binary(self) -> Self
    where
        Ops: HasOp<Add, 2> + HasOp<Sub, 2> + HasOp<Mul, 2> + HasOp<Div, 2>,
    {
        self.nary::<2, Add>()
            .nary::<2, Sub>()
            .nary::<2, Mul>()
            .nary::<2, Div>()
    }

    pub fn unary<Op>(self) -> Self
    where
        Ops: HasOp<Op, 1>,
        Op: OpMeta<1>,
    {
        self.nary::<1, Op>()
    }

    pub fn binary<Op>(self) -> Self
    where
        Ops: HasOp<Op, 2>,
        Op: OpMeta<2>,
    {
        self.nary::<2, Op>()
    }

    pub fn nary<const A: usize, Op>(mut self) -> Self
    where
        Ops: HasOp<Op, A>,
        Op: OpMeta<A>,
    {
        assert!(A >= 1 && A <= D, "arity {A} not supported for D={D}");
        let arity_u8: u8 = A
            .try_into()
            .unwrap_or_else(|_| panic!("arity {A} does not fit in u8"));

        let commutative = <Op as OpMeta<A>>::COMMUTATIVE;
        let associative = <Op as OpMeta<A>>::ASSOCIATIVE;

        self.operators.push(
            A,
            OpSpec {
                op: OpId {
                    arity: arity_u8,
                    id: <Ops as HasOp<Op, A>>::ID,
                },
                commutative,
                associative,
                complexity: <Op as OpMeta<A>>::COMPLEXITY,
            },
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamic_expressions::operator_enum::builtin::{Neg, Sub};
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use dynamic_expressions::operator_enum::scalar::HasOp;

    crate::custom_opset! {
        struct V2Ops<T = f64>;

        1 => {
            square { eval: |[x]| x * x, partial: |[x]| 2.0 * x },
        },
    }

    #[test]
    fn from_names_by_arity_resolves_dash_by_arity() {
        let unary = ["-"];
        let ops = Operators::<3>::from_names_by_arity::<BuiltinOpsF64>(&unary, &[], &[]).unwrap();
        assert_eq!(ops.nops(1), 1);
        assert_eq!(
            ops.ops_by_arity[0][0].op.id,
            <BuiltinOpsF64 as HasOp<Neg, 1>>::ID
        );

        let binary = ["-"];
        let ops = Operators::<3>::from_names_by_arity::<BuiltinOpsF64>(&[], &binary, &[]).unwrap();
        assert_eq!(ops.nops(2), 1);
        assert_eq!(
            ops.ops_by_arity[1][0].op.id,
            <BuiltinOpsF64 as HasOp<Sub, 2>>::ID
        );
    }

    #[test]
    fn v2_custom_opset_dsl_builds_operator_set() {
        let ops = V2Ops::from_names(&["square"]).unwrap();
        assert_eq!(ops.nops(1), 1);
    }
}
