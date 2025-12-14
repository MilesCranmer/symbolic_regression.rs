use dynamic_expressions::operators::builtin::{Add, Div, Max, Min, Mul, Sub};
use dynamic_expressions::operators::scalar::{HasOp, OpId};
use rand::Rng;
use std::any::TypeId;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct OpSpec {
    pub op: OpId,
    pub commutative: bool,
    pub associative: bool,
    pub complexity: f32,
}

#[derive(Clone, Debug)]
pub struct Operators<const D: usize> {
    pub ops_by_arity: [Vec<OpSpec>; D],
}

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

    pub fn sample_arity<R: Rng>(&self, rng: &mut R, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        let total: usize = (1..=max_arity).map(|a| self.nops(a)).sum();
        assert!(total > 0, "no operators available up to arity={max_arity}");
        let mut r = rng.gen_range(0..total);
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
        let i = rng.gen_range(0..v.len());
        &v[i]
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
        Op: 'static,
    {
        self.nary::<1, Op>()
    }

    pub fn binary<Op>(self) -> Self
    where
        Ops: HasOp<Op, 2>,
        Op: 'static,
    {
        self.nary::<2, Op>()
    }

    pub fn nary<const A: usize, Op>(mut self) -> Self
    where
        Ops: HasOp<Op, A>,
        Op: 'static,
    {
        assert!(A >= 1 && A <= D, "arity {A} not supported for D={D}");
        let arity_u8: u8 = A
            .try_into()
            .unwrap_or_else(|_| panic!("arity {A} does not fit in u8"));

        let (commutative, associative) = if A == 2 {
            let id = TypeId::of::<Op>();
            (
                id == TypeId::of::<Add>()
                    || id == TypeId::of::<Mul>()
                    || id == TypeId::of::<Max>()
                    || id == TypeId::of::<Min>(),
                id == TypeId::of::<Add>()
                    || id == TypeId::of::<Mul>()
                    || id == TypeId::of::<Max>()
                    || id == TypeId::of::<Min>(),
            )
        } else {
            (false, false)
        };

        self.operators.push(
            A,
            OpSpec {
                op: OpId {
                    arity: arity_u8,
                    id: <Ops as HasOp<Op, A>>::ID,
                },
                commutative,
                associative,
                complexity: 1.0,
            },
        );
        self
    }
}
