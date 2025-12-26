use num_traits::Float;

pub trait Operator<T: Float, const A: usize> {
    const NAME: &'static str;
    const DISPLAY: &'static str = Self::NAME;
    const INFIX: Option<&'static str> = None;
    const COMMUTATIVE: bool = false;
    const ASSOCIATIVE: bool = false;
    const COMPLEXITY: u16 = 1;

    fn eval(args: &[T; A]) -> T;
    fn partial(args: &[T; A], idx: usize) -> T;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct OpId {
    pub arity: u8,
    pub id: u16,
}

/// Each operator-set implementation generates a `&'static [OpMeta]` per arity and does
/// `META.get(op.id as usize)`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OpMeta {
    pub arity: u8,
    pub id: u16,
    pub name: &'static str,
    pub display: &'static str,
    pub infix: Option<&'static str>,
    pub commutative: bool,
    pub associative: bool,
    pub complexity: u16,
}

/// Associate a fixed arity with an operator marker type.
///
/// Rust can't (in general) infer `A` from `Tag: Operator<T, A>`, so we attach arity directly.
pub trait OpTag {
    const ARITY: u8;
}

pub trait HasOp<Tag: OpTag> {
    const ID: u16;

    #[inline]
    fn op_id() -> OpId {
        OpId {
            arity: Tag::ARITY,
            id: Self::ID,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LookupError {
    Unknown(String),
    Ambiguous {
        token: String,
        candidates: Vec<&'static str>,
    },
}

/// Operator-set abstraction.
///
/// - "What ops exist?" -> `ops_with_arity`
/// - "What are their tokens / properties?" -> `meta` (plus default helpers)
/// - "How do I eval/diff/grad?" -> `eval/diff/grad` (dispatch to kernels)
pub trait OperatorSet: Sized {
    type T: Float;

    const MAX_ARITY: u8;

    fn ops_with_arity(arity: u8) -> &'static [u16];

    /// The only required metadata entrypoint.
    fn meta(op: OpId) -> Option<&'static OpMeta>;

    // ---- Convenience defaults derived from meta() ----

    #[inline]
    fn name(op: OpId) -> &'static str {
        Self::meta(op).map(|m| m.name).unwrap_or("unknown_op")
    }

    #[inline]
    fn display(op: OpId) -> &'static str {
        Self::meta(op).map(|m| m.display).unwrap_or("unknown_op")
    }

    #[inline]
    fn infix(op: OpId) -> Option<&'static str> {
        Self::meta(op).and_then(|m| m.infix)
    }

    #[inline]
    fn commutative(op: OpId) -> bool {
        Self::meta(op).is_some_and(|m| m.commutative)
    }

    #[inline]
    fn associative(op: OpId) -> bool {
        Self::meta(op).is_some_and(|m| m.associative)
    }

    #[inline]
    fn complexity(op: OpId) -> u16 {
        Self::meta(op).map(|m| m.complexity).unwrap_or(0)
    }

    // ---- Kernel dispatch ----

    fn eval(op: OpId, ctx: crate::dispatch::EvalKernelCtx<'_, '_, Self::T>) -> bool;
    fn diff(op: OpId, ctx: crate::dispatch::DiffKernelCtx<'_, '_, Self::T>) -> bool;
    fn grad(op: OpId, ctx: crate::dispatch::GradKernelCtx<'_, '_, Self::T>) -> bool;

    // ---- Token lookup ----

    #[inline]
    fn matches_token(op: OpId, tok: &str) -> bool {
        let t = tok.trim();
        t.eq_ignore_ascii_case(Self::name(op)) || t == Self::display(op) || Self::infix(op).is_some_and(|s| t == s)
    }

    #[inline]
    fn for_each_op(mut f: impl FnMut(OpId)) {
        for arity in 1..=Self::MAX_ARITY {
            for &id in Self::ops_with_arity(arity) {
                f(OpId { arity, id });
            }
        }
    }

    fn lookup_all(token: &str) -> Vec<OpId> {
        let mut out = Vec::new();
        Self::for_each_op(|op| {
            if Self::matches_token(op, token) {
                out.push(op);
            }
        });
        out
    }

    fn lookup(token: &str) -> Result<OpId, LookupError> {
        let matches = Self::lookup_all(token);
        match matches.as_slice() {
            [] => Err(LookupError::Unknown(token.trim().to_string())),
            [single] => Ok(*single),
            _ => Err(LookupError::Ambiguous {
                token: token.trim().to_string(),
                candidates: matches.iter().map(|op| Self::name(*op)).collect(),
            }),
        }
    }

    fn lookup_with_arity(token: &str, arity: u8) -> Result<OpId, LookupError> {
        let mut matches = Vec::new();
        for &id in Self::ops_with_arity(arity) {
            let op = OpId { arity, id };
            if Self::matches_token(op, token) {
                matches.push(op);
            }
        }
        match matches.as_slice() {
            [] => Err(LookupError::Unknown(token.trim().to_string())),
            [single] => Ok(*single),
            _ => Err(LookupError::Ambiguous {
                token: token.trim().to_string(),
                candidates: matches.iter().map(|op| Self::name(*op)).collect(),
            }),
        }
    }
}
