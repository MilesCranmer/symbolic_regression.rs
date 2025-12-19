use crate::operator_enum::scalar::OpId;

#[derive(Copy, Clone, Debug)]
pub struct OpInfo {
    pub op: OpId,
    pub name: &'static str,
    pub display: &'static str,
    pub infix: Option<&'static str>,
    pub commutative: bool,
    pub associative: bool,
    pub complexity: u16,
}

impl OpInfo {
    #[inline]
    pub fn matches_token(&self, tok: &str) -> bool {
        let t = tok.trim();
        t.eq_ignore_ascii_case(self.name) || t == self.display || self.infix.is_some_and(|s| t == s)
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

pub trait OpRegistry {
    fn registry() -> &'static [OpInfo];

    fn lookup_all(token: &str) -> Vec<&'static OpInfo> {
        Self::registry()
            .iter()
            .filter(|info| info.matches_token(token))
            .collect()
    }

    fn lookup(token: &str) -> Result<&'static OpInfo, LookupError> {
        let matches = Self::lookup_all(token);
        match matches.as_slice() {
            [] => Err(LookupError::Unknown(token.trim().to_string())),
            [single] => Ok(*single),
            _ => {
                // Common CLI ambiguity: "-" can be unary neg or binary sub. Prefer binary sub.
                let t = token.trim();
                if t == "-" {
                    if let Some(m) = matches
                        .iter()
                        .copied()
                        .find(|info| info.op.arity == 2 && info.name.eq_ignore_ascii_case("sub"))
                    {
                        return Ok(m);
                    }
                }
                Err(LookupError::Ambiguous {
                    token: t.to_string(),
                    candidates: matches.iter().map(|m| m.name).collect(),
                })
            }
        }
    }

    fn lookup_with_arity(token: &str, arity: u8) -> Result<&'static OpInfo, LookupError> {
        let matches: Vec<&'static OpInfo> = Self::registry()
            .iter()
            .filter(|info| info.op.arity == arity && info.matches_token(token))
            .collect();
        match matches.as_slice() {
            [] => Err(LookupError::Unknown(token.trim().to_string())),
            [single] => Ok(*single),
            _ => Err(LookupError::Ambiguous {
                token: token.trim().to_string(),
                candidates: matches.iter().map(|m| m.name).collect(),
            }),
        }
    }
}
