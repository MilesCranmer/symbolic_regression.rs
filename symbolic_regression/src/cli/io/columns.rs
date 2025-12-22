use anyhow::{bail, Context};

#[derive(Copy, Clone, Debug)]
pub struct ColumnSelectorParseOpts {
    pub one_indexed: bool,
}

#[derive(Clone, Debug)]
pub enum ColumnSelector {
    Name(String),
    Index(usize),
}

impl ColumnSelector {
    pub fn parse(raw: &str, opts: ColumnSelectorParseOpts) -> anyhow::Result<Self> {
        let s = raw.trim();
        if s.is_empty() {
            bail!("empty column selector");
        }
        if let Ok(mut idx) = s.parse::<usize>() {
            if opts.one_indexed {
                idx = idx
                    .checked_sub(1)
                    .with_context(|| format!("1-based column index {s} is invalid"))?;
            }
            return Ok(Self::Index(idx));
        }
        Ok(Self::Name(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_index_0_based() {
        let sel =
            ColumnSelector::parse("2", ColumnSelectorParseOpts { one_indexed: false }).unwrap();
        match sel {
            ColumnSelector::Index(i) => assert_eq!(i, 2),
            _ => panic!("expected index selector"),
        }
    }

    #[test]
    fn parses_index_1_based() {
        let sel =
            ColumnSelector::parse("3", ColumnSelectorParseOpts { one_indexed: true }).unwrap();
        match sel {
            ColumnSelector::Index(i) => assert_eq!(i, 2),
            _ => panic!("expected index selector"),
        }
    }
}
