use anyhow::{bail, Context};

pub struct Table {
    pub headers: Vec<String>,
    pub columns: Vec<Vec<f64>>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl Table {
    pub fn new(headers: Vec<String>, columns: Vec<Vec<f64>>) -> anyhow::Result<Self> {
        if headers.len() != columns.len() {
            bail!(
                "header count {} does not match column count {}",
                headers.len(),
                columns.len()
            );
        }
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        for (i, c) in columns.iter().enumerate() {
            if c.len() != n_rows {
                let col_name = headers.get(i).cloned().unwrap_or_else(|| format!("col{i}"));
                bail!(
                    "column {} has {} rows but expected {}",
                    col_name,
                    c.len(),
                    n_rows
                );
            }
        }
        Ok(Self {
            headers,
            columns,
            n_rows,
            n_cols,
        })
    }

    pub fn column_by_index(&self, idx: usize) -> anyhow::Result<&[f64]> {
        self.columns
            .get(idx)
            .map(|c| c.as_slice())
            .with_context(|| format!("column index {idx} out of bounds (n_cols={})", self.n_cols))
    }

    pub fn column_index(&self, selector: &super::columns::ColumnSelector) -> anyhow::Result<usize> {
        match selector {
            super::columns::ColumnSelector::Index(i) => {
                if *i >= self.n_cols {
                    bail!("column index {i} out of bounds (n_cols={})", self.n_cols);
                }
                Ok(*i)
            }
            super::columns::ColumnSelector::Name(name) => {
                if let Some((i, _)) = self.headers.iter().enumerate().find(|(_, h)| h == &name) {
                    return Ok(i);
                }

                let name_lc = name.to_ascii_lowercase();
                let mut matches: Vec<usize> = self
                    .headers
                    .iter()
                    .enumerate()
                    .filter(|(_, h)| h.to_ascii_lowercase() == name_lc)
                    .map(|(i, _)| i)
                    .collect();

                match matches.len() {
                    0 => bail!("unknown column name {name:?}"),
                    1 => Ok(matches.pop().unwrap()),
                    _ => {
                        bail!("ambiguous column name {name:?} (multiple case-insensitive matches)")
                    }
                }
            }
        }
    }
}
