use super::Table;
use anyhow::{bail, Context};
use std::path::Path;

pub fn load_csv(path: &Path, has_header: bool) -> anyhow::Result<Table> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_path(path)
        .with_context(|| format!("failed to open CSV {}", path.display()))?;

    let headers: Vec<String> = if has_header {
        rdr.headers()
            .with_context(|| format!("failed to read CSV headers from {}", path.display()))?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut n_cols: Option<usize> = None;

    for (row_idx, rec) in rdr.records().enumerate() {
        let row_num = row_idx + 1;
        let rec = rec.with_context(|| format!("failed to read CSV record at row {}", row_num))?;

        if n_cols.is_none() {
            n_cols = Some(rec.len());
            let n = rec.len();
            if has_header && headers.len() != n {
                bail!(
                    "CSV header has {} columns but first row has {} columns",
                    headers.len(),
                    n
                );
            }
            columns = vec![Vec::new(); n];
        }

        let n = n_cols.unwrap();
        if rec.len() != n {
            bail!(
                "ragged CSV at row {}: expected {} fields but got {}",
                row_num,
                n,
                rec.len()
            );
        }

        for (col_idx, v) in rec.iter().enumerate() {
            let parsed: f64 = v.parse().with_context(|| {
                format!(
                    "failed to parse float at row {}, column {}: raw={v:?}",
                    row_num, col_idx
                )
            })?;
            columns[col_idx].push(parsed);
        }
    }

    let n_cols = n_cols.unwrap_or(0);
    let headers = if has_header {
        headers
    } else {
        (0..n_cols).map(|i| format!("col{i}")).collect()
    };

    Table::new(headers, columns)
        .with_context(|| format!("failed to build table from {}", path.display()))
}
