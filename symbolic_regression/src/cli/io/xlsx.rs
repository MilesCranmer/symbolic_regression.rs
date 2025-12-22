use super::Table;
use anyhow::{bail, Context};
use calamine::{open_workbook_auto, Data, Reader};
use std::path::Path;

pub fn load_xlsx(path: &Path, sheet: Option<&str>, has_header: bool) -> anyhow::Result<Table> {
    let mut wb =
        open_workbook_auto(path).with_context(|| format!("failed to open {}", path.display()))?;

    let sheet_name = match sheet {
        Some(s) => s.to_string(),
        None => wb
            .sheet_names()
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("workbook has no sheets"))?,
    };

    let range = wb
        .worksheet_range(&sheet_name)
        .with_context(|| format!("failed to read sheet {sheet_name:?}"))?;

    let mut rows = range.rows();
    let first = rows
        .next()
        .ok_or_else(|| anyhow::anyhow!("sheet {sheet_name:?} is empty"))?;

    let n_cols = first.len();
    if n_cols == 0 {
        bail!("sheet {sheet_name:?} has zero columns");
    }

    let (headers, mut data_rows): (Vec<String>, Vec<Vec<Data>>) = if has_header {
        let mut headers: Vec<String> = Vec::with_capacity(n_cols);
        for (i, cell) in first.iter().enumerate() {
            let h = match cell {
                Data::String(s) if !s.trim().is_empty() => s.trim().to_string(),
                Data::Float(f) => f.to_string(),
                Data::Int(i) => i.to_string(),
                Data::Bool(b) => b.to_string(),
                _ => format!("col{i}"),
            };
            headers.push(h);
        }
        let data_rows: Vec<Vec<Data>> = rows.map(|r: &[Data]| r.to_vec()).collect();
        (headers, data_rows)
    } else {
        let data_rows: Vec<Vec<Data>> = std::iter::once(first.to_vec())
            .chain(rows.map(|r: &[Data]| r.to_vec()))
            .collect();
        let headers = (0..n_cols).map(|i| format!("col{i}")).collect();
        (headers, data_rows)
    };

    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_cols];
    for (row_idx, row) in data_rows.drain(..).enumerate() {
        if row.len() != n_cols {
            bail!(
                "ragged XLSX row {}: expected {} columns but got {}",
                row_idx + 1,
                n_cols,
                row.len()
            );
        }
        for (col_idx, cell) in row.into_iter().enumerate() {
            let v = cell_to_f64(cell).with_context(|| {
                format!(
                    "failed to parse numeric cell at row {}, col {}",
                    row_idx + 1,
                    col_idx
                )
            })?;
            columns[col_idx].push(v);
        }
    }

    Table::new(headers, columns)
}

fn cell_to_f64(cell: Data) -> anyhow::Result<f64> {
    match cell {
        Data::Float(f) => Ok(f),
        Data::Int(i) => Ok(i as f64),
        Data::String(s) => s
            .trim()
            .parse::<f64>()
            .with_context(|| format!("raw={s:?}")),
        Data::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Data::Empty => bail!("empty cell"),
        other => bail!("unsupported cell type {other:?}"),
    }
}
