mod columns;
mod csv;
mod table;
mod xlsx;

use crate::cli::args::Cli;
use anyhow::{bail, Context};
use columns::{ColumnSelector, ColumnSelectorParseOpts};
use ndarray::{Array1, Array2};
pub use table::Table;

pub fn load_table(cli: &Cli) -> anyhow::Result<Table> {
    let path = cli
        .data
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing input dataset path"))?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let has_header = !cli.no_header;

    match ext.as_str() {
        "csv" => csv::load_csv(path, has_header),
        "xlsx" => xlsx::load_xlsx(path, cli.sheet.as_deref(), has_header),
        _ => bail!("unsupported input extension {ext:?} (expected .csv or .xlsx)"),
    }
}

pub fn build_datasets(
    table: &Table,
    cli: &Cli,
) -> anyhow::Result<Vec<(String, crate::Dataset<f64>)>> {
    let parse_opts = ColumnSelectorParseOpts {
        one_indexed: cli.one_indexed,
    };

    let y_selectors = cli
        .y
        .iter()
        .map(|s| ColumnSelector::parse(s, parse_opts))
        .collect::<anyhow::Result<Vec<_>>>()
        .context("failed to parse --y selectors")?;

    let y_indices = y_selectors
        .iter()
        .map(|sel| table.column_index(sel))
        .collect::<anyhow::Result<Vec<_>>>()
        .context("failed to resolve --y selectors")?;

    let weights_index = match &cli.weights {
        None => None,
        Some(w) => {
            let sel = ColumnSelector::parse(w, parse_opts).context("failed to parse --weights")?;
            Some(
                table
                    .column_index(&sel)
                    .context("failed to resolve --weights")?,
            )
        }
    };

    let x_indices: Vec<usize> = match &cli.x {
        Some(xs) => {
            let selectors = xs
                .iter()
                .map(|s| ColumnSelector::parse(s, parse_opts))
                .collect::<anyhow::Result<Vec<_>>>()
                .context("failed to parse --x selectors")?;
            selectors
                .iter()
                .map(|sel| table.column_index(sel))
                .collect::<anyhow::Result<Vec<_>>>()
                .context("failed to resolve --x selectors")?
        }
        None => {
            let mut excluded = vec![false; table.n_cols];
            for &yi in &y_indices {
                excluded[yi] = true;
            }
            if let Some(wi) = weights_index {
                excluded[wi] = true;
            }
            (0..table.n_cols).filter(|&i| !excluded[i]).collect()
        }
    };

    if x_indices.is_empty() {
        bail!("no feature columns selected for X (use --x to specify explicitly)");
    }

    for &xi in &x_indices {
        if y_indices.contains(&xi) {
            bail!("X and y overlap at column index {xi}");
        }
        if weights_index.is_some_and(|wi| wi == xi) {
            bail!("X and weights overlap at column index {xi}");
        }
    }

    let n_rows = table.n_rows;
    let n_features = x_indices.len();
    let mut x = Array2::<f64>::zeros((n_rows, n_features));
    for (j, &col_idx) in x_indices.iter().enumerate() {
        let col = table.column_by_index(col_idx)?;
        for (i, &v) in col.iter().enumerate() {
            x[(i, j)] = v;
        }
    }

    let weights = match weights_index {
        None => None,
        Some(wi) => {
            let wcol = table.column_by_index(wi)?;
            Some(Array1::from_iter(wcol.iter().copied()))
        }
    };

    let variable_names: Vec<String> = x_indices
        .iter()
        .map(|&i| table.headers[i].clone())
        .collect();

    let mut out = Vec::new();
    for &yi in &y_indices {
        let ycol = table.column_by_index(yi)?;
        let y = Array1::from_iter(ycol.iter().copied());
        let target_name = table.headers[yi].clone();

        let ds = crate::Dataset::with_weights_and_names(
            x.clone(),
            y,
            weights.clone(),
            variable_names.clone(),
        );
        out.push((target_name, ds));
    }

    Ok(out)
}
