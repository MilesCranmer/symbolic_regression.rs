// CLI output formatting.

use std::fmt::Display;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, bail};
use dynamic_expressions::strings;
use num_traits::Float;

use crate::cli::args::OutputFormat;

pub struct TargetResult<T: Float, Ops, const D: usize> {
    pub target: String,
    pub front: Vec<crate::PopMember<T, Ops, D>>,
    pub variable_names: Vec<String>,
}

pub fn print_front<T, Ops, const D: usize>(
    target: &str,
    dataset: &crate::Dataset<T>,
    front: &[crate::PopMember<T, Ops, D>],
    pretty: bool,
) where
    T: Float + Display,
    Ops: dynamic_expressions::strings::OpNames,
{
    println!("target: {target}");
    println!("{:<10} {:<14} equation", "complexity", "loss");
    for m in front {
        let eq = strings::string_tree(
            &m.expr,
            strings::StringTreeOptions {
                variable_names: Some(&dataset.variable_names),
                pretty,
            },
        );
        println!("{:<10} {:<14} {}", m.complexity, m.loss, eq);
    }
    println!();
}

pub fn write_results<T, Ops, const D: usize>(
    path: &Path,
    format: Option<OutputFormat>,
    results: &[TargetResult<T, Ops, D>],
) -> anyhow::Result<()>
where
    T: Float + Display,
    Ops: dynamic_expressions::strings::OpNames,
{
    let fmt = match format {
        Some(f) => f,
        None => infer_format(path)?,
    };

    match fmt {
        OutputFormat::Table => bail!("table output is only supported on stdout"),
        OutputFormat::Csv => write_csv(path, results),
        OutputFormat::Json => write_json(path, results),
    }
}

fn infer_format(path: &Path) -> anyhow::Result<OutputFormat> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => Ok(OutputFormat::Csv),
        "json" => Ok(OutputFormat::Json),
        _ => Ok(OutputFormat::Csv),
    }
}

fn write_csv<T, Ops, const D: usize>(path: &Path, results: &[TargetResult<T, Ops, D>]) -> anyhow::Result<()>
where
    T: Float + Display,
    Ops: dynamic_expressions::strings::OpNames,
{
    let mut wtr = csv::Writer::from_path(path).with_context(|| format!("failed to create {}", path.display()))?;
    wtr.write_record(["target", "complexity", "loss", "equation"])?;

    for r in results {
        for m in &r.front {
            let eq = strings::string_tree(
                &m.expr,
                strings::StringTreeOptions {
                    variable_names: Some(&r.variable_names),
                    pretty: false,
                },
            );
            let complexity = m.complexity.to_string();
            let loss = m.loss.to_string();
            wtr.write_record([r.target.as_str(), complexity.as_str(), loss.as_str(), eq.as_str()])?;
        }
    }
    wtr.flush()?;
    Ok(())
}

#[derive(serde::Serialize)]
struct JsonRow<'a> {
    target: &'a str,
    complexity: usize,
    loss: String,
    equation: String,
}

fn write_json<T, Ops, const D: usize>(path: &Path, results: &[TargetResult<T, Ops, D>]) -> anyhow::Result<()>
where
    T: Float + Display,
    Ops: dynamic_expressions::strings::OpNames,
{
    let mut rows = Vec::new();
    for r in results {
        for m in &r.front {
            let eq = strings::string_tree(
                &m.expr,
                strings::StringTreeOptions {
                    variable_names: Some(&r.variable_names),
                    pretty: false,
                },
            );
            rows.push(JsonRow {
                target: r.target.as_str(),
                complexity: m.complexity,
                loss: m.loss.to_string(),
                equation: eq,
            });
        }
    }
    let json = serde_json::to_string_pretty(&rows)?;
    let mut f = std::fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    f.write_all(json.as_bytes())?;
    Ok(())
}
