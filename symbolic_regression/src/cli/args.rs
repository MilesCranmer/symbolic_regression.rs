// CLI argument parsing.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::options::cli_args::OptionsArgs;

#[rustfmt::skip]
#[derive(Parser, Debug, Clone)]
#[command(name = "symreg")]
#[command(about = "Symbolic regression CLI (experimental)")]
pub struct Cli {
    /// Input dataset path (.csv or .xlsx).
    #[arg(required_unless_present = "list_operators")]
    pub data: Option<PathBuf>,

    /// Sheet name for .xlsx (defaults to the first sheet).
    #[arg(long)]
    pub sheet: Option<String>,

    /// Treat input as having no header row.
    #[arg(long)]
    pub no_header: bool,

    /// Interpret integer column indices as 1-based (default: 0-based).
    #[arg(long)]
    pub one_indexed: bool,

    /// Input feature columns (comma-separated). Defaults to all columns except y/weights.
    #[arg(long, value_delimiter = ',')]
    pub x: Option<Vec<String>>,

    /// Target column(s) (comma-separated). Required.
    #[arg(long, value_delimiter = ',', required_unless_present = "list_operators")]
    pub y: Vec<String>,

    /// Optional weights column (single column selector).
    #[arg(long)]
    pub weights: Option<String>,

    /// Unary operators to enable (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub unary_operators: Vec<String>,

    /// Binary operators to enable (comma-separated). If omitted, defaults to +,-,*,/
    /// Note: quote operator lists in shells (e.g. `--binary-operators='+,*'`) to avoid globbing.
    #[arg(long, value_delimiter = ',')]
    pub binary_operators: Option<Vec<String>>,

    /// Ternary operators to enable (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub ternary_operators: Vec<String>,

    /// List available builtin operators and exit.
    #[arg(long)]
    pub list_operators: bool,

    /// Output path for results (optional).
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Output format (defaults to inferring from --output extension).
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Use pretty names in printed equations (where available).
    #[arg(long)]
    pub pretty: bool,

    #[command(flatten)]
    pub options: OptionsArgs,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    Table,
    Csv,
    Json,
}
