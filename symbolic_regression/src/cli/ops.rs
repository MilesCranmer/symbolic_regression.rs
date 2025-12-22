// CLI operator list / parsing helpers.

use anyhow::{Context, bail};
use dynamic_expressions::operator_registry::OpRegistry;

use crate::Operators;
use crate::cli::args::Cli;

pub fn build_operators<OpsReg, const D: usize>(cli: &Cli) -> anyhow::Result<Operators<D>>
where
    OpsReg: OpRegistry,
{
    let unary: Vec<&str> = cli
        .unary_operators
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let binary_default = vec!["+", "-", "*", "/"];
    let binary_vec: Vec<&str> = match &cli.binary_operators {
        None => binary_default,
        Some(v) => v.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect(),
    };
    let ternary: Vec<&str> = cli
        .ternary_operators
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    let ops = Operators::<D>::from_names_by_arity::<OpsReg>(&unary, &binary_vec, &ternary)
        .context("failed to build operator set")?;
    if ops.total_ops_up_to(D) == 0 {
        bail!("no operators enabled (use --binary-operators/--unary-operators/--ternary-operators)");
    }
    Ok(ops)
}

pub fn print_operator_list<OpsReg: OpRegistry>() {
    let mut by_arity: [Vec<&dynamic_expressions::operator_registry::OpInfo>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for info in OpsReg::registry() {
        if (1u8..=3u8).contains(&info.op.arity) {
            by_arity[(info.op.arity - 1) as usize].push(info);
        }
    }

    let headings = ["--unary-operators", "--binary-operators", "--ternary-operators"];
    for (arity, items) in by_arity.iter_mut().enumerate() {
        items.sort_by_key(|i| i.name);
        println!("{}:", headings[arity]);
        for i in items.iter() {
            if let Some(infix) = i.infix {
                println!("  {:<10} display={:<4} infix={}", i.name, i.display, infix);
            } else {
                println!("  {:<10} display={}", i.name, i.display);
            }
        }
        println!();
    }
}
