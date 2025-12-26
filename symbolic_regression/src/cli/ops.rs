// CLI operator list / parsing helpers.

use anyhow::{Context, bail};
use dynamic_expressions::{OpId, OperatorSet};

use crate::Operators;
use crate::cli::args::Cli;

pub fn build_operators<OpsReg, const D: usize>(cli: &Cli) -> anyhow::Result<Operators<D>>
where
    OpsReg: OperatorSet,
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

pub fn print_operator_list<OpsReg: OperatorSet>() {
    let max_arity = (OpsReg::MAX_ARITY as usize).min(3);
    let mut by_arity: [Vec<OpId>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for arity in 1..=max_arity {
        let arity_u8 = arity as u8;
        for &id in OpsReg::ops_with_arity(arity_u8) {
            by_arity[arity - 1].push(OpId { arity: arity_u8, id });
        }
    }

    let headings = ["--unary-operators", "--binary-operators", "--ternary-operators"];
    for (arity, items) in by_arity.iter_mut().enumerate() {
        items.sort_by_key(|op| OpsReg::name(*op));
        println!("{}:", headings[arity]);
        for i in items.iter() {
            if let Some(infix) = OpsReg::infix(*i) {
                println!(
                    "  {:<10} display={:<4} infix={}",
                    OpsReg::name(*i),
                    OpsReg::display(*i),
                    infix
                );
            } else {
                println!("  {:<10} display={}", OpsReg::name(*i), OpsReg::display(*i));
            }
        }
        println!();
    }
}
