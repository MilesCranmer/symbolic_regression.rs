// CLI entrypoint + option wiring.

use crate::cli::args::Cli;
use crate::cli::output::{print_front, write_results, TargetResult};
use anyhow::Context;
use clap::Parser;

type T = f64;
const D: usize = 3;
type Ops = dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.list_operators {
        super::ops::print_operator_list::<Ops>();
        return Ok(());
    }

    let table = super::io::load_table(&cli).context("failed to load input table")?;
    let datasets = super::io::build_datasets(&table, &cli).context("failed to build dataset")?;

    let operators = super::ops::build_operators::<Ops, D>(&cli)?;
    let mut options = crate::Options::<T, D> {
        operators,
        ..Default::default()
    };
    cli.options.apply_to(&mut options);
    validate_options(&options)?;

    let mut results: Vec<TargetResult<T, Ops, D>> = Vec::new();
    for (target_name, dataset) in datasets {
        let res = crate::equation_search::<T, Ops, D>(&dataset, &options);
        let front = res.hall_of_fame.pareto_front();
        print_front(&target_name, &dataset, &front, cli.pretty);
        results.push(TargetResult {
            target: target_name,
            front,
            variable_names: dataset.variable_names,
        });
    }

    if let Some(path) = &cli.output {
        write_results(path, cli.format, &results)
            .with_context(|| format!("failed to write output to {}", path.display()))?;
    }

    Ok(())
}

fn validate_options(opt: &crate::Options<T, D>) -> anyhow::Result<()> {
    anyhow::ensure!(opt.population_size > 0, "population_size must be > 0");
    anyhow::ensure!(opt.populations > 0, "populations must be > 0");
    anyhow::ensure!(opt.niterations > 0, "niterations must be > 0");
    anyhow::ensure!(opt.batch_size > 0, "batch_size must be > 0");
    anyhow::ensure!(
        opt.tournament_selection_n < opt.population_size,
        "tournament_selection_n must be < population_size"
    );
    anyhow::ensure!(opt.maxsize >= 4, "maxsize must be >= 4");
    anyhow::ensure!(opt.operators.total_ops_up_to(D) > 0, "no operators enabled");
    Ok(())
}
