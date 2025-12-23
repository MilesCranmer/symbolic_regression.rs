use std::fmt::Display;
use std::ops::AddAssign;

use num_traits::Float;
use progress_bars::SearchProgress;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_constraints;
use crate::dataset::{Dataset, TaggedDataset};
use crate::hall_of_fame::HallOfFame;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::options::Options;
use crate::pop_member::{Evaluator, MemberId, PopMember};
use crate::population::Population;
use crate::{migration, progress_bars, single_iteration, warmup};

pub struct SearchResult<T: Float + AddAssign, Ops, const D: usize> {
    pub hall_of_fame: HallOfFame<T, Ops, D>,
    pub best: PopMember<T, Ops, D>,
}

struct SearchCounters {
    total_cycles: usize,
    cycles_started: usize,
    cycles_completed: usize,
}

impl SearchCounters {
    fn cycles_remaining(&self) -> usize {
        self.total_cycles.saturating_sub(self.cycles_completed)
    }

    fn cycles_remaining_start_for_next_dispatch(&mut self) -> usize {
        let remaining = self.total_cycles.saturating_sub(self.cycles_started);
        self.cycles_started += 1;
        remaining
    }

    fn mark_completed(&mut self) -> usize {
        self.cycles_completed += 1;
        self.cycles_remaining()
    }
}

struct SearchTaskResult<T: Float + AddAssign, Ops, const D: usize> {
    pop_idx: usize,
    curmaxsize: usize,
    evals: u64,
    best_seen: HallOfFame<T, Ops, D>,
    best_sub_pop: Vec<PopMember<T, Ops, D>>,
    pop_state: PopState<T, Ops, D>,
}

pub(crate) struct PopState<T: Float + AddAssign, Ops, const D: usize> {
    pub(crate) pop: Population<T, Ops, D>,
    pub(crate) evaluator: Evaluator<T, D>,
    pub(crate) grad_ctx: dynamic_expressions::GradContext<T, D>,
    pub(crate) rng: StdRng,
    pub(crate) batch_dataset: Option<Dataset<T>>,
    pub(crate) next_id: u64,
    pub(crate) next_birth: u64,
}

impl<T: Float + AddAssign, Ops, const D: usize> PopState<T, Ops, D> {
    fn run_iteration_phase<'a, F, Ret>(
        &'a mut self,
        full_dataset: TaggedDataset<'a, T>,
        options: &'a Options<T, D>,
        curmaxsize: usize,
        stats: &'a RunningSearchStatistics,
        f: F,
    ) -> Ret
    where
        F: FnOnce(
            &mut Population<T, Ops, D>,
            &mut single_iteration::IterationCtx<'a, T, Ops, D, StdRng>,
            TaggedDataset<'a, T>,
        ) -> Ret,
    {
        let phase_dataset = if options.batching {
            let full_data = full_dataset.data;
            if full_data.n_rows == 0 {
                panic!("Cannot batch from an empty dataset (n_rows = 0).");
            }
            let bs = options.batch_size.max(1);
            let needs_new = match self.batch_dataset.as_ref() {
                None => true,
                Some(b) => b.n_rows != bs || b.n_features != full_data.n_features,
            };
            if needs_new {
                self.batch_dataset = Some(Dataset::make_batch_buffer(full_data, bs));
            }
            let batch = self.batch_dataset.as_mut().expect("set above");
            batch.resample_from(full_data, &mut self.rng);
            TaggedDataset {
                data: batch,
                baseline_loss: full_dataset.baseline_loss,
            }
        } else {
            full_dataset
        };

        let mut ctx = single_iteration::IterationCtx {
            rng: &mut self.rng,
            full_dataset,
            curmaxsize,
            stats,
            options,
            evaluator: &mut self.evaluator,
            grad_ctx: &mut self.grad_ctx,
            next_id: &mut self.next_id,
            next_birth: &mut self.next_birth,
            _ops: core::marker::PhantomData,
        };

        f(&mut self.pop, &mut ctx, phase_dataset)
    }
}

struct PopPools<T: Float + AddAssign, Ops, const D: usize> {
    pops: Vec<Option<PopState<T, Ops, D>>>,
    best_sub_pops: Vec<Vec<PopMember<T, Ops, D>>>,
    best: PopMember<T, Ops, D>,
    total_evals: u64,
}

struct EquationSearchState<'a, T: Float + AddAssign, Ops, const D: usize> {
    full_dataset: TaggedDataset<'a, T>,
    options: &'a Options<T, D>,
    n_workers: usize,
    counters: SearchCounters,
    stats: RunningSearchStatistics,
    hall: HallOfFame<T, Ops, D>,
    progress: SearchProgress,
    pools: PopPools<T, Ops, D>,
    order_rng: StdRng,
}

pub fn equation_search<T, Ops, const D: usize>(dataset: &Dataset<T>, options: &Options<T, D>) -> SearchResult<T, Ops, D>
where
    T: Float + AddAssign + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + Send + Sync,
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>
        + dynamic_expressions::strings::OpNames
        + dynamic_expressions::operator_registry::OpRegistry
        + Send
        + Sync,
{
    equation_search_parallel(dataset, options)
}

pub fn equation_search_parallel<T, Ops, const D: usize>(
    dataset: &Dataset<T>,
    options: &Options<T, D>,
) -> SearchResult<T, Ops, D>
where
    T: Float + AddAssign + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + Send + Sync,
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>
        + dynamic_expressions::strings::OpNames
        + dynamic_expressions::operator_registry::OpRegistry
        + Send
        + Sync,
{
    let baseline_loss = if options.use_baseline {
        baseline_loss_from_zero_expression::<T, Ops, D>(dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(dataset, baseline_loss);

    let counters = SearchCounters {
        total_cycles: options.niterations * options.populations,
        cycles_started: 0,
        cycles_completed: 0,
    };

    let stats = RunningSearchStatistics::new(options.maxsize, 100_000);
    let mut hall = HallOfFame::new(options.maxsize);

    let mut progress = SearchProgress::new(options, counters.total_cycles);

    let pools = init_populations(full_dataset, options, &mut hall);
    progress.set_initial_evals(pools.total_evals);

    let order_rng = StdRng::seed_from_u64(options.seed ^ 0x9e37_79b9_7f4a_7c15);

    let pool_threads = rayon::current_num_threads();
    // If we're already running inside Rayon, reserve the current worker thread for orchestration.
    // (Blocking it on `result_rx.recv()` would otherwise reduce the pool capacity by one.)
    let in_rayon_pool = rayon::current_thread_index().is_some();
    let usable_threads = if in_rayon_pool {
        pool_threads.saturating_sub(1)
    } else {
        pool_threads
    };

    assert!(
        usable_threads > 0,
        "equation_search_parallel requires at least 2 Rayon threads when called from inside the Rayon pool"
    );

    let n_workers = usable_threads.min(pools.pops.len()).max(1);

    let mut state = EquationSearchState {
        full_dataset,
        options,
        n_workers,
        counters,
        stats,
        hall,
        progress,
        pools,
        order_rng,
    };

    rayon::scope(|scope| {
        run_scoped_search(scope, &mut state);
    });
    state.progress.finish();

    SearchResult {
        hall_of_fame: state.hall,
        best: state.pools.best,
    }
}

pub struct SearchEngine<T: Float + AddAssign, Ops, const D: usize> {
    dataset: Dataset<T>,
    baseline_loss: Option<T>,
    options: Options<T, D>,
    counters: SearchCounters,
    stats: RunningSearchStatistics,
    hall: HallOfFame<T, Ops, D>,
    progress: SearchProgress,
    pools: PopPools<T, Ops, D>,
    order_rng: StdRng,
    cur_iter: usize,
    task_order: Vec<usize>,
    next_task: usize,
    progress_finished: bool,
}

impl<T, Ops, const D: usize> SearchEngine<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + AddAssign,
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>
        + dynamic_expressions::strings::OpNames
        + dynamic_expressions::operator_registry::OpRegistry,
{
    pub fn new(dataset: Dataset<T>, options: Options<T, D>) -> Self {
        let baseline_loss = if options.use_baseline {
            baseline_loss_from_zero_expression::<T, Ops, D>(&dataset, options.loss.as_ref())
        } else {
            None
        };
        let counters = SearchCounters {
            total_cycles: options.niterations * options.populations,
            cycles_started: 0,
            cycles_completed: 0,
        };

        let stats = RunningSearchStatistics::new(options.maxsize, 100_000);
        let mut hall = HallOfFame::new(options.maxsize);

        let mut progress = SearchProgress::new(&options, counters.total_cycles);

        let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
        let pools = init_populations(full_dataset, &options, &mut hall);
        progress.set_initial_evals(pools.total_evals);

        let order_rng = StdRng::seed_from_u64(options.seed ^ 0x9e37_79b9_7f4a_7c15);

        Self {
            dataset,
            baseline_loss,
            options,
            counters,
            stats,
            hall,
            progress,
            pools,
            order_rng,
            cur_iter: 0,
            task_order: Vec::new(),
            next_task: 0,
            progress_finished: false,
        }
    }

    fn full_dataset_tagged(&self) -> TaggedDataset<'_, T> {
        TaggedDataset::new(&self.dataset, self.baseline_loss)
    }

    pub fn total_cycles(&self) -> usize {
        self.counters.total_cycles
    }

    pub fn cycles_completed(&self) -> usize {
        self.counters.cycles_completed
    }

    pub fn total_evals(&self) -> u64 {
        self.pools.total_evals
    }

    pub fn is_finished(&self) -> bool {
        self.counters.cycles_remaining() == 0
    }

    pub fn hall_of_fame(&self) -> &HallOfFame<T, Ops, D> {
        &self.hall
    }

    pub fn best(&self) -> &PopMember<T, Ops, D> {
        &self.pools.best
    }

    pub fn dataset(&self) -> &Dataset<T> {
        &self.dataset
    }

    pub fn options(&self) -> &Options<T, D> {
        &self.options
    }

    pub fn step(&mut self, n_cycles: usize) -> usize {
        let mut completed = 0usize;
        for _ in 0..n_cycles {
            if !self.step_one_cycle() {
                break;
            }
            completed += 1;
        }
        completed
    }

    pub fn run_to_completion(mut self) -> SearchResult<T, Ops, D> {
        while self.step_one_cycle() {}
        SearchResult {
            hall_of_fame: self.hall,
            best: self.pools.best,
        }
    }

    fn step_one_cycle(&mut self) -> bool {
        if self.is_finished() {
            if !self.progress_finished {
                self.progress.finish();
                self.progress_finished = true;
            }
            return false;
        }

        self.prepare_iteration();
        if self.is_finished() {
            return false;
        }

        // `prepare_iteration` guarantees `next_task < task_order.len()` unless we're finished.
        let pop_idx = self.task_order[self.next_task];
        self.next_task += 1;

        let Some(pop_state) = self.pools.pops[pop_idx].take() else {
            return true;
        };

        let cycles_remaining_start = self.counters.cycles_remaining_start_for_next_dispatch();
        let curmaxsize = warmup::get_cur_maxsize(&self.options, self.counters.total_cycles, cycles_remaining_start);

        let mut stats_snapshot = self.stats.clone();
        stats_snapshot.normalize();

        let full_dataset = self.full_dataset_tagged();
        let res = execute_task(
            full_dataset,
            &self.options,
            pop_idx,
            curmaxsize,
            stats_snapshot,
            pop_state,
        );
        apply_task_result(
            &self.options,
            &mut self.counters,
            &mut self.stats,
            &mut self.hall,
            &mut self.progress,
            &mut self.pools,
            res,
        );

        if self.is_finished() && !self.progress_finished {
            self.progress.finish();
            self.progress_finished = true;
        }

        true
    }

    fn prepare_iteration(&mut self) {
        if self.cur_iter >= self.options.niterations {
            return;
        }
        if !self.task_order.is_empty() && self.next_task < self.task_order.len() {
            return;
        }

        self.task_order = (0..self.pools.pops.len()).collect();
        self.task_order.shuffle(&mut self.order_rng);
        self.next_task = 0;
        self.cur_iter += 1;
    }
}

fn execute_task<T, Ops, const D: usize>(
    full_dataset: TaggedDataset<'_, T>,
    options: &Options<T, D>,
    pop_idx: usize,
    curmaxsize: usize,
    stats: RunningSearchStatistics,
    mut pop_state: PopState<T, Ops, D>,
) -> SearchTaskResult<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops:
        dynamic_expressions::operator_enum::scalar::ScalarOpSet<T> + dynamic_expressions::operator_registry::OpRegistry,
{
    let (evals1, best_seen) =
        pop_state.run_iteration_phase(full_dataset, options, curmaxsize, &stats, |pop, ctx, eval_dataset| {
            single_iteration::s_r_cycle(pop, ctx, eval_dataset)
        });

    let evals2 = pop_state.run_iteration_phase(full_dataset, options, curmaxsize, &stats, |pop, ctx, opt_dataset| {
        single_iteration::optimize_and_simplify_population(pop, ctx, opt_dataset)
    });
    let evals = (evals1.max(0.0) + evals2.max(0.0)) as u64;

    let best_sub_pop = migration::best_sub_pop(&pop_state.pop, options.topn);

    SearchTaskResult {
        pop_idx,
        curmaxsize,
        evals,
        best_seen,
        best_sub_pop,
        pop_state,
    }
}

fn apply_task_result<T, Ops, const D: usize>(
    options: &Options<T, D>,
    counters: &mut SearchCounters,
    stats: &mut RunningSearchStatistics,
    hall: &mut HallOfFame<T, Ops, D>,
    progress: &mut SearchProgress,
    pools: &mut PopPools<T, Ops, D>,
    res: SearchTaskResult<T, Ops, D>,
) where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + AddAssign,
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T> + dynamic_expressions::strings::OpNames,
{
    pools.total_evals = pools.total_evals.saturating_add(res.evals);

    let pop_idx = res.pop_idx;
    let curmaxsize = res.curmaxsize;
    pools.best_sub_pops[pop_idx] = res.best_sub_pop;
    pools.pops[pop_idx] = Some(res.pop_state);

    let st = pools.pops[pop_idx].as_mut().expect("pop exists");

    stats.update_from_population(st.pop.members.iter().map(|m| m.complexity));
    stats.move_window();

    for m in res.best_seen.members() {
        hall.consider(m, options, curmaxsize);
    }
    for m in &st.pop.members {
        hall.consider(m, options, curmaxsize);
        if check_constraints(&m.expr, options, curmaxsize) && m.loss < pools.best.loss {
            pools.best = m.clone();
        }
    }

    if options.migration {
        let mut candidates: Vec<PopMember<T, Ops, D>> = Vec::new();
        for (i, v) in pools.best_sub_pops.iter().enumerate() {
            if i != pop_idx {
                candidates.extend(v.iter().cloned());
            }
        }
        migration::migrate_into(
            &mut st.pop,
            &candidates,
            options.fraction_replaced,
            &mut st.rng,
            &mut st.next_id,
            &mut st.next_birth,
        );
    }

    if options.hof_migration {
        let dominating = hall.pareto_front();
        migration::migrate_into(
            &mut st.pop,
            &dominating,
            options.fraction_replaced_hof,
            &mut st.rng,
            &mut st.next_id,
            &mut st.next_birth,
        );
    }

    let cycles_remaining = counters.mark_completed();
    progress.on_cycle_complete(hall, pools.total_evals, cycles_remaining);
}

fn run_scoped_search<'scope, 'env, T, Ops, const D: usize>(
    scope: &rayon::Scope<'scope>,
    state: &mut EquationSearchState<'env, T, Ops, D>,
) where
    'env: 'scope,
    T: Float + AddAssign + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + Send + Sync + 'scope,
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>
        + dynamic_expressions::strings::OpNames
        + dynamic_expressions::operator_registry::OpRegistry
        + Send
        + Sync
        + 'scope,
{
    let full_dataset = state.full_dataset;
    let options = state.options;

    let (result_tx, result_rx) = std::sync::mpsc::channel::<SearchTaskResult<T, Ops, D>>();

    for _iter in 0..options.niterations {
        let mut task_order: Vec<usize> = (0..state.pools.pops.len()).collect();
        task_order.shuffle(&mut state.order_rng);

        let mut next_task = 0usize;
        let mut in_flight = 0usize;

        while next_task < task_order.len() || in_flight > 0 {
            while in_flight < state.n_workers && next_task < task_order.len() {
                let pop_idx = task_order[next_task];
                next_task += 1;

                let Some(st) = state.pools.pops[pop_idx].take() else {
                    continue;
                };

                let cycles_remaining_start = state.counters.cycles_remaining_start_for_next_dispatch();
                let curmaxsize = warmup::get_cur_maxsize(options, state.counters.total_cycles, cycles_remaining_start);
                let mut stats_snapshot = state.stats.clone();
                stats_snapshot.normalize();

                let result_tx = result_tx.clone();
                scope.spawn(move |_| {
                    let res = execute_task(full_dataset, options, pop_idx, curmaxsize, stats_snapshot, st);
                    let _ = result_tx.send(res);
                });
                in_flight += 1;
            }

            let res = result_rx.recv().expect("worker result channel closed early");
            in_flight -= 1;
            apply_task_result(
                options,
                &mut state.counters,
                &mut state.stats,
                &mut state.hall,
                &mut state.progress,
                &mut state.pools,
                res,
            );
        }
    }
}

fn init_populations<T, Ops, const D: usize>(
    full_dataset: TaggedDataset<'_, T>,
    options: &Options<T, D>,
    hall: &mut HallOfFame<T, Ops, D>,
) -> PopPools<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>,
{
    let dataset = full_dataset.data;
    let mut total_evals: u64 = 0;
    let mut pops: Vec<Option<PopState<T, Ops, D>>> = Vec::with_capacity(options.populations);

    for pop_i in 0..options.populations {
        let mut rng = StdRng::seed_from_u64(options.seed.wrapping_add(pop_i as u64));
        let mut evaluator = Evaluator::new(dataset.n_rows);
        let grad_ctx = dynamic_expressions::GradContext::new(dataset.n_rows);

        let mut next_id = (pop_i as u64) << 32;
        let mut next_birth = 0u64;

        let nlength = 3usize;
        let mut members = Vec::with_capacity(options.population_size);
        for _ in 0..options.population_size {
            let expr = crate::mutation_functions::random_expr_append_ops(
                &mut rng,
                &options.operators,
                dataset.n_features,
                nlength,
                options.maxsize,
            );
            let mut m = PopMember::from_expr(MemberId(next_id), None, next_birth, expr, dataset.n_features);
            next_id += 1;
            next_birth += 1;
            let _ = m.evaluate(&full_dataset, options, &mut evaluator);
            total_evals += 1;
            hall.consider(&m, options, options.maxsize);
            members.push(m);
        }

        pops.push(Some(PopState {
            pop: Population::new(members),
            evaluator,
            grad_ctx,
            rng,
            batch_dataset: None,
            next_id,
            next_birth,
        }));
    }

    let mut best: Option<PopMember<T, Ops, D>> = None;
    for st in pops.iter().flatten() {
        for m in &st.pop.members {
            if !check_constraints(&m.expr, options, options.maxsize) {
                continue;
            }
            match &best {
                None => best = Some(m.clone()),
                Some(cur) => {
                    if m.loss < cur.loss {
                        best = Some(m.clone());
                    }
                }
            }
        }
    }
    let best = best.unwrap_or_else(|| {
        pops.iter()
            .flatten()
            .next()
            .expect("at least one population member exists")
            .pop
            .members[0]
            .clone()
    });

    let mut best_sub_pops: Vec<Vec<PopMember<T, Ops, D>>> = vec![Vec::new(); pops.len()];
    for i in 0..pops.len() {
        let st = pops[i].as_ref().expect("population exists");
        best_sub_pops[i] = migration::best_sub_pop(&st.pop, options.topn);
    }

    PopPools {
        pops,
        best_sub_pops,
        best,
        total_evals,
    }
}
