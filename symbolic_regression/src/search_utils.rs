use std::fmt::Display;
use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::Float;

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_constraints;
use crate::dataset::{Dataset, TaggedDataset};
use crate::hall_of_fame::HallOfFame;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::options::Options;
use crate::pop_member::{Evaluator, MemberId, PopMember};
use crate::population::Population;
use crate::progress_bars::SearchProgress;
use crate::random::shuffle;
use crate::stop_controller::StopController;
use crate::{migration, single_iteration, warmup};

pub struct SearchResult<T: Float + AddAssign, Ops, const D: usize> {
    pub hall_of_fame: HallOfFame<T, Ops, D>,
    pub best: PopMember<T, Ops, D>,
}

fn usable_rayon_threads() -> usize {
    let pool_threads = rayon::current_num_threads();
    let in_rayon_pool = rayon::current_thread_index().is_some();
    if in_rayon_pool {
        pool_threads.saturating_sub(1)
    } else {
        pool_threads
    }
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
    pub(crate) rng: Rng,
    pub(crate) batch_dataset: Option<Dataset<T>>,
    pub(crate) next_id: u64,
}

impl<T: Float + AddAssign, Ops, const D: usize> PopState<T, Ops, D> {
    fn run_iteration_phase<'a, F, Ret>(
        &'a mut self,
        f: F,
        full_dataset: TaggedDataset<'a, T>,
        options: &'a Options<T, D>,
        curmaxsize: usize,
        stats: &'a RunningSearchStatistics,
        controller: &'a StopController,
    ) -> Ret
    where
        F: FnOnce(
            &mut Population<T, Ops, D>,
            &mut single_iteration::IterationCtx<'a, T, Ops, D>,
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
            controller,
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

pub fn equation_search<T, Ops, const D: usize>(dataset: &Dataset<T>, options: &Options<T, D>) -> SearchResult<T, Ops, D>
where
    T: Float + AddAssign + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    let baseline_loss = if options.use_baseline {
        baseline_loss_from_zero_expression::<T, Ops, D>(dataset, options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(dataset, baseline_loss);

    let controller = StopController::from_options(options);

    let stats = RunningSearchStatistics::new(options.maxsize, 100_000);
    let mut hall = HallOfFame::new(options.maxsize);

    let pools = init_populations(full_dataset, options, &controller, &mut hall);
    let counters = SearchCounters {
        total_cycles: options.niterations * pools.pops.len(),
        cycles_started: 0,
        cycles_completed: 0,
    };

    let mut progress = SearchProgress::new(options, counters.total_cycles);
    progress.set_initial_evals(pools.total_evals);

    let mut core = SearchCore {
        counters,
        stats,
        hall,
        progress,
        pools,
        order_rng: Rng::with_seed(options.seed ^ 0x9e37_79b9_7f4a_7c15),
        cur_iter: 0,
        task_order: Vec::new(),
        next_task: 0,
        progress_finished: false,
    };

    let _ = core.step(dataset, baseline_loss, options, &controller, usize::MAX);
    core.finish_progress_if_needed();

    let SearchCore { hall, pools, .. } = core;
    SearchResult {
        hall_of_fame: hall,
        best: pools.best,
    }
}

struct SearchCore<T: Float + AddAssign, Ops, const D: usize> {
    counters: SearchCounters,
    stats: RunningSearchStatistics,
    hall: HallOfFame<T, Ops, D>,
    progress: SearchProgress,
    pools: PopPools<T, Ops, D>,
    order_rng: Rng,
    cur_iter: usize,
    task_order: Vec<usize>,
    next_task: usize,
    progress_finished: bool,
}

impl<T: Float + AddAssign, Ops, const D: usize> SearchCore<T, Ops, D> {
    fn prepare_iteration_state(&mut self, niterations: usize) {
        if self.cur_iter >= niterations {
            return;
        }
        if !self.task_order.is_empty() && self.next_task < self.task_order.len() {
            return;
        }

        self.task_order = (0..self.pools.pops.len()).collect();
        shuffle(&mut self.order_rng, &mut self.task_order);
        self.next_task = 0;
        self.cur_iter += 1;
    }

    #[inline]
    fn finish_progress_if_needed(&mut self) {
        if !self.progress_finished {
            self.progress.finish();
            self.progress_finished = true;
        }
    }
}

impl<T, Ops, const D: usize> SearchCore<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + AddAssign + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    fn step(
        &mut self,
        dataset: &Dataset<T>,
        baseline_loss: Option<T>,
        options: &Options<T, D>,
        controller: &StopController,
        n_cycles: usize,
    ) -> usize {
        if n_cycles == 0 {
            return 0;
        }

        let is_finished = |counters: &SearchCounters| counters.cycles_remaining() == 0 || controller.is_cancelled();

        if is_finished(&self.counters) {
            self.finish_progress_if_needed();
            return 0;
        }

        let usable_threads = usable_rayon_threads();
        let need_inline = usable_threads == 0;
        let n_workers = usable_threads.min(self.pools.pops.len()).max(1);

        let completed_total = rayon::scope(|scope| {
            let (result_tx, result_rx) = std::sync::mpsc::channel::<SearchTaskResult<T, Ops, D>>();
            let mut in_flight = 0usize;
            let mut completed_total = 0usize;
            let mut stop_dispatching = false;

            while completed_total < n_cycles && (!stop_dispatching || in_flight > 0) {
                if is_finished(&self.counters) {
                    stop_dispatching = true;
                }

                if controller.should_stop(self.pools.total_evals) {
                    stop_dispatching = true;
                    controller.cancel();
                }

                if !stop_dispatching && in_flight == 0 && self.next_task >= self.task_order.len() {
                    self.prepare_iteration_state(options.niterations);
                    if self.next_task >= self.task_order.len() {
                        stop_dispatching = true;
                    }
                }

                while !stop_dispatching
                    && in_flight < n_workers
                    && completed_total + in_flight < n_cycles
                    && self.next_task < self.task_order.len()
                {
                    if is_finished(&self.counters) {
                        stop_dispatching = true;
                        break;
                    }

                    if controller.should_stop(self.pools.total_evals) {
                        stop_dispatching = true;
                        controller.cancel();
                        break;
                    }

                    let pop_idx = self.task_order[self.next_task];
                    self.next_task += 1;

                    let Some(pop_state) = self.pools.pops[pop_idx].take() else {
                        continue;
                    };

                    let cycles_remaining_start = self.counters.cycles_remaining_start_for_next_dispatch();
                    let curmaxsize =
                        warmup::get_cur_maxsize(options, self.counters.total_cycles, cycles_remaining_start);
                    let mut stats_snapshot = self.stats.clone();
                    stats_snapshot.normalize();

                    if need_inline {
                        // Inline exec avoids deadlock in 1-thread pools, while preserving the same
                        // "send -> recv -> apply" completion-driven semantics as the parallel path.
                        let full_dataset = TaggedDataset::new(dataset, baseline_loss);
                        let res = execute_task(
                            full_dataset,
                            options,
                            pop_idx,
                            curmaxsize,
                            stats_snapshot,
                            pop_state,
                            controller,
                        );
                        let _ = result_tx.send(res);
                    } else {
                        let result_tx = result_tx.clone();
                        scope.spawn(move |_| {
                            let full_dataset = TaggedDataset::new(dataset, baseline_loss);
                            let res = execute_task(
                                full_dataset,
                                options,
                                pop_idx,
                                curmaxsize,
                                stats_snapshot,
                                pop_state,
                                controller,
                            );
                            let _ = result_tx.send(res);
                        });
                    }
                    in_flight += 1;
                }

                if in_flight == 0 {
                    if stop_dispatching {
                        break;
                    }
                    continue;
                }

                let res = result_rx.recv().expect("worker result channel closed early");
                in_flight -= 1;
                apply_task_result(
                    options,
                    &mut self.counters,
                    &mut self.stats,
                    &mut self.hall,
                    &mut self.progress,
                    &mut self.pools,
                    res,
                );
                completed_total += 1;

                if controller.should_stop(self.pools.total_evals) {
                    stop_dispatching = true;
                    controller.cancel();
                }
            }

            completed_total
        });

        if is_finished(&self.counters) {
            self.finish_progress_if_needed();
        }

        completed_total
    }
}

pub struct SearchEngine<T: Float + AddAssign, Ops, const D: usize> {
    dataset: Dataset<T>,
    baseline_loss: Option<T>,
    options: Options<T, D>,
    controller: StopController,
    core: SearchCore<T, Ops, D>,
}

impl<T, Ops, const D: usize> SearchEngine<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + Display + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    pub fn new(dataset: Dataset<T>, options: Options<T, D>) -> Self {
        let baseline_loss = if options.use_baseline {
            baseline_loss_from_zero_expression::<T, Ops, D>(&dataset, options.loss.as_ref())
        } else {
            None
        };

        let controller = StopController::from_options(&options);

        let stats = RunningSearchStatistics::new(options.maxsize, 100_000);
        let mut hall = HallOfFame::new(options.maxsize);

        let full_dataset = TaggedDataset::new(&dataset, baseline_loss);
        let pools = init_populations(full_dataset, &options, &controller, &mut hall);
        let counters = SearchCounters {
            total_cycles: options.niterations * pools.pops.len(),
            cycles_started: 0,
            cycles_completed: 0,
        };

        let mut progress = SearchProgress::new(&options, counters.total_cycles);
        progress.set_initial_evals(pools.total_evals);

        let order_rng = Rng::with_seed(options.seed ^ 0x9e37_79b9_7f4a_7c15);

        let core = SearchCore {
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
        };

        Self {
            dataset,
            baseline_loss,
            options,
            controller,
            core,
        }
    }

    pub fn total_cycles(&self) -> usize {
        self.core.counters.total_cycles
    }

    pub fn cycles_completed(&self) -> usize {
        self.core.counters.cycles_completed
    }

    pub fn total_evals(&self) -> u64 {
        self.core.pools.total_evals
    }

    pub fn is_finished(&self) -> bool {
        self.core.counters.cycles_remaining() == 0 || self.controller.is_cancelled()
    }

    pub fn hall_of_fame(&self) -> &HallOfFame<T, Ops, D> {
        &self.core.hall
    }

    pub fn best(&self) -> &PopMember<T, Ops, D> {
        &self.core.pools.best
    }

    pub fn dataset(&self) -> &Dataset<T> {
        &self.dataset
    }

    pub fn options(&self) -> &Options<T, D> {
        &self.options
    }

    pub fn step(&mut self, n_cycles: usize) -> usize
    where
        T: Send + Sync,
        Ops: Send + Sync,
    {
        self.core.step(
            &self.dataset,
            self.baseline_loss,
            &self.options,
            &self.controller,
            n_cycles,
        )
    }

    pub fn run_to_completion(mut self) -> SearchResult<T, Ops, D>
    where
        T: Send + Sync,
        Ops: Send + Sync,
    {
        while self.step(usize::MAX) > 0 {}

        let SearchCore { hall, pools, .. } = self.core;
        SearchResult {
            hall_of_fame: hall,
            best: pools.best,
        }
    }
}

fn execute_task<T, Ops, const D: usize>(
    full_dataset: TaggedDataset<'_, T>,
    options: &Options<T, D>,
    pop_idx: usize,
    curmaxsize: usize,
    stats: RunningSearchStatistics,
    mut pop_state: PopState<T, Ops, D>,
    controller: &StopController,
) -> SearchTaskResult<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    if controller.is_cancelled() {
        return SearchTaskResult {
            pop_idx,
            curmaxsize,
            evals: 0,
            best_seen: HallOfFame::new(options.maxsize),
            best_sub_pop: migration::best_sub_pop(&pop_state.pop, options.topn),
            pop_state,
        };
    }
    let (evals1, best_seen) = pop_state.run_iteration_phase(
        single_iteration::s_r_cycle,
        full_dataset,
        options,
        curmaxsize,
        &stats,
        controller,
    );

    let evals2 = pop_state.run_iteration_phase(
        single_iteration::optimize_and_simplify_population,
        full_dataset,
        options,
        curmaxsize,
        &stats,
        controller,
    );
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
    Ops: dynamic_expressions::OperatorSet<T = T>,
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
            options.deterministic,
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
            options.deterministic,
        );
    }

    let cycles_remaining = counters.mark_completed();
    progress.on_cycle_complete(hall, pools.total_evals, cycles_remaining);
}

fn init_populations<T, Ops, const D: usize>(
    full_dataset: TaggedDataset<'_, T>,
    options: &Options<T, D>,
    controller: &StopController,
    hall: &mut HallOfFame<T, Ops, D>,
) -> PopPools<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let dataset = full_dataset.data;
    let mut total_evals: u64 = 0;
    let mut pops: Vec<Option<PopState<T, Ops, D>>> = Vec::with_capacity(options.populations);

    for pop_i in 0..options.populations {
        if controller.is_cancelled() {
            break;
        }
        let mut rng = Rng::with_seed(options.seed.wrapping_add(pop_i as u64));
        let mut evaluator = Evaluator::new(dataset.n_rows);
        let grad_ctx = dynamic_expressions::GradContext::new(dataset.n_rows);

        let mut next_id = (pop_i as u64) << 32;

        let nlength = 3usize;
        let mut members = Vec::with_capacity(options.population_size);
        for _ in 0..options.population_size {
            if controller.is_cancelled() {
                break;
            }
            let expr = crate::mutation_functions::random_expr_append_ops(
                &mut rng,
                &options.operators,
                dataset.n_features,
                nlength,
                options.maxsize,
            );
            let mut m = PopMember::from_expr(MemberId(next_id), None, expr, dataset.n_features, options);
            next_id += 1;
            let _ = m.evaluate(&full_dataset, options, &mut evaluator);
            total_evals += 1;
            hall.consider(&m, options, options.maxsize);
            members.push(m);
        }

        if members.is_empty() {
            break;
        }
        pops.push(Some(PopState {
            pop: Population::new(members),
            evaluator,
            grad_ctx,
            rng,
            batch_dataset: None,
            next_id,
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

#[cfg(test)]
mod batching_search_tests {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use ndarray::{Array1, Array2};

    use super::SearchEngine;
    use crate::dataset::Dataset;
    use crate::{Operators, Options};

    #[test]
    fn search_engine_allocates_batch_buffer_when_batching_enabled() {
        type T = f64;
        type Ops = BuiltinOpsF64;
        const D: usize = 3;

        let n_rows = 20;
        let n_features = 2;
        let mut x = Array2::<T>::zeros((n_features, n_rows));
        for row in 0..n_rows {
            x[(0, row)] = row as T;
            x[(1, row)] = (row as T) + 100.0;
        }
        let y = Array1::from_iter((0..n_rows).map(|i| i as T));
        let dataset = Dataset::new(x, y);

        let operators = Operators::<D>::from_names_by_arity::<Ops>(&["sin"], &["+", "*"], &[]).expect("valid opset");
        let options: Options<T, D> = Options {
            operators,
            batching: true,
            batch_size: 5,
            niterations: 1,
            ncycles_per_iteration: 1,
            populations: 1,
            population_size: 6,
            progress: false,
            should_optimize_constants: false,
            should_simplify: false,
            deterministic: true,
            ..Default::default()
        };

        let mut engine = SearchEngine::<T, Ops, D>::new(dataset, options.clone());
        assert!(
            engine.core.pools.pops[0]
                .as_ref()
                .expect("pop exists")
                .batch_dataset
                .is_none(),
            "batch buffer should be allocated lazily"
        );

        let _ = engine.step(1);

        let pop_state = engine.core.pools.pops[0].as_ref().expect("pop exists");
        let batch = pop_state.batch_dataset.as_ref().expect("batch buffer created");
        assert_eq!(batch.n_rows, options.batch_size.max(1));
        assert_eq!(batch.n_features, engine.dataset.n_features);
    }

    #[test]
    fn search_engine_does_not_allocate_batch_buffer_when_batching_disabled() {
        type T = f64;
        type Ops = BuiltinOpsF64;
        const D: usize = 3;

        let n_rows = 20;
        let n_features = 2;
        let mut x = Array2::<T>::zeros((n_features, n_rows));
        for row in 0..n_rows {
            x[(0, row)] = row as T;
            x[(1, row)] = (row as T) + 100.0;
        }
        let y = Array1::from_iter((0..n_rows).map(|i| i as T));
        let dataset = Dataset::new(x, y);

        let operators = Operators::<D>::from_names_by_arity::<Ops>(&["sin"], &["+", "*"], &[]).expect("valid opset");
        let options: Options<T, D> = Options {
            operators,
            batching: false,
            batch_size: 5,
            niterations: 1,
            ncycles_per_iteration: 1,
            populations: 1,
            population_size: 6,
            progress: false,
            should_optimize_constants: false,
            should_simplify: false,
            deterministic: true,
            ..Default::default()
        };

        let mut engine = SearchEngine::<T, Ops, D>::new(dataset, options);
        let _ = engine.step(1);

        let pop_state = engine.core.pools.pops[0].as_ref().expect("pop exists");
        assert!(pop_state.batch_dataset.is_none());
    }
}
