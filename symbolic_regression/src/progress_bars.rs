#[cfg(feature = "progress")]
mod imp {
    use std::fmt::Display;
    use std::io::IsTerminal;
    use std::time::{Duration, Instant};

    use dynamic_expressions::strings::OpNames;
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
    use num_traits::Float;

    use crate::hall_of_fame::HallOfFame;
    use crate::options::{Options, OutputStyle};

    struct ProgressTracking {
        last_speed_time: Instant,
        evals_last: u64,
        speeds: Vec<f64>,
        last_msg_update: Instant,
    }

    #[derive(Copy, Clone, Debug)]
    struct RenderOptions {
        ansi: bool,
    }

    pub(crate) struct SearchProgress {
        show: bool,
        bar: ProgressBar,
        start_time: Instant,
        msg_min_interval: Duration,
        tracking: ProgressTracking,
        render: RenderOptions,
    }

    impl SearchProgress {
        pub(crate) fn new<T: Float, const D: usize>(options: &Options<T, D>, total_cycles: usize) -> Self {
            let show = options.progress && std::io::stderr().is_terminal();

            let ansi = match options.output_style {
                OutputStyle::Plain => {
                    console::set_colors_enabled_stderr(false);
                    false
                }
                OutputStyle::Ansi => {
                    console::set_colors_enabled_stderr(true);
                    true
                }
                OutputStyle::Auto => console::colors_enabled_stderr(),
            };
            let render = RenderOptions { ansi };

            let bar = if show {
                let pb = ProgressBar::new(total_cycles as u64);
                pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
                pb.set_prefix(format!("Evolving for {} iterations...", options.niterations));
                let style = ProgressStyle::with_template(
                    "{prefix} {wide_bar} {pos:>7}/{len:7} [{elapsed_precise}<{eta_precise}]\n{msg}",
                )
                .unwrap();
                pb.set_style(style);
                pb
            } else {
                ProgressBar::hidden()
            };

            Self {
                show,
                bar,
                start_time: Instant::now(),
                msg_min_interval: Duration::from_millis(500),
                tracking: ProgressTracking {
                    last_speed_time: Instant::now(),
                    evals_last: 0,
                    speeds: Vec::new(),
                    last_msg_update: Instant::now(),
                },
                render,
            }
        }

        pub(crate) fn set_initial_evals(&mut self, total_evals: u64) {
            self.tracking.evals_last = total_evals;
            self.tracking.last_speed_time = Instant::now();
        }

        pub(crate) fn on_cycle_complete<T, Ops, const D: usize>(
            &mut self,
            hall: &HallOfFame<T, Ops, D>,
            total_evals: u64,
            cycles_remaining: usize,
        ) where
            T: Float + num_traits::ToPrimitive + Display,
            Ops: OpNames,
        {
            if !self.show {
                return;
            }
            self.bar.inc(1);
            update_progress_msg(ProgressMsgCtx {
                pb: &self.bar,
                hall,
                start_time: self.start_time,
                msg_min_interval: self.msg_min_interval,
                progress: &mut self.tracking,
                render: self.render,
                total_evals,
                cycles_remaining,
            });
        }

        pub(crate) fn finish(&self) {
            if self.show {
                self.bar.finish();
            }
        }
    }

    struct ProgressMsgCtx<'a, T: Float, Ops, const D: usize> {
        pb: &'a ProgressBar,
        hall: &'a HallOfFame<T, Ops, D>,
        start_time: Instant,
        msg_min_interval: Duration,
        progress: &'a mut ProgressTracking,
        render: RenderOptions,
        total_evals: u64,
        cycles_remaining: usize,
    }

    fn update_progress_msg<T, Ops, const D: usize>(ctx: ProgressMsgCtx<'_, T, Ops, D>)
    where
        T: Float + num_traits::ToPrimitive + Display,
        Ops: OpNames,
    {
        let ProgressMsgCtx {
            pb,
            hall,
            start_time,
            msg_min_interval,
            progress,
            render,
            total_evals,
            cycles_remaining,
        } = ctx;

        let now = Instant::now();
        if now.duration_since(progress.last_speed_time) >= Duration::from_secs(1) {
            let dt = now.duration_since(progress.last_speed_time).as_secs_f64().max(1e-9);
            let evals_since = total_evals.saturating_sub(progress.evals_last);
            progress.speeds.push((evals_since as f64) / dt);
            if progress.speeds.len() > 20 {
                progress.speeds.remove(0);
            }
            progress.evals_last = total_evals;
            progress.last_speed_time = now;
        }

        if now.duration_since(progress.last_msg_update) < msg_min_interval && cycles_remaining != 0 {
            return;
        }

        let term_width = {
            let (_, w) = console::Term::stderr().size();
            (w as usize).max(80)
        };

        let avg_speed = if progress.speeds.is_empty() {
            None
        } else {
            Some(progress.speeds.iter().copied().sum::<f64>() / (progress.speeds.len() as f64))
        };

        let info_line = match avg_speed {
            Some(s) => format!(
                "Info: eval/s~{:<8.2e} | evals={} | t={:.1}s",
                s,
                total_evals,
                start_time.elapsed().as_secs_f64()
            ),
            None => format!(
                "Info: eval/s=[.....]  | evals={} | t={:.1}s",
                total_evals,
                start_time.elapsed().as_secs_f64()
            ),
        };

        let hof = format_hall_of_fame(hall, term_width, 12, render);
        let hof_title = "Hall of Fame:".to_string();
        pb.set_message(format!(
            "{}\n{}\n{}",
            truncate_to_width(&info_line, term_width, render),
            truncate_to_width(&hof_title, term_width, render),
            hof
        ));

        progress.last_msg_update = now;
    }

    fn format_hall_of_fame<T, Ops, const D: usize>(
        hall: &HallOfFame<T, Ops, D>,
        terminal_width: usize,
        max_entries: usize,
        render: RenderOptions,
    ) -> String
    where
        T: Float + num_traits::ToPrimitive + Display,
        Ops: OpNames,
    {
        let terminal_width = terminal_width.max(80);
        let raw_border = "─".repeat(terminal_width.saturating_sub(1));
        let raw_header = format!("{:<10}  {:<10}  {}", "Complexity", "Loss", "Equation");

        let border = raw_border;
        let header = if render.ansi {
            let s = console::Style::new().bold().underlined();

            let complexity_plain = "Complexity";
            let loss_plain = "Loss";
            let equation_plain = "Equation";

            let complexity = format!(
                "{}{}",
                s.apply_to(complexity_plain),
                " ".repeat(10_usize.saturating_sub(complexity_plain.len()))
            );
            let loss = format!(
                "{}{}",
                s.apply_to(loss_plain),
                " ".repeat(10_usize.saturating_sub(loss_plain.len()))
            );
            let equation = s.apply_to(equation_plain).to_string();

            format!("{complexity}  {loss}  {equation}")
        } else {
            raw_header
        };

        let mut out = String::new();
        out.push_str(&border);
        out.push('\n');
        out.push_str(&truncate_to_width(&header, terminal_width, render));
        out.push('\n');

        for m in hall.pareto_front().into_iter().take(max_entries) {
            let loss = m.loss.to_f64().unwrap_or(f64::INFINITY);
            let stats = format!("{:<10}  {:<10.3e}  ", m.complexity, loss);
            let left_cols_width = stats.chars().count();

            let eqn_plain = m.expr.to_string();
            let eqn_lines = wrap_equation(&eqn_plain, terminal_width, left_cols_width);

            if let Some((first, rest)) = eqn_lines.split_first() {
                let first_eq = first.to_string();
                out.push_str(&truncate_to_width(&(stats.clone() + &first_eq), terminal_width, render));
                out.push('\n');
                for line in rest {
                    let eq = line.to_string();
                    let padded = format!("{}{}", " ".repeat(left_cols_width), eq);
                    out.push_str(&truncate_to_width(&padded, terminal_width, render));
                    out.push('\n');
                }
            } else {
                out.push_str(&truncate_to_width(&stats, terminal_width, render));
                out.push('\n');
            }
        }

        out.push_str(&border);
        out
    }

    fn wrap_equation(eqn: &str, terminal_width: usize, left_cols_width: usize) -> Vec<String> {
        let dots = "...";
        let avail = terminal_width
            .saturating_sub(1)
            .saturating_sub(left_cols_width)
            .saturating_sub(dots.len())
            .max(10);

        let mut out = Vec::new();
        let mut start = 0;
        while start < eqn.len() {
            let mut end = (start + avail).min(eqn.len());
            while end > start && !eqn.is_char_boundary(end) {
                end -= 1;
            }
            if end == start {
                break;
            }
            let mut chunk = eqn[start..end].to_string();
            if end < eqn.len() {
                chunk.push_str(dots);
            }
            out.push(chunk);
            start = end;
        }
        out
    }

    fn truncate_to_width(s: &str, terminal_width: usize, render: RenderOptions) -> String {
        let max = terminal_width.saturating_sub(1);
        if render.ansi {
            return console::truncate_str(s, max, "").into_owned();
        }
        if s.chars().count() <= max {
            s.to_string()
        } else {
            s.chars().take(max).collect()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn header_equation_column_aligns_with_rows() {
            let render = RenderOptions { ansi: true };

            let raw_header = format!("{:<10}  {:<10}  {}", "Complexity", "Loss", "Equation");
            let header = if render.ansi {
                let s = console::Style::new().bold().underlined();

                let complexity_plain = "Complexity";
                let loss_plain = "Loss";
                let equation_plain = "Equation";

                let complexity = format!(
                    "{}{}",
                    s.apply_to(complexity_plain),
                    " ".repeat(10_usize.saturating_sub(complexity_plain.len()))
                );
                let loss = format!(
                    "{}{}",
                    s.apply_to(loss_plain),
                    " ".repeat(10_usize.saturating_sub(loss_plain.len()))
                );
                let equation = s.apply_to(equation_plain).to_string();
                format!("{complexity}  {loss}  {equation}")
            } else {
                raw_header
            };

            let header_plain = console::strip_ansi_codes(&header);
            let eq_header_start = header_plain.find("Equation").expect("header should contain Equation");

            let stats = format!("{:<10}  {:<10.3e}  ", 5_u32, 1.234_f64);
            let row = format!("{stats}x0");
            let eq_row_start = row.find("x0").expect("row should contain equation");

            assert_eq!(
                eq_header_start, eq_row_start,
                "Equation header must align with equation column"
            );
        }
    }
}

#[cfg(not(feature = "progress"))]
mod imp {
    use num_traits::Float;

    use crate::hall_of_fame::HallOfFame;
    use crate::options::Options;

    pub(crate) struct SearchProgress;

    impl SearchProgress {
        pub(crate) fn new<T: Float, const D: usize>(_options: &Options<T, D>, _total_cycles: usize) -> Self {
            Self
        }

        pub(crate) fn set_initial_evals(&mut self, _total_evals: u64) {}

        pub(crate) fn on_cycle_complete<T: Float, Ops, const D: usize>(
            &mut self,
            _hall: &HallOfFame<T, Ops, D>,
            _total_evals: u64,
            _cycles_remaining: usize,
        ) {
        }

        pub(crate) fn finish(&self) {}
    }
}

pub(crate) use imp::SearchProgress;
