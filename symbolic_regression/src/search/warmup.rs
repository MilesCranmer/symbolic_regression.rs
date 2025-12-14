use crate::options::Options;
use num_traits::Float;

pub fn get_cur_maxsize<T: Float, const D: usize>(
    options: &Options<T, D>,
    total_cycles: usize,
    cycles_remaining: usize,
) -> usize {
    let cycles_elapsed = total_cycles.saturating_sub(cycles_remaining);
    let fraction_elapsed = (cycles_elapsed as f32) / (total_cycles as f32);
    let in_warmup = fraction_elapsed <= options.warmup_maxsize_by;

    if options.warmup_maxsize_by > 0.0 && in_warmup {
        3 + (((options.maxsize - 3) as f32) * fraction_elapsed / options.warmup_maxsize_by).floor()
            as usize
    } else {
        options.maxsize
    }
}
