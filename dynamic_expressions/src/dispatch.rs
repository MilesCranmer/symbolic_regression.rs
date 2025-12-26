use num_traits::Float;

use crate::evaluate::EvalOptions;

#[derive(Copy, Clone, Debug)]
pub enum SrcRef<'a, T> {
    Slice(&'a [T]),
    Const(T),
}

#[derive(Copy, Clone, Debug)]
pub enum GradRef<'a, T> {
    /// Dir-major gradient slab: `grad[dir*n_rows + row]`.
    Slice(&'a [T]),
    /// One-hot basis direction (value is 1 if `dir == basis_dir` else 0).
    Basis(usize),
    /// All zeros.
    Zero,
}

#[inline]
pub fn grad_at<T: Float>(g: GradRef<'_, T>, dir: usize, row: usize, n_rows: usize) -> T {
    match g {
        GradRef::Slice(s) => s[dir * n_rows + row],
        GradRef::Basis(k) => {
            if dir == k {
                T::one()
            } else {
                T::zero()
            }
        }
        GradRef::Zero => T::zero(),
    }
}

pub struct EvalKernelCtx<'a, 'b, T> {
    pub out: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub opts: &'b EvalOptions,
}

pub struct DiffKernelCtx<'a, 'b, T> {
    pub out_val: &'b mut [T],
    pub out_der: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub dargs: &'b [SrcRef<'a, T>],
    pub opts: &'b EvalOptions,
}

pub struct GradKernelCtx<'a, 'b, T> {
    pub out_val: &'b mut [T],
    /// Dir-major buffer: `out_grad[dir*n_rows + row]`.
    pub out_grad: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub arg_grads: &'b [GradRef<'a, T>],
    pub n_dir: usize,
    pub n_rows: usize,
    pub opts: &'b EvalOptions,
}
