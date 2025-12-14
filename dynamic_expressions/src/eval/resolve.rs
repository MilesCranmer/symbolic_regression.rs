use crate::expr::Src;
use crate::operators::scalar::{GradRef, SrcRef};
use num_traits::Float;

fn slot_slice<'a, T>(
    slot: usize,
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> &'a [T] {
    if slot < dst_slot {
        &before[slot]
    } else if slot > dst_slot {
        &after[slot - dst_slot - 1]
    } else {
        panic!("source references dst slot");
    }
}

pub fn resolve_val_src<'a, T: Float>(
    src: Src,
    x_data: &'a [T],
    n_features: usize,
    consts: &'a [T],
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> SrcRef<'a, T> {
    match src {
        Src::Var(f) => SrcRef::Strided {
            data: x_data,
            offset: f as usize,
            stride: n_features,
        },
        Src::Const(c) => SrcRef::Const(consts[c as usize]),
        Src::Slot(s) => SrcRef::Slice(slot_slice(s as usize, dst_slot, before, after)),
    }
}

pub fn resolve_der_src<'a, T: Float>(
    src: Src,
    direction: usize,
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> SrcRef<'a, T> {
    match src {
        Src::Var(f) => {
            if f as usize == direction {
                SrcRef::Const(T::one())
            } else {
                SrcRef::Const(T::zero())
            }
        }
        Src::Const(_) => SrcRef::Const(T::zero()),
        Src::Slot(s) => SrcRef::Slice(slot_slice(s as usize, dst_slot, before, after)),
    }
}

pub fn resolve_grad_src<'a, T: Float>(
    src: Src,
    variable: bool,
    dst_slot: usize,
    before: &'a [Vec<T>],
    after: &'a [Vec<T>],
) -> GradRef<'a, T> {
    match src {
        Src::Var(f) => {
            if variable {
                GradRef::Basis(f as usize)
            } else {
                GradRef::Zero
            }
        }
        Src::Const(c) => {
            if variable {
                GradRef::Zero
            } else {
                GradRef::Basis(c as usize)
            }
        }
        Src::Slot(s) => GradRef::Slice(slot_slice(s as usize, dst_slot, before, after)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::scalar::SrcRef;
    use ndarray::Array2;

    #[test]
    fn slot_slice_panics_if_src_references_dst() {
        let x = Array2::from_shape_vec((2, 1), vec![0.0f64; 2]).unwrap();
        let x_data = x.as_slice().unwrap();
        let n_features = x.ncols();
        let consts: Vec<f64> = vec![];
        let before: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        let after: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];

        let dst_slot = 0usize;
        let res = std::panic::catch_unwind(|| {
            resolve_val_src(
                Src::Slot(dst_slot as u16),
                x_data,
                n_features,
                &consts,
                dst_slot,
                &before,
                &after,
            )
        });
        assert!(res.is_err());

        let r = resolve_val_src(
            Src::Slot(1),
            x_data,
            n_features,
            &consts,
            dst_slot,
            &before,
            &after,
        );
        assert!(matches!(r, SrcRef::Slice(s) if s.len() == 2));
    }

    #[test]
    fn slot_slice_uses_before_when_slot_is_less_than_dst() {
        let x = Array2::from_shape_vec((2, 1), vec![0.0f64; 2]).unwrap();
        let x_data = x.as_slice().unwrap();
        let n_features = x.ncols();
        let consts: Vec<f64> = vec![];
        let before: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        let after: Vec<Vec<f64>> = vec![vec![3.0, 4.0]];

        let dst_slot = 1usize;
        let r = resolve_val_src(
            Src::Slot(0),
            x_data,
            n_features,
            &consts,
            dst_slot,
            &before,
            &after,
        );
        assert!(matches!(r, SrcRef::Slice(s) if s.len() == 2 && s[0] == 1.0 && s[1] == 2.0));
    }
}
