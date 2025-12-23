use num_traits::Float;

use crate::pop_member::PopMember;

pub struct Population<T: Float, Ops, const D: usize> {
    pub members: Vec<PopMember<T, Ops, D>>,
}

impl<T: Float, Ops, const D: usize> Population<T, Ops, D> {
    pub fn new(members: Vec<PopMember<T, Ops, D>>) -> Self {
        Self { members }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn oldest_index(&self) -> usize {
        let (idx, _) = self
            .members
            .iter()
            .enumerate()
            .min_by_key(|(_, m)| m.birth)
            .expect("population must be non-empty");
        idx
    }

    pub fn two_oldest_indices(&self) -> (usize, usize) {
        let mut first: Option<(usize, u64)> = None;
        let mut second: Option<(usize, u64)> = None;
        for (i, m) in self.members.iter().enumerate() {
            let b = m.birth;
            match first {
                None => first = Some((i, b)),
                Some((_, bf)) if b < bf => {
                    second = first;
                    first = Some((i, b));
                }
                _ => match second {
                    None => {
                        if first.map(|(ifst, _)| ifst) != Some(i) {
                            second = Some((i, b));
                        }
                    }
                    Some((_, bs)) if b < bs && first.map(|(ifst, _)| ifst) != Some(i) => {
                        second = Some((i, b));
                    }
                    _ => {}
                },
            }
        }
        let (i1, _) = first.expect("population must be non-empty");
        let (i2, _) = second.unwrap_or_else(|| panic!("population must have at least two members"));
        (i1, i2)
    }

    pub fn replace_oldest(&mut self, child: PopMember<T, Ops, D>) {
        let idx = self.oldest_index();
        self.members[idx] = child;
    }

    pub fn replace_two_oldest(&mut self, a: PopMember<T, Ops, D>, b: PopMember<T, Ops, D>) {
        let (i1, i2) = self.two_oldest_indices();
        self.members[i1] = a;
        self.members[i2] = b;
    }
}
