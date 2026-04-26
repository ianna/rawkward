// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Project one content out of a UnionArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_project.cpp`.

/// Collect `fromindex[i]` into `tocarry` for every `i` where `fromtags[i] == which`.
///
/// # Returns
///
/// The number of entries written.
pub fn union_array_project<C, I>(
    tocarry: &mut [i64],
    fromtags: &[C],
    fromindex: &[I],
    which: i64,
) -> usize
where
    C: Copy + Into<i64>,
    I: Copy + Into<i64>,
{
    assert_eq!(fromtags.len(), fromindex.len());
    let mut lenout = 0usize;
    for (&t, &idx) in fromtags.iter().zip(fromindex.iter()) {
        if t.into() == which {
            tocarry[lenout] = idx.into();
            lenout += 1;
        }
    }
    lenout
}

pub fn union_array8_32_project_64(
    tocarry: &mut [i64],
    fromtags: &[i8],
    fromindex: &[i32],
    which: i64,
) -> usize {
    union_array_project(tocarry, fromtags, fromindex, which)
}
pub fn union_array8_u32_project_64(
    tocarry: &mut [i64],
    fromtags: &[i8],
    fromindex: &[u32],
    which: i64,
) -> usize {
    union_array_project(tocarry, fromtags, fromindex, which)
}
pub fn union_array8_64_project_64(
    tocarry: &mut [i64],
    fromtags: &[i8],
    fromindex: &[i64],
    which: i64,
) -> usize {
    union_array_project(tocarry, fromtags, fromindex, which)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let tags = [0i8, 1, 0, 1, 0];
        let index = [10i64, 20, 30, 40, 50];
        let mut carry = [0i64; 5];
        let n = union_array8_64_project_64(&mut carry, &tags, &index, 0);
        assert_eq!(n, 3);
        assert_eq!(&carry[..n], &[10, 30, 50]);
    }

    #[test]
    fn no_match() {
        let tags = [0i8; 4];
        let index = [1i64, 2, 3, 4];
        let mut carry = [0i64; 4];
        let n = union_array8_64_project_64(&mut carry, &tags, &index, 1);
        assert_eq!(n, 0);
    }
}
