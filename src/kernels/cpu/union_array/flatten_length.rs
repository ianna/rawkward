// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the total flattened length of a UnionArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_flatten_length.cpp`.

/// Sum all `offsetsraws[tags[i]][index[i]+1] - offsetsraws[tags[i]][index[i]]`.
///
/// `offsetsraws` is a slice of per-content offset arrays (one per union variant).
pub fn union_array_flatten_length<C, I>(
    fromtags: &[C],
    fromindex: &[I],
    offsetsraws: &[&[i64]],
) -> i64
where
    C: Copy + Into<i64>,
    I: Copy + Into<i64>,
{
    assert_eq!(fromtags.len(), fromindex.len());
    let mut total = 0i64;
    for (&t, &idx) in fromtags.iter().zip(fromindex.iter()) {
        let tag = t.into() as usize;
        let i = idx.into() as usize;
        let offsets = offsetsraws[tag];
        total += offsets[i + 1] - offsets[i];
    }
    total
}

pub fn union_array32_flatten_length_64(
    fromtags: &[i8],
    fromindex: &[i32],
    offsetsraws: &[&[i64]],
) -> i64 {
    union_array_flatten_length(fromtags, fromindex, offsetsraws)
}
pub fn union_array_u32_flatten_length_64(
    fromtags: &[i8],
    fromindex: &[u32],
    offsetsraws: &[&[i64]],
) -> i64 {
    union_array_flatten_length(fromtags, fromindex, offsetsraws)
}
pub fn union_array64_flatten_length_64(
    fromtags: &[i8],
    fromindex: &[i64],
    offsetsraws: &[&[i64]],
) -> i64 {
    union_array_flatten_length(fromtags, fromindex, offsetsraws)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Two contents with offsets [0,2,5] and [0,3].
        // tags=[0,1,0], index=[0,0,1] → lengths 2+3+3 = 8
        let offsets0 = [0i64, 2, 5];
        let offsets1 = [0i64, 3];
        let offsetsraws: &[&[i64]] = &[&offsets0, &offsets1];
        let tags = [0i8, 1, 0];
        let index = [0i64, 0, 1];
        assert_eq!(
            union_array64_flatten_length_64(&tags, &index, offsetsraws),
            8
        );
    }

    #[test]
    fn empty() {
        let offsets0: &[i64] = &[0];
        let offsets1: &[i64] = &[0];
        let offsetsraws: &[&[i64]] = &[offsets0, offsets1];
        let tags: [i8; 0] = [];
        let index: [i64; 0] = [];
        assert_eq!(
            union_array64_flatten_length_64(&tags, &index, offsetsraws),
            0
        );
    }
}
