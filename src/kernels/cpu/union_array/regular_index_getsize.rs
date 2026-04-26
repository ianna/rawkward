// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the number of distinct tags in a UnionArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_regular_index_getsize.cpp`.

/// Return `max(fromtags) + 1`, i.e. the number of union variants.
///
/// Returns `1` for an empty array (there is always at least one variant).
pub fn union_array_regular_index_getsize<C>(fromtags: &[C]) -> i64
where
    C: Copy + Into<i64>,
{
    let max = fromtags.iter().map(|&t| t.into()).max().unwrap_or(-1);
    max + 1
}

pub fn union_array64_regular_index_getsize(fromtags: &[i64]) -> i64 {
    union_array_regular_index_getsize(fromtags)
}
pub fn union_array8_regular_index_getsize(fromtags: &[i8]) -> i64 {
    union_array_regular_index_getsize(fromtags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_variants() {
        assert_eq!(union_array8_regular_index_getsize(&[0i8, 1, 0, 1]), 2);
    }

    #[test]
    fn three_variants() {
        assert_eq!(union_array8_regular_index_getsize(&[0i8, 2, 1]), 3);
    }

    #[test]
    fn empty_array() {
        // max of empty = -1, so size = 0
        assert_eq!(union_array8_regular_index_getsize(&[]), 0);
    }

    #[test]
    fn single_variant() {
        assert_eq!(union_array8_regular_index_getsize(&[0i8; 5]), 1);
    }
}
