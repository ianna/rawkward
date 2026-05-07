// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Find the shortest list length in a ListArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_min_range.cpp`.

/// Return the minimum of `stops[i] - starts[i]` over all `i`.
///
/// # Panics
///
/// Panics if `lenstarts == 0`.
pub fn list_array_min_range<C>(fromstarts: &[C], fromstops: &[C]) -> i64
where
    C: Copy + Into<i64>,
{
    assert!(!fromstarts.is_empty(), "lenstarts must be > 0");
    fromstarts
        .iter()
        .zip(fromstops.iter())
        .map(|(&s, &e)| e.into() - s.into())
        .min()
        .unwrap_or(0)
}

pub fn list_array32_min_range(fromstarts: &[i32], fromstops: &[i32]) -> i64 {
    list_array_min_range(fromstarts, fromstops)
}
pub fn list_array_u32_min_range(fromstarts: &[u32], fromstops: &[u32]) -> i64 {
    list_array_min_range(fromstarts, fromstops)
}
pub fn list_array64_min_range(fromstarts: &[i64], fromstops: &[i64]) -> i64 {
    list_array_min_range(fromstarts, fromstops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(list_array64_min_range(&[0, 3, 7], &[3, 5, 8]), 1);
    }

    #[test]
    fn all_same() {
        assert_eq!(list_array64_min_range(&[0, 4], &[4, 8]), 4);
    }

    #[test]
    fn one_element() {
        assert_eq!(list_array64_min_range(&[2], &[5]), 3);
    }
}
