// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Sum all `offsets[i+1] - offsets[i]` in a range-slice offsets array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_getitem_next_range_counts.cpp`.

/// Sum `fromoffsets[i+1] - fromoffsets[i]` for `i` in `0..lenstarts`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_getitem_next_range_counts::list_array64_getitem_next_range_counts_64;
///
/// let offsets = [0i64, 3, 5, 8];
/// assert_eq!(list_array64_getitem_next_range_counts_64(&offsets, 3), 8);
/// ```
pub fn list_array_getitem_next_range_counts<C>(fromoffsets: &[C], lenstarts: usize) -> i64
where
    C: Copy + Into<i64>,
{
    (0..lenstarts)
        .map(|i| fromoffsets[i + 1].into() - fromoffsets[i].into())
        .sum()
}

pub fn list_array32_getitem_next_range_counts_64(fromoffsets: &[i32], lenstarts: usize) -> i64 {
    list_array_getitem_next_range_counts(fromoffsets, lenstarts)
}
pub fn list_array_u32_getitem_next_range_counts_64(fromoffsets: &[u32], lenstarts: usize) -> i64 {
    list_array_getitem_next_range_counts(fromoffsets, lenstarts)
}
pub fn list_array64_getitem_next_range_counts_64(fromoffsets: &[i64], lenstarts: usize) -> i64 {
    list_array_getitem_next_range_counts(fromoffsets, lenstarts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(
            list_array64_getitem_next_range_counts_64(&[0, 3, 5, 8], 3),
            8
        );
    }

    #[test]
    fn partial() {
        assert_eq!(
            list_array64_getitem_next_range_counts_64(&[0, 3, 5, 8], 2),
            5
        );
    }

    #[test]
    fn zero_lenstarts() {
        assert_eq!(list_array64_getitem_next_range_counts_64(&[0, 5], 0), 0);
    }
}
