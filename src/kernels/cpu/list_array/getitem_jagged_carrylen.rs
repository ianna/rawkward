// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the total carry length for a jagged-slice getitem.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_getitem_jagged_carrylen.cpp`.

/// Sum all `slicestops[i] - slicestarts[i]` to get the total number of items
/// that will be carried.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_getitem_jagged_carrylen::list_array_getitem_jagged_carrylen_64;
///
/// let starts = [0i64, 2, 5];
/// let stops  = [2i64, 5, 6];
/// assert_eq!(list_array_getitem_jagged_carrylen_64(&starts, &stops), 6);
/// ```
pub fn list_array_getitem_jagged_carrylen<T>(slicestarts: &[T], slicestops: &[T]) -> i64
where
    T: Copy + Into<i64>,
{
    slicestarts
        .iter()
        .zip(slicestops.iter())
        .map(|(&a, &b)| b.into() - a.into())
        .sum()
}

pub fn list_array_getitem_jagged_carrylen_64(slicestarts: &[i64], slicestops: &[i64]) -> i64 {
    list_array_getitem_jagged_carrylen(slicestarts, slicestops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(
            list_array_getitem_jagged_carrylen_64(&[0, 2, 5], &[2, 5, 6]),
            6
        );
    }

    #[test]
    fn empty_slices() {
        assert_eq!(list_array_getitem_jagged_carrylen_64(&[3, 3], &[3, 3]), 0);
    }

    #[test]
    fn empty_input() {
        assert_eq!(list_array_getitem_jagged_carrylen_64(&[], &[]), 0);
    }
}
