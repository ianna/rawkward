// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apply a range slice to every row of a RegularArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_RegularArray_getitem_next_range.cpp`.

/// For each row `i` and each step `j` in `0..nextsize`:
/// `tocarry[i*nextsize + j] = i*size + regular_start + j*step`
///
/// # Parameters
///
/// * `regular_start` – Already-regularised (non-negative) start offset within a row.
/// * `step`          – Stride between selected elements.
/// * `length`        – Number of rows.
/// * `size`          – Row length of the RegularArray.
/// * `nextsize`      – Number of elements selected per row.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_next_range::regular_array_getitem_next_range_64;
///
/// // 2 rows of size 5; start=1, step=2, nextsize=2 → picks positions 1,3 per row
/// let mut carry = [0i64; 4];
/// regular_array_getitem_next_range_64(&mut carry, 1, 2, 2, 5, 2);
/// assert_eq!(carry, [1, 3, 6, 8]);
/// ```
pub fn regular_array_getitem_next_range_64(
    tocarry: &mut [i64],
    regular_start: i64,
    step: i64,
    length: usize,
    size: i64,
    nextsize: usize,
) {
    assert!(tocarry.len() >= length * nextsize);
    for i in 0..length {
        for j in 0..nextsize {
            tocarry[i * nextsize + j] = i as i64 * size + regular_start + j as i64 * step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut carry = [0i64; 4];
        regular_array_getitem_next_range_64(&mut carry, 1, 2, 2, 5, 2);
        assert_eq!(carry, [1, 3, 6, 8]);
    }

    #[test]
    fn step_one_full_row() {
        // Picks all 3 elements of each of 2 rows
        let mut carry = [0i64; 6];
        regular_array_getitem_next_range_64(&mut carry, 0, 1, 2, 3, 3);
        assert_eq!(carry, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn single_element_per_row() {
        let mut carry = [0i64; 3];
        regular_array_getitem_next_range_64(&mut carry, 2, 1, 3, 4, 1);
        assert_eq!(carry, [2, 6, 10]);
    }

    #[test]
    fn zero_length() {
        let mut carry: [i64; 0] = [];
        regular_array_getitem_next_range_64(&mut carry, 0, 1, 0, 4, 2);
    }
}
